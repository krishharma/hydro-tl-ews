"""Supplementary analyses for the manuscript (added 2026-07-01).

Runs the four items that were blocking publication:
  1. Held-out evaluation of the local-baseline checkpoint (Table 1 row).
  2. Held-out evaluation of the Approach-B progressive checkpoint (Table 1 row).
  3. Seasonal-climatology benchmark for the early-warning metrics
     (day-of-year event frequency from the pre-eval 1990-2010 record),
     including Brier Skill Scores for the model vs climatology.
  4. Zero-clamped predictions: continuous metrics and re-derived drought /
     flood warning skill with predictions clipped at 0 mm/day.

Only the target basin's two CAMELS files are needed; they are extracted from
data/basin_timeseries_v1p2_metForcing_obsFlow.zip on first run if missing.

Outputs (all JSON, all traceable):
  results/baseline_eval_metrics.json
  results/ews_climatology_benchmark.json
  results/ews_clamped_metrics.json

Run:  .venv/Scripts/python.exe scripts/paper_supplement_analysis.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hydro_tl_ews.data.camels import CamelsDataset, STATIC_ATTRIBUTES
from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.evaluation.extreme_thresholds import (
    ExtremeThresholds,
    predicted_warning_probabilities,
    warning_labels,
)
from hydro_tl_ews.evaluation.metrics import (
    auc_roc,
    brier_score,
    f1_at_threshold,
    kge,
    nse,
    pbias,
)
from hydro_tl_ews.training.trainer import Trainer

TARGET = "11264500"
CAMELS_ROOT = ROOT / "data"
RESULTS = ROOT / "results"
EVAL_PERIOD = ("2011-01-01", "2014-12-31")
NORM_PERIOD = ("2009-01-01", "2010-12-31")   # matches zero_shot.yaml (no leakage)
CLIM_PERIOD = ("1990-01-01", "2010-12-31")   # pre-eval record for climatology
SEQ_LEN = 365
LEADS = (1, 3, 7)


# --------------------------------------------------------------- data setup
def ensure_target_extracted() -> None:
    """Extract the target basin's forcing + streamflow files from the CAMELS
    zip if they are not already on disk (only these two files are needed)."""
    base = CAMELS_ROOT / "basin_dataset_public_v1p2"
    have_forcing = list(base.glob(f"basin_mean_forcing/daymet/*/{TARGET}_*.txt"))
    have_flow = list(base.glob(f"usgs_streamflow/*/{TARGET}_*.txt"))
    if have_forcing and have_flow:
        print(f"[setup] target basin files already extracted: "
              f"{have_forcing[0].name}, {have_flow[0].name}")
        return
    zpath = CAMELS_ROOT / "basin_timeseries_v1p2_metForcing_obsFlow.zip"
    if not zpath.exists():
        raise FileNotFoundError(f"CAMELS zip not found: {zpath}")
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist() if TARGET in n and n.endswith(".txt")]
        wanted = [n for n in members
                  if ("basin_mean_forcing/daymet" in n) or ("usgs_streamflow" in n)]
        if not wanted:
            raise RuntimeError(
                f"No forcing/streamflow members for {TARGET} in zip; "
                f"candidates were: {members[:10]}")
        for n in wanted:
            # Rebuild destination under data/ keeping the canonical layout,
            # regardless of any leading folder inside the zip.
            idx = n.find("basin_dataset_public_v1p2")
            rel = n[idx:] if idx >= 0 else f"basin_dataset_public_v1p2/{n}"
            dest = CAMELS_ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(n))
            print(f"[setup] extracted {rel}")


# ------------------------------------------------------ checkpoint evaluation
def evaluate_checkpoint(ckpt_path: Path, target, attrs) -> dict:
    """Mirror of the zero-shot stage: predict 2011-2014 with the warmup-fitted
    normalizer, no fine-tuning, so all Table-1 rows are directly comparable."""
    dyn_norm = Normalizer.fit(target.forcings.loc[NORM_PERIOD[0]:NORM_PERIOD[1]])
    static_norm = StaticNormalizer.fit(attrs.loc[:, STATIC_ATTRIBUTES])
    from torch.utils.data import DataLoader
    eval_ds = MultiBasinSequenceDataset(
        {TARGET: target}, EVAL_PERIOD, dyn_norm, static_norm,
        sequence_length=SEQ_LEN,
    )
    loader = DataLoader(eval_ds, batch_size=256, shuffle=False)
    model = Trainer.load_model(str(ckpt_path))
    trainer = Trainer(model=model, mode="zero_shot", head_lr=1e-3)
    preds, obs = trainer.predict(loader)
    return {
        "NSE": float(nse(obs, preds)),
        "KGE": float(kge(obs, preds)),
        "PBIAS": float(pbias(obs, preds)),
        "n_samples": int(len(preds)),
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "evaluation_period": list(EVAL_PERIOD),
        "normalization_period": list(NORM_PERIOD),
        "note": "no fine-tuning at eval time; identical protocol to zero_shot",
    }


# ------------------------------------------------- climatology EWS benchmark
def climatology_probs(hist_flow: pd.Series, eval_index: pd.DatetimeIndex,
                      thr: float, kind: str) -> pd.DataFrame:
    """Day-of-year climatological event probabilities from the pre-eval
    record, smoothed +/-7 days, composed over lead windows."""
    if kind == "flood":
        event = (hist_flow >= thr).astype(float)
    else:
        event = (hist_flow <= thr).astype(float)
    doy = hist_flow.index.dayofyear.values
    p_doy = np.zeros(367)
    for d in range(1, 367):
        # +/-7-day circular window around each day of year
        lo, hi = d - 7, d + 7
        mask = ((doy >= lo) & (doy <= hi))
        if lo < 1:
            mask |= (doy >= 366 + lo)
        if hi > 366:
            mask |= (doy <= hi - 366)
        sel = event.values[mask]
        p_doy[d] = sel.mean() if len(sel) else 0.0
    daily = pd.Series(p_doy[eval_index.dayofyear.values], index=eval_index)
    log1m = np.log1p(-np.clip(daily, 1e-9, 1 - 1e-9))
    out = pd.DataFrame(index=eval_index)
    for L in LEADS:
        # Match model EWS composition: rolling then shift(-L), full windows only.
        any_event = 1.0 - np.exp(
            log1m.rolling(L, min_periods=L).sum().shift(-L)
        )
        out[f"lead{L}d"] = any_event.fillna(0.0)
    return out


def run_climatology_benchmark(target) -> dict:
    wf = json.loads((RESULTS / "walk_forward_metrics.json").read_text())
    thr = wf["thresholds"]
    warns = pd.read_csv(RESULTS / "walk_forward_warnings.csv",
                        index_col=0, parse_dates=True)
    hist = target.streamflow.loc[CLIM_PERIOD[0]:CLIM_PERIOD[1]].dropna()
    out: dict = {"climatology_record": list(CLIM_PERIOD),
                 "doy_smoothing_days": 15, "benchmarks": {}}
    specs = [("flood", "q95", thr["q95"]), ("flood", "q99", thr["q99"]),
             ("drought", "q5", thr["q5"])]
    for kind, pct, t in specs:
        clim = climatology_probs(hist, warns.index, t, kind)
        for L in LEADS:
            key = f"{kind}_{pct}_lead{L}d"
            y = warns[key].to_numpy()
            p_clim = clim[f"lead{L}d"].to_numpy()
            p_model = warns[f"{key}_prob"].to_numpy()
            b_clim = float(brier_score(y, p_clim))
            b_model = float(brier_score(y, p_model))
            out["benchmarks"][key] = {
                "AUC_climatology": float(auc_roc(y, p_clim)),
                "AUC_model": float(auc_roc(y, p_model)),
                "Brier_climatology": b_clim,
                "Brier_model": b_model,
                "BSS_model_vs_climatology":
                    float(1.0 - b_model / b_clim) if b_clim > 0 else float("nan"),
                "event_base_rate": float(np.mean(y)),
            }
    return out


# ----------------------------------------------------- zero-clamp re-derive
def run_clamped_analysis() -> dict:
    wf = json.loads((RESULTS / "walk_forward_metrics.json").read_text())
    t = wf["thresholds"]
    thr = ExtremeThresholds(q5=t["q5"], q95=t["q95"], q99=t["q99"])
    df = pd.read_parquet(RESULTS / "walk_forward.parquet")
    obs = pd.Series(df["observed"].to_numpy(), index=df.index)
    pred = pd.Series(np.clip(df["predicted"].to_numpy(), 0.0, None),
                     index=df.index)
    out: dict = {
        "note": "walk-forward predictions clipped at 0 mm/day "
                "(21.5% were negative in the stored run)",
        "continuous": {
            "NSE": float(nse(obs.to_numpy(), pred.to_numpy())),
            "KGE": float(kge(obs.to_numpy(), pred.to_numpy())),
            "PBIAS": float(pbias(obs.to_numpy(), pred.to_numpy())),
        },
        "early_warning": {},
    }
    for kind, pct in [("flood", "q95"), ("flood", "q99"), ("drought", "q5")]:
        labels = warning_labels(obs, thr, kind=kind, percentile=pct,
                                lead_times=LEADS)
        probs = predicted_warning_probabilities(pred, thr, kind=kind,
                                                percentile=pct,
                                                lead_times=LEADS)
        for col in labels.columns:
            y = labels[col].to_numpy()
            p = probs[col].to_numpy()
            out["early_warning"][col] = {
                "AUC": float(auc_roc(y, p)),
                "F1@0.5": float(f1_at_threshold(y, p, 0.5)),
                "Brier": float(brier_score(y, p)),
            }
    # drought probability degeneracy check after clamping
    dp = predicted_warning_probabilities(pred, thr, kind="drought",
                                         percentile="q5",
                                         lead_times=(3,))["drought_q5_lead3d"]
    out["drought_prob_distribution_after_clamp"] = {
        "frac_at_floor(<1e-3)": float((dp < 1e-3).mean()),
        "frac_at_one(>0.999999)": float((dp >= 0.999999).mean()),
    }
    return out


def main() -> None:
    ensure_target_extracted()
    ds = CamelsDataset(str(CAMELS_ROOT))
    target = ds.load_basin(TARGET)
    attrs = ds.load_attributes()

    # 1+2 — Table 1 baseline rows
    baselines = {}
    for name, ckpt in [("local_baseline", RESULTS / "checkpoints/local_baseline.pt"),
                       ("finetune_progressive",
                        RESULTS / "checkpoints/finetune_progressive.pt")]:
        print(f"[eval] {name} ...")
        baselines[name] = evaluate_checkpoint(ckpt, target, attrs)
        print(f"[eval] {name}: NSE={baselines[name]['NSE']:.4f} "
              f"KGE={baselines[name]['KGE']:.4f} "
              f"PBIAS={baselines[name]['PBIAS']:.2f}%")
    (RESULTS / "baseline_eval_metrics.json").write_text(
        json.dumps(baselines, indent=2, default=float))

    # 3 — climatology benchmark
    print("[clim] computing day-of-year climatology benchmark ...")
    clim = run_climatology_benchmark(target)
    (RESULTS / "ews_climatology_benchmark.json").write_text(
        json.dumps(clim, indent=2, default=float))
    for k, v in clim["benchmarks"].items():
        print(f"[clim] {k}: model AUC {v['AUC_model']:.3f} vs clim "
              f"{v['AUC_climatology']:.3f} | BSS {v['BSS_model_vs_climatology']:.3f}")

    # 4 — clamped re-derivation
    print("[clamp] re-deriving metrics with predictions clipped at 0 ...")
    clamped = run_clamped_analysis()
    (RESULTS / "ews_clamped_metrics.json").write_text(
        json.dumps(clamped, indent=2, default=float))
    c = clamped["continuous"]
    print(f"[clamp] continuous: NSE={c['NSE']:.4f} KGE={c['KGE']:.4f} "
          f"PBIAS={c['PBIAS']:.2f}%")
    for k, v in clamped["early_warning"].items():
        if k.startswith("drought"):
            print(f"[clamp] {k}: AUC {v['AUC']:.3f} | Brier {v['Brier']:.3f}")

    print("[done] wrote results/baseline_eval_metrics.json, "
          "results/ews_climatology_benchmark.json, "
          "results/ews_clamped_metrics.json")


if __name__ == "__main__":
    main()
