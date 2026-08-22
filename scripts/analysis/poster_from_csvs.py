#!/usr/bin/env python3
"""Poster metrics and figures from existing midwest_mini CSVs (no retraining).

Computes:
  - next-day persistence and day-of-year climatology (climatology fit on the
    2009--2010 warmup only, matching the scarce-data protocol)
  - Brier skill score vs the event base rate
  - F1 across decision thresholds
  - reliability curves
  - five poster figures

Usage (repo root)::

    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    python scripts/analysis/poster_from_csvs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hydro_tl_ews.data.camels import STATIC_ATTRIBUTES, CamelsDataset
from hydro_tl_ews.data.clustering import select_donor_basins
from hydro_tl_ews.evaluation.metrics import (
    auc_roc,
    brier_score,
    f1_at_threshold,
    kge,
    nse,
    pbias,
    reliability_curve,
)
from stages.pretrain_stage import (
    exclude_targets_and_buffer,
    filter_donors_by_bbox,
)

MINI = ROOT / "results" / "midwest_mini"
OPT_ENS = ROOT / "results" / "midwest_opt" / "ensemble"
ABLATE = ROOT / "results" / "midwest_opt_ablate" / "prune_fulltemp"
OUT = MINI / "poster"
FIG = ROOT / "docs" / "paper_images"
TARGET = "05507600"
EVAL = ("2011-01-01", "2014-12-31")
WARMUP = ("2009-01-01", "2010-12-31")
BBOX = [36.5, 49.5, -104.0, -80.5]

TEAL = "#01696F"
TEAL_DARK = "#0C4E54"
CORAL = "#C45C26"
SLATE = "#5C5850"
GOLD = "#C4A35A"
INK = "#1F1E1B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#D4D0C8",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.titlecolor": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#EEECE6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df.sort_index()


def brier_skill_score(y: np.ndarray, p: np.ndarray) -> float:
    """BSS vs climatological base-rate forecast (mean of y)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return float("nan")
    bs = float(np.mean((p - y) ** 2))
    clim = float(y.mean())
    bs_ref = float(np.mean((clim - y) ** 2))
    if bs_ref <= 0:
        return float("nan")
    return 1.0 - bs / bs_ref


def best_f1(y: np.ndarray, p: np.ndarray) -> dict:
    thresholds = np.linspace(0.05, 0.95, 19)
    scores = [f1_at_threshold(y, p, float(t)) for t in thresholds]
    scores = np.array(scores, dtype=float)
    if np.all(np.isnan(scores)):
        return {"F1_best": float("nan"), "threshold": float("nan")}
    i = int(np.nanargmax(scores))
    return {"F1_best": float(scores[i]), "threshold": float(thresholds[i])}


def reconstruct_donors(n_request: int = 20) -> list[str]:
    ds = CamelsDataset(ROOT / "data")
    attrs = ds.load_attributes()
    pool = filter_donors_by_bbox(attrs, list(attrs.index), BBOX)
    attr_for_sim = attrs.loc[pool, STATIC_ATTRIBUTES]
    if TARGET not in attr_for_sim.index:
        attr_for_sim = pd.concat(
            [attr_for_sim, attrs.loc[[TARGET], STATIC_ATTRIBUTES]])
    donors = select_donor_basins(attr_for_sim, TARGET, n_request)
    donors = [b for b in donors if b in set(pool)]
    donors = exclude_targets_and_buffer(attrs, donors, [TARGET], 50)
    available = []
    for bid in donors:
        try:
            ds._find_forcing_file(bid)
            ds._find_streamflow_file(bid)
            available.append(bid)
        except FileNotFoundError:
            continue
    return available


def naive_baselines(wf: pd.DataFrame, q_hist: pd.Series) -> dict:
    obs = wf["observed"].astype(float)
    # Persistence: Q(t) <- Q(t-1). Seed the first eval day from 2010-12-31.
    prior = q_hist.loc[:EVAL[0]].iloc[:-1]
    yesterday = pd.concat([prior.iloc[[-1]], obs]).shift(1).loc[obs.index]
    persist = yesterday.to_numpy()
    # Day-of-year climatology from warmup only (scarce-data protocol).
    warm = q_hist.loc[WARMUP[0]:WARMUP[1]].dropna()
    doy_mean = warm.groupby(warm.index.dayofyear).mean()
    clim = obs.index.dayofyear.map(lambda d: float(doy_mean.get(d, warm.mean())))
    clim = np.asarray(clim, dtype=float)
    y = obs.to_numpy()
    return {
        "persistence": {
            "NSE": nse(y, persist),
            "KGE": kge(y, persist),
            "PBIAS": pbias(y, persist),
        },
        "doy_climatology": {
            "NSE": nse(y, clim),
            "KGE": kge(y, clim),
            "PBIAS": pbias(y, clim),
            "fit_period": list(WARMUP),
        },
        "series": {
            "observed": y,
            "persistence": persist,
            "climatology": clim,
        },
    }


def _summarize_probs(y: np.ndarray, p: np.ndarray) -> dict:
    centers, freq, counts = reliability_curve(y, p, n_bins=10)
    return {
        "base_rate": float(np.nanmean(y)),
        "mean_prob": float(np.nanmean(p)),
        "AUC": auc_roc(y, p),
        "Brier": brier_score(y, p),
        "BSS": brier_skill_score(y, p),
        "F1@0.5": f1_at_threshold(y, p, 0.5),
        **best_f1(y, p),
        "reliability": {
            "centers": centers.tolist(),
            "observed_freq": [None if np.isnan(v) else float(v) for v in freq],
            "counts": counts.tolist(),
        },
    }


def _window_max_from_lead1(p1: np.ndarray, lead: int) -> np.ndarray:
    """P(event in next L days) ≈ max of successive 1-day probabilities.

    Avoids the independence product ``1-∏(1-p_i)``, which drives multi-day
    probabilities toward 1 on this flashy basin.
    """
    out = np.full_like(p1, np.nan, dtype=float)
    for i in range(len(p1) - lead + 1):
        out[i] = np.nanmax(p1[i:i + lead])
    return out


def ews_calibration(warn: pd.DataFrame) -> dict:
    events = {}
    p1_by_family = {}
    for col in warn.columns:
        if col.endswith("_lead1d_prob"):
            p1_by_family[col.replace("_lead1d_prob", "")] = warn[col].to_numpy(float)
    for col in warn.columns:
        if col.endswith("_prob"):
            continue
        pcol = f"{col}_prob"
        if pcol not in warn.columns:
            continue
        y = warn[col].to_numpy(dtype=float)
        p = warn[pcol].to_numpy(dtype=float)
        rec = _summarize_probs(y, p)
        rec["method"] = "independence_product"
        # Window-max alternative for multi-day leads.
        if "_lead" in col:
            family, _, lead_s = col.rpartition("_lead")
            lead = int(lead_s.replace("d", ""))
            if family in p1_by_family and lead >= 1:
                pmax = _window_max_from_lead1(p1_by_family[family], lead)
                rec["window_max"] = _summarize_probs(y, pmax)
                rec["window_max"]["method"] = "window_max_of_1d_probs"
        events[col] = rec
    return events


def skill_row(path: Path) -> dict:
    df = _load_pred(path)
    y, p = df["observed"].to_numpy(float), df["predicted"].to_numpy(float)
    return {"NSE": nse(y, p), "KGE": kge(y, p), "PBIAS": pbias(y, p)}


def _save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    p1 = OUT / name
    p2 = FIG / name
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p1}")
    return p2


def fig_donor_map(donors: list[str], attrs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    # Light Midwest context: other gauges in bbox, faded.
    lat_ok = (attrs["gauge_lat"] >= BBOX[0]) & (attrs["gauge_lat"] <= BBOX[1])
    lon_ok = (attrs["gauge_lon"] >= BBOX[2]) & (attrs["gauge_lon"] <= BBOX[3])
    ctx = attrs.loc[lat_ok & lon_ok]
    ax.scatter(ctx["gauge_lon"], ctx["gauge_lat"], s=10, c="#E4E0D6",
               linewidths=0, zorder=1, label="other Midwest CAMELS gauges")
    d = attrs.loc[donors]
    ax.scatter(d["gauge_lon"], d["gauge_lat"], s=48, c=TEAL, zorder=3,
               edgecolors="white", linewidths=0.6, label=f"donors (n={len(donors)})")
    t = attrs.loc[TARGET]
    ax.scatter([t["gauge_lon"]], [t["gauge_lat"]], marker="*", s=280,
               c=CORAL, edgecolors=INK, linewidths=0.7, zorder=4,
               label="target: Lick Creek, MO")
    ax.annotate("Lick Creek\n(USGS 05507600)",
                (t["gauge_lon"], t["gauge_lat"]),
                textcoords="offset points", xytext=(10, -22),
                fontsize=8.5, color=TEAL_DARK, fontweight="medium")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Seventeen similar Midwest donors and the Lick Creek target")
    ax.legend(loc="lower left", frameon=True, fontsize=8,
              fancybox=False, edgecolor="#E4E0D6")
    ax.set_aspect(1.2)
    fig.tight_layout()
    _save(fig, "fig_donor_target_map.png")


def fig_hydrograph(wf: pd.DataFrame, local: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6),
                             gridspec_kw={"height_ratios": [1.15, 1]})
    for ax in axes:
        ax.plot(wf.index, wf["observed"], color=INK, lw=0.9, label="Observed", zorder=3)
        ax.plot(wf.index, np.clip(wf["predicted"], 0, None), color=TEAL, lw=1.1,
                alpha=0.9, label="Walk-forward transfer", zorder=2)
        ax.plot(local.index, np.clip(local["predicted"], 0, None), color=CORAL,
                lw=0.9, alpha=0.75, label="Local from scratch", zorder=1)
        ax.set_ylabel("mm d$^{-1}$")
    axes[0].set_title("Lick Creek, 2011–2014: observed, walk-forward, and local-from-scratch")
    axes[0].legend(loc="upper right", fontsize=8, frameon=True, fancybox=False,
                   edgecolor="#E4E0D6")
    zoom = slice("2013-03-01", "2013-07-01")
    axes[1].set_xlim(pd.Timestamp("2013-03-01"), pd.Timestamp("2013-07-01"))
    ymax = float(wf.loc[zoom, "observed"].max()) * 1.08
    axes[1].set_ylim(0, ymax)
    axes[1].set_title("Zoom on spring 2013: transfer tracks event timing; local does not")
    fig.tight_layout()
    _save(fig, "fig_hydrograph_2011_2014.png")


def fig_skill_bars(rows: dict) -> None:
    order = [
        ("Persistence", rows["persistence"]),
        ("Local from scratch", rows["local"]),
        ("Zero-shot", rows["zero_shot"]),
        ("Fine-tune (head)", rows["finetune"]),
        ("Walk-forward", rows["walk_forward"]),
    ]
    labels = [k for k, _ in order]
    nse_v = [v["NSE"] for _, v in order]
    kge_v = [v["KGE"] for _, v in order]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(x - w / 2, nse_v, w, color=TEAL, label="NSE")
    ax.bar(x + w / 2, kge_v, w, color=GOLD, label="KGE")
    ax.axhline(0, color=SLATE, lw=0.8)
    ax.axhline(rows["persistence"]["NSE"], color=SLATE, ls=":", lw=0.9,
               label=f"Persistence NSE = {rows['persistence']['NSE']:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Skill")
    ax.set_ylim(-0.5, 0.55)
    clim_nse = rows["doy_climatology"]["NSE"]
    ax.set_title("Continuous skill on 2011–2014 vs naive and local baselines")
    ax.text(0.02, 0.04,
            f"Day-of-year climatology (warmup 2009–2010): NSE = {clim_nse:.2f} (off scale)",
            transform=ax.transAxes, fontsize=8, color=SLATE)
    ax.legend(frameon=True, fancybox=False, edgecolor="#E4E0D6", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_skill_bars.png")


def fig_donor_compare(mini: dict, prune: dict, ens: dict) -> None:
    labels = ["Fine-tune NSE", "Walk-forward NSE"]
    series = {
        "17 donors (Tmin+Tmax)": [mini["finetune"]["NSE"], mini["walk_forward"]["NSE"]],
        "8 donors (Tmin+Tmax)": [prune["finetune"]["NSE"], prune["walk_forward"]["NSE"]],
        "8 donors (mean T, 3-seed)": [ens["finetune"]["NSE"], ens["walk_forward"]["NSE"]],
    }
    colors = [TEAL, SLATE, CORAL]
    x = np.arange(len(labels))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1) * w, vals, w, color=colors[i], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("NSE")
    ax.set_ylim(0, 0.25)
    ax.set_title("Donor pruning did not improve Lick Creek skill")
    ax.legend(fontsize=8, frameon=True, fancybox=False, edgecolor="#E4E0D6")
    fig.tight_layout()
    _save(fig, "fig_donor_pruning.png")


def _rel_xy(rel: dict):
    c = np.array(rel["centers"])
    f = np.array([np.nan if v is None else v for v in rel["observed_freq"]], float)
    n = np.array(rel["counts"], float)
    m = n > 0
    return c[m], f[m], n[m]


def fig_reliability(ews: dict) -> None:
    keys = [
        ("flood_q95_lead1d", "Flood Q95, 1-day lead"),
        ("flood_q95_lead3d", "Flood Q95, 3-day lead"),
        ("flood_q95_lead7d", "Flood Q95, 7-day lead"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=True)
    for ax, (key, title) in zip(axes, keys):
        rec = ews[key]
        ax.plot([0, 1], [0, 1], color=SLATE, lw=0.9, ls="--")
        c, f, n = _rel_xy(rec["reliability"])
        ax.scatter(c, f, s=18 + 70 * np.sqrt(n / max(n.max(), 1)),
                   c=CORAL, zorder=3, edgecolors="white",
                   label="Independence product")
        ax.plot(c, f, color=CORAL, lw=1.0, zorder=2)
        bss = rec["BSS"]
        extra = ""
        if "window_max" in rec:
            c2, f2, n2 = _rel_xy(rec["window_max"]["reliability"])
            ax.scatter(c2, f2, s=18 + 70 * np.sqrt(n2 / max(n2.max(), 1)),
                       c=TEAL, zorder=4, edgecolors="white",
                       label="Window max (no independence)")
            ax.plot(c2, f2, color=TEAL, lw=1.1, zorder=3)
            extra = f"; window-max BSS = {rec['window_max']['BSS']:.2f}"
        ax.set_title(f"{title}\nproduct BSS = {bss:.2f}{extra}", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Forecast probability")
        ax.set_aspect("equal")
        if ax is axes[0]:
            ax.legend(fontsize=6.5, loc="upper left", frameon=True,
                      fancybox=False, edgecolor="#E4E0D6")
    axes[0].set_ylabel("Observed frequency")
    fig.suptitle("Reliability of flood Q95 warnings", y=1.03, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig_reliability_q95.png")


def main() -> int:
    wf = _load_pred(MINI / "walk_forward.csv")
    local = _load_pred(MINI / "local_eval_predictions.csv")
    zs = _load_pred(MINI / "zero_shot_predictions.csv")
    ft = _load_pred(MINI / "finetune_eval_predictions.csv")
    warn = pd.read_csv(MINI / "walk_forward_warnings.csv", index_col=0, parse_dates=True)

    ds = CamelsDataset(ROOT / "data")
    q = ds.load_streamflow(TARGET)
    attrs = ds.load_attributes()
    donors17 = reconstruct_donors(20)
    donors8 = json.loads((ABLATE / "summary.json").read_text())["donors"]["donors"]

    naive = naive_baselines(wf, q)
    ews = ews_calibration(warn)

    mini_rows = {
        "zero_shot": skill_row(MINI / "zero_shot_predictions.csv"),
        "finetune": skill_row(MINI / "finetune_eval_predictions.csv"),
        "local": skill_row(MINI / "local_eval_predictions.csv"),
        "walk_forward": skill_row(MINI / "walk_forward.csv"),
        "persistence": naive["persistence"],
        "doy_climatology": naive["doy_climatology"],
    }
    prune_rows = {
        "finetune": skill_row(ABLATE / "finetune_eval_predictions.csv"),
        "walk_forward": skill_row(ABLATE / "walk_forward.csv"),
        "local": skill_row(ABLATE / "local_eval_predictions.csv"),
    }
    ens_rows = {
        "finetune": skill_row(OPT_ENS / "finetune_eval_predictions.csv"),
        "walk_forward": skill_row(OPT_ENS / "walk_forward.csv"),
        "local": skill_row(OPT_ENS / "local_eval_predictions.csv"),
    }

    summary = {
        "target_basin": TARGET,
        "evaluation_period": list(EVAL),
        "n_eval_days": int(len(wf)),
        "donors_17": donors17,
        "donors_8": donors8,
        "skill": mini_rows,
        "pruning": {
            "17_fulltemp": mini_rows,
            "8_fulltemp": prune_rows,
            "8_meanT_ensemble": ens_rows,
        },
        "early_warning": ews,
        "notes": {
            "persistence": "Q(t) = observed Q(t-1) on the 2011–2014 window.",
            "climatology": "Day-of-year mean of observed Q during warmup 2009–2010 only.",
            "BSS": "1 − BS / BS_clim, where BS_clim uses the event base rate.",
            "F1_best": "Maximum F1 over thresholds 0.05–0.95 on the scored window (diagnostic).",
            "window_max": "Multi-day P ≈ max of 1-day probabilities in [t+1, t+L]; drops the independence product.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "poster_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {OUT / 'poster_metrics.json'}")
    print("NSE persistence", mini_rows["persistence"]["NSE"],
          "climatology", mini_rows["doy_climatology"]["NSE"],
          "walk-forward", mini_rows["walk_forward"]["NSE"])
    for k in ["flood_q95_lead1d", "flood_q95_lead3d", "flood_q95_lead7d",
              "drought_q5_lead1d"]:
        e = ews[k]
        f1 = e["F1@0.5"]
        f1s = "nan" if f1 != f1 else f"{f1:.3f}"
        extra = ""
        if "window_max" in e:
            w = e["window_max"]
            extra = (f" | maxP Brier={w['Brier']:.3f} BSS={w['BSS']:.3f} "
                     f"F1@0.5={w.get('F1@0.5')}")
        print(f"{k}: AUC={e['AUC']:.3f} Brier={e['Brier']:.3f} BSS={e['BSS']:.3f} "
              f"F1@0.5={f1s} F1*={e['F1_best']:.3f}@{e['threshold']:.2f}{extra}")

    fig_donor_map(donors17, attrs)
    fig_hydrograph(wf, local)
    fig_skill_bars(mini_rows)
    fig_donor_compare(mini_rows, prune_rows, ens_rows)
    fig_reliability(ews)
    _ = (zs, ft)  # loaded to confirm files exist
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
