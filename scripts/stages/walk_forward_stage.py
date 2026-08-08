"""Phase 3 — Rolling-origin walk-forward evaluation on the target basin."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydro_tl_ews.data.camels import STATIC_ATTRIBUTES
from hydro_tl_ews.data.features import open_camels
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.evaluation.extreme_thresholds import (
    predicted_warning_probabilities,
    regional_thresholds,
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
from hydro_tl_ews.training.transfer import FineTuneConfig, fine_tune_progressive
from hydro_tl_ews.training.walk_forward import WalkForwardConfig, walk_forward
from hydro_tl_ews.utils.config import ExperimentConfig
from hydro_tl_ews.utils.logging import get_logger
from hydro_tl_ews.xai.shap_analysis import explain_predictions, global_importance

log = get_logger(__name__)


def _warmup_residual_sigma(model, target, dyn_norm, static_norm, wf_cfg,
                           train_start, train_end) -> float | None:
    """Estimate Gaussian residual σ on the warmup window for EWS mapping."""
    from hydro_tl_ews.training.walk_forward import _build_loader, _predict_window

    start = pd.Timestamp(train_start) if train_start else target.forcings.index.min()
    end = pd.Timestamp(train_end)
    loader, dates = _build_loader(
        target.forcings.loc[start:end],
        target.streamflow.loc[start:end],
        target.attributes, dyn_norm, static_norm, wf_cfg, shuffle=False,
    )
    if loader is None or dates is None or len(dates) < 30:
        return None
    preds = _predict_window(model, loader)
    obs = target.streamflow.reindex(pd.DatetimeIndex(dates)).to_numpy()
    resid = obs - preds
    resid = resid[~np.isnan(resid)]
    if len(resid) < 30:
        return None
    return max(float(np.std(resid)), 1e-3)


def run_walk_forward(cfg: ExperimentConfig) -> None:
    ds = open_camels(cfg)
    target_id = cfg.data["target_basin"]
    target = ds.load_basin(target_id)
    attrs = ds.load_attributes()

    full_period = cfg.data.get("full_period")
    if full_period:
        target.forcings = target.forcings.loc[full_period[0]:full_period[1]]
        target.streamflow = target.streamflow.loc[full_period[0]:full_period[1]]

    init_end = cfg.walk_forward["initial_train_end"]
    if not cfg.walk_forward.get("refit_train_start"):
        log.warning(
            "walk_forward.refit_train_start is unset — refits may train on the "
            "full loaded record and break the data-scarce protocol. "
            "Set it to the warmup start (e.g. 2009-01-01)."
        )
    dyn_norm = Normalizer.fit(target.forcings.loc[:init_end])
    static_norm = StaticNormalizer.fit(attrs.loc[:, STATIC_ATTRIBUTES])

    ckpt = cfg.model.get("pretrained_checkpoint")
    if not ckpt:
        raise ValueError("walk_forward stage requires model.pretrained_checkpoint")
    model = Trainer.load_model(ckpt)

    ft = cfg.walk_forward.get("fine_tune", {})
    wf_cfg = WalkForwardConfig(
        initial_train_end=init_end,
        eval_end=cfg.walk_forward["eval_end"],
        refit_every_days=cfg.walk_forward.get("refit_every_days", 90),
        online_bias_correction=cfg.walk_forward.get("online_bias_correction", True),
        sequence_length=cfg.data.get("sequence_length", 365),
        batch_size=64,
        val_tail_days=cfg.walk_forward.get("val_tail_days", 90),
        refit_train_start=cfg.walk_forward.get("refit_train_start"),
        fine_tune_cfg=FineTuneConfig(
            head_lr=ft.get("head_lr", 1e-3),
            lstm_lr=ft.get("lstm_lr", 1e-5),
            epochs_head_only=ft.get("epochs_head_only", 3),
            epochs_progressive=ft.get("epochs_progressive", 0),
            patience=ft.get("patience", 2),
            unfreeze_fraction=ft.get("unfreeze_fraction", 0.0),
        ),
    )
    # Approach selection: "conservative" (default, head-only refits) or
    # "progressive" (Approach B: partial LSTM unfreeze at each refit).
    approach = cfg.walk_forward.get("approach", "conservative")
    refit_fn = fine_tune_progressive if approach == "progressive" else None
    result = walk_forward(model, target, dyn_norm, static_norm, wf_cfg,
                          refit_fn=refit_fn)

    # Thresholds from the pre-evaluation record only (avoid eval-period leakage).
    threshold_series = target.streamflow.loc[:init_end]
    rfa = regional_thresholds(threshold_series, years_required=20)
    obs_s = pd.Series(result.observed, index=result.dates)
    pred_s = pd.Series(result.predicted, index=result.dates)

    # Residual σ from warmup hindcast with the loaded checkpoint (before WF
    # adaptation). Avoids the default 0.25·Q5 collapse for drought probs.
    ews_sigma = _warmup_residual_sigma(
        model, target, dyn_norm, static_norm, wf_cfg,
        train_start=cfg.walk_forward.get("refit_train_start"),
        train_end=init_end,
    )
    if ews_sigma is not None:
        log.info("EWS residual sigma from warmup: %.4f", ews_sigma)
    else:
        log.warning("Could not estimate warmup residual sigma; using default "
                    "0.25·threshold (drought probs may saturate).")

    lead_times = tuple(cfg.evaluation.get("lead_times", [1, 3, 7]))
    threshold_specs = cfg.evaluation.get(
        "threshold_specs",
        [
            {"kind": "flood", "percentile": "q95"},
            {"kind": "flood", "percentile": "q99"},
            {"kind": "drought", "percentile": "q5"},
        ],
    )

    early_warning_metrics: dict[str, dict[str, float]] = {}
    warning_artifacts = []
    for spec in threshold_specs:
        kind = spec["kind"]
        percentile = spec["percentile"]
        labels = warning_labels(obs_s, rfa, kind=kind, percentile=percentile,
                                lead_times=lead_times)
        probs = predicted_warning_probabilities(
            pred_s, rfa, kind=kind, percentile=percentile,
            sigma=ews_sigma, lead_times=lead_times,
        )
        warning_artifacts.append(labels)
        warning_artifacts.append(probs.add_suffix("_prob"))
        for col in labels.columns:
            early_warning_metrics[col] = {
                "AUC": auc_roc(labels[col].to_numpy(), probs[col].to_numpy()),
                "F1@0.5": f1_at_threshold(labels[col].to_numpy(),
                                          probs[col].to_numpy(), 0.5),
                "Brier": brier_score(labels[col].to_numpy(), probs[col].to_numpy()),
            }

    metrics = {
        "continuous": {
            "NSE": nse(result.observed, result.predicted),
            "KGE": kge(result.observed, result.predicted),
            "PBIAS": pbias(result.observed, result.predicted),
        },
        "thresholds": {"q5": rfa.q5, "q95": rfa.q95, "q99": rfa.q99},
        "early_warning": early_warning_metrics,
        "n_predictions": int(len(result.predicted)),
        "n_refits": int(len(result.refit_dates)),
    }
    out_metrics = Path(cfg.output.get("metrics_path",
                                      "results/walk_forward_metrics.json"))
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, default=float))

    out_df = pd.DataFrame({
        "observed": result.observed,
        "predicted": result.predicted,
        "bias_correction": result.bias_corrections,
    }, index=result.dates)
    out_path = Path(cfg.output.get("results_path", "results/walk_forward.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path) if out_path.suffix == ".parquet" else out_df.to_csv(out_path)
    warnings_path = Path(cfg.output.get("warnings_path", "results/walk_forward_warnings.csv"))
    warnings_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(warning_artifacts, axis=1).to_csv(warnings_path)

    if bool(cfg.xai.get("enabled", False)):
        bg_size = int(cfg.xai.get("background_size", 200))
        smp_size = int(cfg.xai.get("sample_size", 100))
        seq_len = cfg.data.get("sequence_length", 365)
        f_hist = target.forcings.loc[:cfg.walk_forward["initial_train_end"]]
        q_hist = target.streamflow.loc[:cfg.walk_forward["initial_train_end"]]
        from hydro_tl_ews.data.preprocessing import (
            align_forcing_streamflow,
            make_sequences,
            quality_control,
        )
        f_hist, q_hist = quality_control(f_hist, q_hist)
        f_hist, q_hist = align_forcing_streamflow(f_hist, q_hist)
        dyn_cols = [c for c in dyn_norm.mean.index if c in f_hist.columns] or list(f_hist.columns)
        f_norm = dyn_norm.transform(f_hist)[dyn_cols]
        X_all, _ = make_sequences(f_norm.to_numpy(), q_hist.to_numpy(), sequence_length=seq_len)
        if len(X_all) >= 2:
            statics = static_norm.transform(target.attributes.to_frame().T).reindex(
                columns=STATIC_ATTRIBUTES).to_numpy().astype(np.float32)
            S_all = np.tile(statics[0], (len(X_all), 1))
            bg = min(bg_size, len(X_all))
            smp = min(smp_size, len(X_all))
            feat_names = [f"dyn_{k}" for k in dyn_cols] + [f"static_{k}" for k in STATIC_ATTRIBUTES]
            shap_result = explain_predictions(
                model,
                background_X=X_all[:bg],
                background_S=S_all[:bg],
                samples_X=X_all[-smp:],
                samples_S=S_all[-smp:],
                feature_names=feat_names,
            )
            if shap_result is not None:
                imp = global_importance(shap_result)
                shap_out = Path(cfg.output.get("shap_importance_path",
                                               "results/shap_global_importance.csv"))
                shap_out.parent.mkdir(parents=True, exist_ok=True)
                imp.rename("mean_abs_shap").to_csv(shap_out, header=True)
            else:
                log.warning("SHAP enabled but explainability run returned no result.")

    log.info("Walk-forward complete | metrics: %s", metrics)
