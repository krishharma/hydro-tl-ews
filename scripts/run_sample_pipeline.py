#!/usr/bin/env python3
"""End-to-end quick test on the mini sample CAMELS archive.

Creates ``data/sample_camels/`` if missing, then runs a shortened
pretrain → fine-tune → walk-forward path through the *real* ``CamelsDataset``
loader (not the synthetic smoke path). Typical runtime: a few minutes on
CPU/MPS.

Usage::

    python scripts/run_sample_pipeline.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hydro_tl_ews.data.camels import (  # noqa: E402
    DYNAMIC_FEATURES,
    STATIC_ATTRIBUTES,
    CamelsDataset,
)
from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset  # noqa: E402
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer  # noqa: E402
from hydro_tl_ews.evaluation.extreme_thresholds import (  # noqa: E402
    predicted_warning_probabilities,
    regional_thresholds,
    warning_labels,
)
from hydro_tl_ews.evaluation.metrics import (  # noqa: E402
    auc_roc,
    brier_score,
    f1_at_threshold,
    kge,
    nse,
    pbias,
)
from hydro_tl_ews.models.ealstm import EALSTM, EALSTMConfig  # noqa: E402
from hydro_tl_ews.training.trainer import Trainer  # noqa: E402
from hydro_tl_ews.training.transfer import FineTuneConfig, fine_tune_conservative  # noqa: E402
from hydro_tl_ews.training.walk_forward import WalkForwardConfig, walk_forward  # noqa: E402
from hydro_tl_ews.utils.logging import get_logger  # noqa: E402
from hydro_tl_ews.utils.seed import set_global_seed  # noqa: E402

log = get_logger("sample_pipeline")

SAMPLE_ROOT = ROOT / "data" / "sample_camels"
OUT = ROOT / "results" / "sample"
TARGET = "11264500"
SEQ_LEN = 90
HIDDEN = 32
BATCH = 64


def ensure_sample() -> None:
    marker = SAMPLE_ROOT / "basin_dataset_public_v1p2"
    if marker.is_dir():
        return
    log.info("Sample dataset missing — generating under %s", SAMPLE_ROOT)
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_sample_camels.py")])


def main() -> int:
    set_global_seed(7)
    ensure_sample()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)

    ds = CamelsDataset(SAMPLE_ROOT)
    attrs = ds.load_attributes()
    donors = [b for b in attrs.index if b != TARGET]
    log.info("Loaded sample CAMELS: %d basins (target=%s, donors=%d)",
             len(attrs), TARGET, len(donors))

    basins = ds.load_basins(donors)
    target = ds.load_basin(TARGET)

    pretrain_period = ("1995-01-01", "2004-12-31")
    val_period = ("2005-01-01", "2006-12-31")
    warmup = ("2005-01-01", "2006-08-31")
    warmup_val = ("2005-09-01", "2006-12-31")

    forc_train = pd.concat([b.forcings.loc[pretrain_period[0]:pretrain_period[1]]
                            for b in basins.values()])
    dyn_norm = Normalizer.fit(forc_train)
    static_norm = StaticNormalizer.fit(attrs.loc[donors, STATIC_ATTRIBUTES])

    train_ds = MultiBasinSequenceDataset(
        basins, pretrain_period, dyn_norm, static_norm, sequence_length=SEQ_LEN)
    val_ds = MultiBasinSequenceDataset(
        basins, val_period, dyn_norm, static_norm, sequence_length=SEQ_LEN)
    log.info("Pretrain samples: train=%d val=%d", len(train_ds), len(val_ds))

    model = EALSTM(EALSTMConfig(
        dynamic_input_size=len(DYNAMIC_FEATURES),
        static_input_size=len(STATIC_ATTRIBUTES),
        hidden_size=HIDDEN,
        dropout=0.2,
    ))
    trainer = Trainer(model=model, mode="pretrain", head_lr=1e-3)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    state = trainer.fit(train_loader, val_loader, epochs=4, patience=3)
    ckpt = OUT / "checkpoints" / "pretrain.pt"
    torch.save({"model_state": model.state_dict(), "config": model.cfg.__dict__}, ckpt)
    log.info("Saved %s | best_val=%.4f", ckpt, state.best_val_loss)

    # Fine-tune on target warmup
    ft_dyn = Normalizer.fit(target.forcings.loc[warmup[0]:warmup[1]])
    ft_static = StaticNormalizer.fit(attrs.loc[:, STATIC_ATTRIBUTES])
    warm_ds = MultiBasinSequenceDataset(
        {TARGET: target}, warmup, ft_dyn, ft_static, sequence_length=SEQ_LEN)
    warm_val = MultiBasinSequenceDataset(
        {TARGET: target}, warmup_val, ft_dyn, ft_static, sequence_length=SEQ_LEN)
    model = Trainer.load_model(ckpt)
    ft_state = fine_tune_conservative(
        model,
        DataLoader(warm_ds, batch_size=32, shuffle=True),
        DataLoader(warm_val, batch_size=32, shuffle=False) if len(warm_val) else None,
        FineTuneConfig(epochs_head_only=3, patience=2, head_lr=1e-3),
    )
    ft_ckpt = OUT / "checkpoints" / "finetune.pt"
    torch.save({"model_state": model.state_dict(), "config": model.cfg.__dict__}, ft_ckpt)
    log.info("Fine-tune done | best=%.4f", ft_state.best_val_loss)

    # Walk-forward on 2007–2009
    model = Trainer.load_model(ft_ckpt)
    # Keep long record available for thresholds; WF only trains from warmup start.
    wf_cfg = WalkForwardConfig(
        initial_train_end="2006-12-31",
        eval_end="2009-12-31",
        refit_every_days=90,
        online_bias_correction=True,
        sequence_length=SEQ_LEN,
        batch_size=32,
        val_tail_days=60,
        refit_train_start="2005-01-01",
        fine_tune_cfg=FineTuneConfig(epochs_head_only=2, patience=1, head_lr=1e-3),
    )
    # Norms from pre-eval forcings only
    wf_dyn = Normalizer.fit(target.forcings.loc[:"2006-12-31"])
    result = walk_forward(model, target, wf_dyn, ft_static, wf_cfg, device=None)

    # Thresholds on pre-eval years (sample has ~12 yr → require 10)
    rfa = regional_thresholds(target.streamflow.loc[:"2006-12-31"], years_required=10)
    obs_s = pd.Series(result.observed, index=result.dates)
    pred_s = pd.Series(result.predicted, index=result.dates)
    labels = warning_labels(obs_s, rfa, kind="flood", percentile="q95", lead_times=(1, 3))
    # residual sigma from warmup predictions
    resid = result.observed - result.predicted
    resid = resid[~np.isnan(resid)]
    sigma = max(float(np.std(resid[: min(200, len(resid))])), 1e-3) if len(resid) else None
    probs = predicted_warning_probabilities(
        pred_s, rfa, kind="flood", percentile="q95", sigma=sigma, lead_times=(1, 3))

    metrics = {
        "continuous": {
            "NSE": nse(result.observed, result.predicted),
            "KGE": kge(result.observed, result.predicted),
            "PBIAS": pbias(result.observed, result.predicted),
        },
        "thresholds": {"q5": rfa.q5, "q95": rfa.q95, "q99": rfa.q99},
        "early_warning": {
            col: {
                "AUC": auc_roc(labels[col].to_numpy(), probs[col].to_numpy()),
                "F1@0.5": f1_at_threshold(labels[col].to_numpy(), probs[col].to_numpy()),
                "Brier": brier_score(labels[col].to_numpy(), probs[col].to_numpy()),
            }
            for col in labels.columns
        },
        "n_predictions": int(len(result.predicted)),
        "n_refits": int(len(result.refit_dates)),
        "sample_root": str(SAMPLE_ROOT),
        "target_basin": TARGET,
    }

    metrics_path = OUT / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float))
    pd.DataFrame(
        {"observed": result.observed, "predicted": result.predicted},
        index=result.dates,
    ).to_csv(OUT / "walk_forward.csv")
    log.info("Sample pipeline complete → %s", metrics_path)
    print(json.dumps(metrics["continuous"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
