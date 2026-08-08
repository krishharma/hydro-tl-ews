"""Phase 2 — Fine-tuning on the data-scarce target basin."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from hydro_tl_ews.data.camels import (
    STATIC_ATTRIBUTES,
)
from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset
from hydro_tl_ews.data.features import dynamic_features_from_cfg, open_camels
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.models.ealstm import EALSTM, EALSTMConfig
from hydro_tl_ews.training.trainer import Trainer
from hydro_tl_ews.training.transfer import (
    FineTuneConfig,
    fine_tune_conservative,
    fine_tune_progressive,
    train_local_baseline,
)
from hydro_tl_ews.utils.config import ExperimentConfig
from hydro_tl_ews.utils.logging import get_logger

log = get_logger(__name__)


def _setup(cfg: ExperimentConfig):
    ds = open_camels(cfg)
    target_id = cfg.data["target_basin"]
    target = ds.load_basin(target_id)
    attrs_all = ds.load_attributes()

    warmup = tuple(cfg.data["warmup_period"])
    val = tuple(cfg.data["validation_period"])
    seq_len = cfg.data.get("sequence_length", 365)

    dyn_norm = Normalizer.fit(target.forcings.loc[warmup[0]:warmup[1]])
    static_norm = StaticNormalizer.fit(attrs_all.loc[:, STATIC_ATTRIBUTES])

    warmup_ds = MultiBasinSequenceDataset(
        {target_id: target}, warmup, dyn_norm, static_norm,
        sequence_length=seq_len,
    )
    val_ds = MultiBasinSequenceDataset(
        {target_id: target}, val, dyn_norm, static_norm,
        sequence_length=seq_len,
    )
    return target, dyn_norm, static_norm, warmup_ds, val_ds


def run_finetune(cfg: ExperimentConfig, mode: str) -> None:
    target, dyn_norm, static_norm, warmup_ds, val_ds = _setup(cfg)
    train_loader = DataLoader(warmup_ds, batch_size=cfg.training.get("batch_size", 64),
                              shuffle=True)
    val_loader = (DataLoader(val_ds, batch_size=cfg.training.get("batch_size", 64),
                             shuffle=False) if len(val_ds) else None)

    ckpt = cfg.model.get("pretrained_checkpoint")
    if not ckpt:
        raise ValueError("finetune stage requires model.pretrained_checkpoint")
    model = Trainer.load_model(ckpt)

    ft_cfg = FineTuneConfig(
        head_lr=cfg.training.get("head_lr", 1e-3),
        lstm_lr=cfg.training.get("lstm_lr", 1e-5),
        epochs_head_only=cfg.training.get("epochs", cfg.training.get("epochs_head_only", 5)),
        epochs_progressive=cfg.training.get("epochs_progressive", 5),
        patience=cfg.training.get("patience", 3),
        unfreeze_fraction=cfg.training.get("unfreeze_fraction", 0.25),
        weight_decay=cfg.training.get("weight_decay", 0.0),
    )
    if mode == "conservative":
        state = fine_tune_conservative(model, train_loader, val_loader, ft_cfg)
    elif mode == "progressive":
        state = fine_tune_progressive(model, train_loader, val_loader, ft_cfg)
    else:
        raise ValueError(f"Unknown fine-tune mode: {mode}")

    out_path = Path(cfg.output.get("checkpoint_path",
                                   f"results/checkpoints/finetune_{mode}.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "config": model.cfg.__dict__}, out_path)

    history_path = Path(cfg.output.get(
        "history_path", f"results/history/finetune_{mode}.json"))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(state.history, indent=2))
    log.info("Fine-tune (%s) complete | best=%.4f", mode, state.best_val_loss)


def run_local_baseline(cfg: ExperimentConfig) -> None:
    target, dyn_norm, static_norm, warmup_ds, val_ds = _setup(cfg)
    train_loader = DataLoader(warmup_ds, batch_size=cfg.training.get("batch_size", 64),
                              shuffle=True)
    val_loader = (DataLoader(val_ds, batch_size=cfg.training.get("batch_size", 64),
                             shuffle=False) if len(val_ds) else None)

    dyn_feats = dynamic_features_from_cfg(cfg)
    model_cfg = EALSTMConfig(
        dynamic_input_size=len(dyn_feats),
        static_input_size=len(STATIC_ATTRIBUTES),
        hidden_size=cfg.model.get("hidden_size", 256),
        dropout=cfg.model.get("dropout", 0.4),
        initial_forget_bias=cfg.model.get("initial_forget_bias", 3.0),
    )
    model = EALSTM(model_cfg)
    state = train_local_baseline(
        model, train_loader, val_loader,
        epochs=cfg.training.get("epochs", 50),
        patience=cfg.training.get("patience", 10),
        head_lr=cfg.training.get("head_lr", 1e-3),
    )
    out_path = Path(cfg.output.get("checkpoint_path",
                                   "results/checkpoints/local_baseline.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "config": model.cfg.__dict__}, out_path)
    history_path = Path(cfg.output.get(
        "history_path", "results/history/local_baseline.json"))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(state.history, indent=2))
    log.info("Local baseline complete | best=%.4f", state.best_val_loss)
