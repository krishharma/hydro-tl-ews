"""Phase 1 — Regional pre-training on CAMELS-US."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from hydro_tl_ews.data.camels import (
    CamelsDataset,
    DYNAMIC_FEATURES,
    STATIC_ATTRIBUTES,
)
from hydro_tl_ews.data.clustering import select_donor_basins
from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.models.ealstm import EALSTM, EALSTMConfig
from hydro_tl_ews.training.trainer import Trainer
from hydro_tl_ews.utils.config import ExperimentConfig
from hydro_tl_ews.utils.logging import get_logger

log = get_logger(__name__)


def _haversine_km(lat1, lon1, lat2, lon2):
    import math
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def exclude_targets_and_buffer(attrs, donors: list[str], targets: list[str],
                               exclusion_km: float | None) -> list[str]:
    """Drop every target basin from ``donors`` and, when coordinates are
    available, any donor within ``exclusion_km`` of any target.

    Multi-target support exists so a single regional pretrain can serve the
    whole multi-basin evaluation without target leakage.
    """
    targets = [t for t in targets if t]
    donors = [b for b in donors if b not in set(targets)]
    if not exclusion_km:
        return donors
    lat_cols = [c for c in attrs.columns if c.lower() in {"gauge_lat", "lat", "latitude"}]
    lon_cols = [c for c in attrs.columns if c.lower() in {"gauge_lon", "lon", "longitude"}]
    if not (lat_cols and lon_cols):
        log.warning("exclusion_buffer_km requested but no lat/lon columns found in attributes.")
        return donors
    lat_col, lon_col = lat_cols[0], lon_cols[0]
    t_coords = [(float(attrs.loc[t, lat_col]), float(attrs.loc[t, lon_col]))
                for t in targets if t in attrs.index]
    keep = []
    for bid in donors:
        blat = float(attrs.loc[bid, lat_col])
        blon = float(attrs.loc[bid, lon_col])
        if all(_haversine_km(tlat, tlon, blat, blon) > float(exclusion_km)
               for tlat, tlon in t_coords):
            keep.append(bid)
    return keep


def run_pretrain(cfg: ExperimentConfig) -> None:
    ds = CamelsDataset(cfg.data["camels_root"])
    attrs = ds.load_attributes()
    target_basin = cfg.data.get("target_basin")
    # target_basins (list) extends the single-target exclusion so one pretrain
    # can serve every basin in a multi-target evaluation without leakage.
    targets = list(cfg.data.get("target_basins") or [])
    if target_basin and target_basin not in targets:
        targets.insert(0, target_basin)

    n_similar = cfg.data.get("similar_donor_count")
    if n_similar:
        # Similarity-selected donor mode stays anchored on the primary target.
        donors = select_donor_basins(
            attrs.loc[:, STATIC_ATTRIBUTES],
            target_basin=target_basin,
            n_donors=int(n_similar),
        )
    else:
        donors = list(attrs.index)
    donors = exclude_targets_and_buffer(
        attrs, donors, targets, cfg.data.get("exclusion_buffer_km"))

    log.info("Pre-training on %d basins (targets excluded: %s)",
             len(donors), ", ".join(targets))
    basins = ds.load_basins(donors)

    pretrain_period = tuple(cfg.data["pretrain_period"])
    val_period = tuple(cfg.data["validation_period"])
    seq_len = cfg.data.get("sequence_length", 365)

    # Fit normalizers on the pre-training period only
    forc_train = pd.concat([b.forcings.loc[pretrain_period[0]:pretrain_period[1]]
                            for b in basins.values()])
    dyn_norm = Normalizer.fit(forc_train)
    static_norm = StaticNormalizer.fit(attrs.loc[donors, STATIC_ATTRIBUTES])

    train_ds = MultiBasinSequenceDataset(basins, pretrain_period, dyn_norm,
                                         static_norm, sequence_length=seq_len)
    val_ds = MultiBasinSequenceDataset(basins, val_period, dyn_norm, static_norm,
                                       sequence_length=seq_len)

    bs = cfg.training.get("batch_size", 256)
    n_workers = int(cfg.data.get("num_workers", 0))
    loader_kwargs: dict = {"num_workers": n_workers}
    if torch.cuda.is_available():
        loader_kwargs["pin_memory"] = True
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)

    model_cfg = EALSTMConfig(
        dynamic_input_size=len(DYNAMIC_FEATURES),
        static_input_size=len(STATIC_ATTRIBUTES),
        hidden_size=cfg.model.get("hidden_size", 256),
        dropout=cfg.model.get("dropout", 0.4),
        initial_forget_bias=cfg.model.get("initial_forget_bias", 3.0),
    )
    model = EALSTM(model_cfg)
    trainer = Trainer(
        model=model, mode="pretrain",
        head_lr=cfg.training.get("learning_rate", 1e-3),
        weight_decay=cfg.training.get("weight_decay", 0.0),
        clip_grad_norm=cfg.training.get("clip_grad_norm", 1.0),
    )
    state = trainer.fit(
        train_loader, val_loader,
        epochs=cfg.training.get("epochs", 50),
        patience=cfg.training.get("patience", 10),
        checkpoint_path=cfg.output.get("checkpoint_path",
                                       "results/checkpoints/pretrain.pt"),
    )

    history_path = Path(cfg.output.get("history_path",
                                       "results/history/pretrain.json"))
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(state.history, indent=2))
    log.info("Pre-training complete | best val=%.4f at epoch %d",
             state.best_val_loss, state.best_epoch)
