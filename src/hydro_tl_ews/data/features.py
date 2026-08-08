"""Shared helpers for resolving dynamic forcing feature sets from configs."""
from __future__ import annotations

from hydro_tl_ews.data.camels import CamelsDataset, resolve_dynamic_features
from hydro_tl_ews.utils.config import ExperimentConfig


def feature_set_from_cfg(cfg: ExperimentConfig) -> str:
    return str(cfg.data.get("dynamic_feature_set", "full") or "full")


def dynamic_features_from_cfg(cfg: ExperimentConfig) -> list[str]:
    return resolve_dynamic_features(feature_set_from_cfg(cfg))


def open_camels(cfg: ExperimentConfig) -> CamelsDataset:
    return CamelsDataset(
        cfg.data["camels_root"],
        dynamic_feature_set=feature_set_from_cfg(cfg),
    )
