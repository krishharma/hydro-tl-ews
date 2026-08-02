"""End-to-end CLI runner.

Usage::

    python scripts/run_experiment.py --config configs/pretrain.yaml
    python scripts/run_experiment.py --config configs/finetune_conservative.yaml
    python scripts/run_experiment.py --config configs/walk_forward.yaml
    python scripts/run_experiment.py --config configs/smoke_test.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydro_tl_ews.utils.config import ExperimentConfig
from hydro_tl_ews.utils.logging import get_logger
from hydro_tl_ews.utils.seed import set_global_seed

log = get_logger("hydro_tl_ews.cli")


def _load_basins(cfg: ExperimentConfig):
    if cfg.data.get("source") == "synthetic":
        from hydro_tl_ews.data.synthetic_camels import SyntheticCamels
        ds = SyntheticCamels(
            n_basins=cfg.data.get("n_basins", 12),
            n_days=cfg.data.get("n_days", 3650),
            snow_fraction=cfg.data.get("snow_fraction", 0.4),
            seed=cfg.seed,
        )
        # Pick the most snow-dominated synthetic basin as the target
        attrs = ds.load_attributes()
        target_id = attrs["frac_snow"].idxmax()
        return ds, target_id, attrs
    else:
        from hydro_tl_ews.data.camels import CamelsDataset
        ds = CamelsDataset(cfg.data["camels_root"])
        attrs = ds.load_attributes()
        target_id = cfg.data["target_basin"]
        return ds, target_id, attrs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true",
                        help="Run a tiny end-to-end smoke test using synthetic data.")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config)
    set_global_seed(cfg.seed)
    log.info("Loaded config: %s | stage=%s", cfg.name, cfg.stage)

    # Synthetic configs always take the smoke path (``--smoke`` still works).
    if args.smoke or cfg.data.get("source") == "synthetic":
        try:
            from .smoke_pipeline import run_smoke  # type: ignore
        except ImportError:
            from smoke_pipeline import run_smoke
        run_smoke(cfg)
        return 0

    if cfg.stage == "pretrain":
        try:
            from .stages.pretrain_stage import run_pretrain  # type: ignore
        except ImportError:
            from stages.pretrain_stage import run_pretrain
        run_pretrain(cfg)
    elif cfg.stage == "finetune_conservative":
        try:
            from .stages.finetune_stage import run_finetune  # type: ignore
        except ImportError:
            from stages.finetune_stage import run_finetune
        run_finetune(cfg, mode="conservative")
    elif cfg.stage == "finetune_progressive":
        try:
            from .stages.finetune_stage import run_finetune  # type: ignore
        except ImportError:
            from stages.finetune_stage import run_finetune
        run_finetune(cfg, mode="progressive")
    elif cfg.stage == "local_baseline":
        try:
            from .stages.finetune_stage import run_local_baseline  # type: ignore
        except ImportError:
            from stages.finetune_stage import run_local_baseline
        run_local_baseline(cfg)
    elif cfg.stage == "zero_shot":
        try:
            from .stages.zero_shot_stage import run_zero_shot  # type: ignore
        except ImportError:
            from stages.zero_shot_stage import run_zero_shot
        run_zero_shot(cfg)
    elif cfg.stage == "walk_forward":
        try:
            from .stages.walk_forward_stage import run_walk_forward  # type: ignore
        except ImportError:
            from stages.walk_forward_stage import run_walk_forward
        run_walk_forward(cfg)
    elif cfg.stage == "min_data_sensitivity":
        try:
            from .stages.min_data_sensitivity_stage import run_min_data_sensitivity  # type: ignore
        except ImportError:
            from stages.min_data_sensitivity_stage import run_min_data_sensitivity
        run_min_data_sensitivity(cfg)
    else:
        log.error("Unknown stage: %s", cfg.stage)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
