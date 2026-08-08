#!/usr/bin/env python3
"""Run the supervisor-optimized Lick Creek pilot with multi-seed averaging.

Implements feedback (a–d) in combination:
  (a) similar_donor_count=7 (pruned donor pool)
  (b) initial_forget_bias=3.0 (already the project default)
  (c) dynamic_feature_set=mean_temp (Tmin/Tmax → daily mean T)
  (d) three random seeds; average daily streamflow predictions for scoring

Usage (from repo root)::

    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    python scripts/run_midwest_opt.py
    python scripts/run_midwest_opt.py --seeds 42 123 456
    python scripts/run_midwest_opt.py --skip-walk-forward
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "configs" / "midwest_opt"
OUT = ROOT / "results" / "midwest_opt"
DEFAULT_SEEDS = [42, 123, 456]

STAGES = [
    "pretrain.yaml",
    "zero_shot.yaml",
    "finetune_conservative.yaml",
    "local_baseline.yaml",
    "eval_finetune.yaml",
    "eval_local.yaml",
    "walk_forward.yaml",
]

PRED_KEYS = {
    "zero_shot": "zero_shot_predictions.csv",
    "finetune_eval": "finetune_eval_predictions.csv",
    "local_eval": "local_eval_predictions.csv",
    "walk_forward": "walk_forward.csv",
}


def _rewrite_paths(obj, seed: int, old_token: str = "seed_42"):
    """Recursively replace seed_42 path tokens and top-level seed."""
    new_token = f"seed_{seed}"
    if isinstance(obj, dict):
        return {k: _rewrite_paths(v, seed, old_token) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite_paths(v, seed, old_token) for v in obj]
    if isinstance(obj, str) and old_token in obj:
        return obj.replace(old_token, new_token)
    return obj


def materialize_seed_configs(seed: int) -> Path:
    """Write seed-specific YAMLs under results/midwest_opt/configs/seed_N/."""
    out_dir = OUT / "configs" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in STAGES:
        raw = yaml.safe_load((TEMPLATE_DIR / name).read_text())
        raw["seed"] = seed
        raw["name"] = f"{raw.get('name', name)}_{seed}"
        raw = _rewrite_paths(raw, seed)
        (out_dir / name).write_text(yaml.safe_dump(raw, sort_keys=False))
    return out_dir


def run_stage(cfg_path: Path) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
           "--config", str(cfg_path)]
    print(f"\n===== {cfg_path.parent.name}/{cfg_path.name} =====", flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env={
        **dict(**{k: v for k, v in __import__("os").environ.items()}),
        "PYTHONPATH": f"{ROOT / 'src'}" + (
            f":{__import__('os').environ['PYTHONPATH']}"
            if __import__("os").environ.get("PYTHONPATH") else ""
        ),
    })


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def metrics_from_predictions(pred_path: Path) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from hydro_tl_ews.evaluation.metrics import kge, nse, pbias

    df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
    obs = df["observed"].to_numpy(dtype=float)
    pred = df["predicted"].to_numpy(dtype=float)
    return {
        "NSE": float(nse(obs, pred)),
        "KGE": float(kge(obs, pred)),
        "PBIAS": float(pbias(obs, pred)),
        "n_samples": int(len(pred)),
    }


def average_predictions(seeds: list[int], key: str, filename: str) -> Path:
    """Average daily predicted streamflow across seeds; keep observed from seed 0."""
    frames = []
    for seed in seeds:
        path = OUT / f"seed_{seed}" / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions for seed {seed}: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        frames.append(df)

    base = frames[0][["observed"]].copy()
    pred_mat = np.column_stack([f["predicted"].to_numpy(dtype=float) for f in frames])
    base["predicted"] = pred_mat.mean(axis=1)
    if "bias_correction" in frames[0].columns:
        bias_mat = np.column_stack([
            f["bias_correction"].to_numpy(dtype=float) for f in frames
        ])
        base["bias_correction"] = bias_mat.mean(axis=1)
    for i, seed in enumerate(seeds):
        base[f"predicted_seed_{seed}"] = pred_mat[:, i]

    out_dir = OUT / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    base.to_csv(out_path)
    print(f"Averaged {key} predictions -> {out_path}", flush=True)
    return out_path


def write_summary(seeds: list[int], stages_run: list[str]) -> Path:
    per_seed = {}
    for seed in seeds:
        seed_dir = OUT / f"seed_{seed}"
        per_seed[str(seed)] = {
            "zero_shot": load_json(seed_dir / "zero_shot_metrics.json"),
            "finetune_eval": load_json(seed_dir / "finetune_eval_metrics.json"),
            "local_eval": load_json(seed_dir / "local_eval_metrics.json"),
            "walk_forward": load_json(seed_dir / "walk_forward_metrics.json"),
            "donors": load_json(seed_dir / "checkpoints" / "donor_basins.json"),
        }

    ensemble = {}
    for key, filename in PRED_KEYS.items():
        if key == "walk_forward" and "walk_forward.yaml" not in stages_run:
            continue
        if not all((OUT / f"seed_{s}" / filename).exists() for s in seeds):
            continue
        avg_path = average_predictions(seeds, key, filename)
        ensemble[key] = {
            **metrics_from_predictions(avg_path),
            "predictions_path": str(avg_path.relative_to(ROOT)),
            "method": "mean_of_daily_predictions_across_seeds",
            "seeds": seeds,
        }

    # Baseline comparison (midwest_mini, single seed 42)
    baseline_dir = ROOT / "results" / "midwest_mini"
    baseline = {
        "zero_shot": load_json(baseline_dir / "zero_shot_metrics.json"),
        "finetune_eval": load_json(baseline_dir / "finetune_eval_metrics.json"),
        "local_eval": load_json(baseline_dir / "local_eval_metrics.json"),
        "walk_forward": load_json(baseline_dir / "walk_forward_metrics.json"),
        "note": "midwest_mini: ~17 donors, full Daymet (Tmin+Tmax), seed 42, forget bias +3.0",
    }

    donors = None
    for seed in seeds:
        d = load_json(OUT / f"seed_{seed}" / "checkpoints" / "donor_basins.json")
        if d:
            donors = d
            break

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_basin": "05507600",
        "target_name": "Lick Creek near Perry, MO",
        "scope": "Supervisor-optimized Lick Creek pilot (a–d combined)",
        "optimizations": {
            "a_pruned_donors": "similar_donor_count=7 (Pool et al. 2021 style)",
            "b_forget_gate_bias": 3.0,
            "c_mean_temperature": "dynamic_feature_set=mean_temp (5 dynamic inputs)",
            "d_multi_seed": seeds,
        },
        "donors": donors,
        "seeds": seeds,
        "per_seed": per_seed,
        "ensemble_mean_predictions": ensemble,
        "baseline_midwest_mini": baseline,
    }
    path = OUT / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {path}", flush=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--from-stage",
        default=STAGES[0],
        help="Resume from this config filename (inclusive).",
    )
    parser.add_argument(
        "--skip-walk-forward",
        action="store_true",
        help="Skip walk-forward stage (faster continuous-metric comparison).",
    )
    parser.add_argument(
        "--ensemble-only",
        action="store_true",
        help="Skip training; just average existing seed predictions.",
    )
    args = parser.parse_args(argv)
    seeds = list(args.seeds)

    OUT.mkdir(parents=True, exist_ok=True)
    assert (ROOT / "data" / "basin_dataset_public_v1p2").is_dir()
    assert (ROOT / "data" / "camels_attributes_v2.0" / "camels_topo.txt").is_file()

    stages = list(STAGES)
    if args.skip_walk_forward:
        stages = [s for s in stages if s != "walk_forward.yaml"]
    if args.from_stage not in stages and not args.ensemble_only:
        raise SystemExit(f"Unknown stage {args.from_stage}; choose from {stages}")
    if not args.ensemble_only:
        stages = stages[stages.index(args.from_stage):]

    if not args.ensemble_only:
        for seed in seeds:
            cfg_dir = materialize_seed_configs(seed)
            (OUT / f"seed_{seed}" / "checkpoints").mkdir(parents=True, exist_ok=True)
            (OUT / f"seed_{seed}" / "history").mkdir(parents=True, exist_ok=True)
            print(f"\n########## SEED {seed} ##########", flush=True)
            for stage in stages:
                run_stage(cfg_dir / stage)

    write_summary(seeds, stages if not args.ensemble_only else STAGES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
