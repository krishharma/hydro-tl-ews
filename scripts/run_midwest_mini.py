#!/usr/bin/env python3
"""Run the laptop-optimized Midwest mini study end-to-end.

Target: USGS 05507600 (Lick Creek, MO). Donors: up to 40 similar basins inside
a Midwest geographic bbox, using the partial CAMELS extract under ``data/``.

Usage (from repo root)::

    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    python scripts/run_midwest_mini.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "midwest_mini"
OUT = ROOT / "results" / "midwest_mini"

STAGES = [
    "pretrain.yaml",
    "zero_shot.yaml",
    "finetune_conservative.yaml",
    "local_baseline.yaml",
    "eval_finetune.yaml",
    "eval_local.yaml",
    "walk_forward.yaml",
]


def run_stage(cfg_name: str) -> None:
    cfg = CONFIGS / cfg_name
    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
           "--config", str(cfg)]
    print(f"\n===== {cfg_name} =====", flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_summary() -> Path:
    shap_path = OUT / "shap_global_importance.csv"
    shap_top = []
    if shap_path.exists():
        import csv
        with shap_path.open() as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for i, row in enumerate(reader):
                if i >= 15:
                    break
                if len(row) >= 2:
                    shap_top.append({"feature": row[0], "mean_abs_shap": float(row[1])})
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_basin": "05507600",
        "target_name": "Lick Creek near Perry, MO",
        "scope": "Midwest CAMELS subset (laptop-optimized)",
        "zero_shot": load_json(OUT / "zero_shot_metrics.json"),
        "finetune_eval": load_json(OUT / "finetune_eval_metrics.json"),
        "local_eval": load_json(OUT / "local_eval_metrics.json"),
        "walk_forward": load_json(OUT / "walk_forward_metrics.json"),
        "pretrain_history": load_json(OUT / "history" / "pretrain.json"),
        "shap_top_features": shap_top,
    }
    path = OUT / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote {path}", flush=True)
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-stage",
        default=STAGES[0],
        help="Resume from this config filename (inclusive).",
    )
    parser.add_argument("--skip-pretrain", action="store_true",
                        help="Alias for --from-stage zero_shot.yaml")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "history").mkdir(parents=True, exist_ok=True)

    assert (ROOT / "data" / "basin_dataset_public_v1p2").is_dir()
    assert (ROOT / "data" / "camels_attributes_v2.0" / "camels_topo.txt").is_file()

    start = "zero_shot.yaml" if args.skip_pretrain else args.from_stage
    if start not in STAGES:
        raise SystemExit(f"Unknown stage {start}; choose from {STAGES}")
    stages = STAGES[STAGES.index(start):]
    for stage in stages:
        run_stage(stage)
    write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
