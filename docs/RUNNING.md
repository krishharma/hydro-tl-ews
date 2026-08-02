# Running hydro-tl-ews

## Prerequisites

1. Python 3.10+
2. Install deps (`pip install -r requirements.txt` or conda env files)
3. For real experiments: CAMELS-US extracted under `data/` (see [`../data/README.md`](../data/README.md))

Put the package on `PYTHONPATH` or install editable:

```bash
export PYTHONPATH="$PWD/src"
# or
pip install -e .
```

## Recommended order (single target)

1. **Smoke** — confirm the install (`configs/smoke_test.yaml`).
2. **`pretrain_subset200`** — practical regional model (~200 donors).
3. **`finetune_conservative`** (and optionally progressive / local / zero-shot).
4. **`walk_forward`** — main operational evaluation + EWS artifacts.
5. Optional: `min_data_sensitivity`, figures, paper builders.

`bash scripts/run_full_pipeline.sh` runs steps 2–5 for Merced (`11264500`) using
the subset-200 checkpoint chain.

## Full CAMELS + multi-target

```bash
python scripts/run_experiment.py --config configs/pretrain.yaml
python scripts/run_multi_target.py
```

Multi-target configs already point at `results/checkpoints/pretrain.pt`.

## Compute notes

| Workload | Rough cost |
|----------|------------|
| Smoke / unit tests | Minutes, CPU fine |
| `pretrain_subset200` | Hours–1+ day on Apple Silicon / mid GPU |
| `pretrain` (full CAMELS) | Multi-day on laptop; prefer CUDA GPU |
| Walk-forward (1 basin) | Hours |
| 7-basin multi-target | Days after pretrain |

Apple Silicon uses **MPS** automatically when available. Keep the laptop
plugged in for long runs; batch size 256 may be tight on 8 GB RAM — lower
`training.batch_size` in the YAML if you OOM.

## Config keys that matter

- `data.camels_root` — must point at the CAMELS root (`data` by default)
- `model.pretrained_checkpoint` — must exist before transfer / WF stages
- `walk_forward.refit_train_start` — **set this** (shipped configs use `2009-01-01`)
  so refits do not train on decades of pre-warmup local Q
- `data.sequence_length` — default 365; dominates runtime

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `FileNotFoundError` on checkpoint | Wrong stage order or path mismatch (`pretrain.pt` vs `pretrain_subset200.pt`) |
| CAMELS assert / missing basins | Incomplete extract under `data/` |
| OOM during pretrain | Lower `batch_size` or use `pretrain_subset200` |
| `to_parquet` fails | Install `pyarrow` (listed in requirements) |
| Paper/figure scripts crash | Missing `results/*` or optional deps (`reportlab`, `python-docx`) |
