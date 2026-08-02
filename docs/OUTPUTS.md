# Outputs reference

All generated artifacts are written under `results/` (gitignored). Paths below
are the single-target defaults; multi-target mirrors them under
`results/multi_target/<gauge_id>/`.

## Checkpoints (`.pt`)

`torch.save` dict with:

- `model_state` — `state_dict` of the EA-LSTM
- `config` — `EALSTMConfig` fields (`hidden_size`, input sizes, …)

Load via `Trainer.load_model(path)`.

| File | Produced by |
|------|-------------|
| `checkpoints/pretrain_subset200.pt` | `configs/pretrain_subset200.yaml` |
| `checkpoints/pretrain.pt` | `configs/pretrain.yaml` |
| `checkpoints/finetune_conservative.pt` | Approach A fine-tune |
| `checkpoints/finetune_progressive.pt` | Approach B fine-tune |
| `checkpoints/local_baseline.pt` | From-scratch warmup baseline |

## Metrics JSON

### Zero-shot / continuous-only stages

```json
{
  "NSE": 0.0,
  "KGE": 0.0,
  "PBIAS": 0.0,
  "n_samples": 1461,
  "target_basin": "11264500",
  "evaluation_period": ["2011-01-01", "2014-12-31"]
}
```

### Walk-forward

```json
{
  "continuous": {"NSE": ..., "KGE": ..., "PBIAS": ...},
  "thresholds": {"q5": ..., "q95": ..., "q99": ...},
  "early_warning": {
    "flood_q95_lead3d": {"AUC": ..., "F1@0.5": ..., "Brier": ...},
    "...": {}
  },
  "n_predictions": ...,
  "n_refits": ...
}
```

## Time series

| File | Columns / index |
|------|-----------------|
| `walk_forward.parquet` (or `.csv`) | index=`date`; `observed`, `predicted`, `bias_correction` |
| `*_predictions.csv` | index=`date`; `observed`, `predicted` |
| `walk_forward_warnings.csv` | Event labels + `*_prob` columns for flood/drought × lead times |

## History JSON

List/dict of per-epoch train and validation losses from `Trainer.fit`.

## Smoke (`results/smoke/`)

Synthetic run artifacts (`summary.json`, CSVs, small checkpoint). Used to verify
the install; **not** comparable to CAMELS metrics.

## Analysis add-ons

Scripts under `scripts/analysis/` and `scripts/paper_supplement_analysis.py`
may write extra files such as:

- `results/ews_climatology_benchmark.json`
- `results/ews_recalibrated.json`
- `results/multi_target/summary.csv`

These are optional post-processing, not produced by `run_full_pipeline.sh`.
