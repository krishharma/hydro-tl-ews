# hydro-tl-ews

**Transfer Learning for Hydrological Early Warning in Data-Scarce Regions.**

An Entity-Aware LSTM (EA-LSTM) is pre-trained on CAMELS-US donor catchments,
transferred to a data-scarce target basin (~2-year warmup), and evaluated under a
rolling-origin **walk-forward** protocol. Outputs include continuous streamflow
skill (NSE / KGE / PBIAS), flood/drought early-warning probabilities, optional
SHAP attributions, and a 7-basin multi-regime study.

Primary single-target default: **Merced River at Happy Isles** (`11264500`).

---

## Expected inputs

| Input | Required for | Notes |
|-------|----------------|-------|
| **Python ≥ 3.10** + deps | Always | See Setup |
| **CAMELS-US** under `data/` | All real stages | ~14 GB; see [`data/README.md`](data/README.md) |
| **YAML config** under `configs/` | Every stage | Points at data paths, periods, checkpoints |
| **Upstream checkpoint** | Fine-tune / zero-shot / walk-forward / multi-target | e.g. `results/checkpoints/pretrain_subset200.pt` |

No CAMELS is needed for the **smoke** path (synthetic basins).

### CAMELS layout (required)

```
data/
  basin_dataset_public_v1p2/
    basin_mean_forcing/daymet/...
    usgs_streamflow/...
  camels_attributes_v2.0/
    camels_topo.txt
    camels_clim.txt
    ...
```

Upstream citation notes for the attributes package live in `data/readme.txt`.

---

## Setup

```bash
cd hydro-tl-ews
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # optional; enables `hydro-tl-ews` CLI
```

Conda alternatives: `environment.yml` (GPU-oriented) or `environment-cpu.yml`.

Verify CAMELS (after download):

```bash
python -c "
from pathlib import Path
root = Path('data')
assert (root / 'basin_dataset_public_v1p2').is_dir()
assert (root / 'camels_attributes_v2.0' / 'camels_topo.txt').is_file()
print('CAMELS layout OK')
"
```

---

## How to run

All commands assume the repo root and an activated environment.
If you did not `pip install -e .`, prefix with:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

### 1. Smoke test (no CAMELS, minutes)

```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml
# or explicitly:
python scripts/run_experiment.py --config configs/smoke_test.yaml --smoke
python -m pytest -q
```

Writes under `results/smoke/` (synthetic end-to-end check).

### 1b. Mini sample CAMELS (real loader, a few minutes)

No full CAMELS download required. Generates a tiny CAMELS-compatible archive
and runs pretrain → fine-tune → walk-forward through `CamelsDataset`:

```bash
python scripts/make_sample_camels.py      # → data/sample_camels/ (~4 MB)
python scripts/run_sample_pipeline.py     # → results/sample/
```

### 1c. Midwest study (real CAMELS, laptop)

Completed Lick Creek run (USGS `05507600`, 17 similar Midwest donors, full
2011–2014 evaluation). Needs a partial CAMELS extract (HUC `04/05/07/10`
Daymet + streamflow) under `data/` — see [`data/README.md`](data/README.md).

```bash
python scripts/run_midwest_mini.py
# naive baselines, warning BSS, and poster figures (no retraining):
python scripts/analysis/poster_from_csvs.py
python scripts/paper/build_short_paper_docx.py   # → docs/Short_Paper_Hydro_TL_EWS.docx
```

Configs: [`configs/midwest_mini/`](configs/midwest_mini/). Outputs:
`results/midwest_mini/`. Paper:
[`docs/Short_Paper_Hydro_TL_EWS.docx`](docs/Short_Paper_Hydro_TL_EWS.docx).

### 2. Single stage (config-driven)

```bash
python scripts/run_experiment.py --config configs/<stage>.yaml
# after editable install:
hydro-tl-ews --config configs/<stage>.yaml
```

| Stage config | Role | Needs |
|--------------|------|--------|
| `pretrain_subset200.yaml` | ~200 similar donors, 30 epochs (default laptop/GPU path) | CAMELS |
| `pretrain.yaml` | Full CAMELS regional model (heavy; multi-day on laptop) | CAMELS |
| `zero_shot.yaml` | Pretrained model, no fine-tune | `pretrain_subset200.pt` |
| `finetune_conservative.yaml` | Approach A (head-only) | `pretrain_subset200.pt` |
| `finetune_progressive.yaml` | Approach B (progressive unfreeze) | `pretrain_subset200.pt` |
| `local_baseline.yaml` | Train from scratch on warmup | CAMELS |
| `walk_forward.yaml` | Operational 2011–2014 eval + EWS (+ SHAP) | `finetune_conservative.pt` |
| `walk_forward_progressive.yaml` | Same with Approach B refits | `finetune_progressive.pt` |
| `min_data_sensitivity.yaml` | Warmup length sweep | `pretrain_subset200.pt` |

### 3. Full single-target pipeline

```bash
bash scripts/run_full_pipeline.sh
```

Order: `pretrain_subset200` → `zero_shot` → finetunes → `local_baseline` →
`walk_forward` → `min_data_sensitivity`. Log: `results/logs/full_pipeline.log`.

For **full CAMELS** pretrain (feeds multi-target):

```bash
python scripts/run_experiment.py --config configs/pretrain.yaml
# → results/checkpoints/pretrain.pt
```

### 4. Seven-basin multi-target study

Requires `results/checkpoints/pretrain.pt` (from `configs/pretrain.yaml`).

```bash
python scripts/run_multi_target.py            # all basins, resumable
python scripts/run_multi_target.py 11264500  # one basin
python scripts/run_multi_target.py --summary
```

Outputs under `results/multi_target/<gauge_id>/` and `results/multi_target/summary.csv`.

### 5. Figures / manuscript (optional)

Need completed `results/` artifacts and extra deps (`reportlab`, `python-docx`
for paper builders — not in core `requirements.txt`).

```bash
python scripts/figures/make_figures.py
python scripts/paper/build_paper.py
python scripts/paper/build_manuscript_docx.py
```

---

## Expected outputs

Everything lands under **`results/`** (gitignored).

| Artifact | Typical path | Contents |
|----------|--------------|----------|
| Checkpoint | `results/checkpoints/*.pt` | Model weights + config dict |
| Train history | `results/history/*.json` | Epoch train/val losses |
| Zero-shot metrics | `results/zero_shot_metrics.json` | NSE, KGE, PBIAS, `n_samples` |
| Zero-shot series | `results/zero_shot_predictions.csv` | Daily obs / pred |
| Walk-forward series | `results/walk_forward.parquet` | date, observed, predicted, bias |
| Walk-forward metrics | `results/walk_forward_metrics.json` | Continuous + EWS AUC/F1/Brier |
| Warning table | `results/walk_forward_warnings.csv` | Labels + probs by lead time |
| SHAP | `results/shap_global_importance.csv` | Mean \|SHAP\| by feature |
| Min-data table | `results/min_data_sensitivity.csv` | Skill vs warmup months |
| Smoke | `results/smoke/` | Synthetic summary + CSV |
| Multi-target | `results/multi_target/...` | Per-basin copies of the above |

Device selection is automatic: **CUDA → MPS (Apple Silicon) → CPU**
(`hydro_tl_ews.utils.device.training_device`).

---

## Repository layout

```
hydro-tl-ews/
├── src/hydro_tl_ews/     # importable library (data, models, training, eval, xai)
├── scripts/              # CLI + stage runners + analysis / figures / paper
├── configs/              # YAML experiments (+ configs/multi_target/)
├── tests/                # pytest
├── data/                 # CAMELS (not versioned) — see data/README.md
├── figures/              # checked-in paper figures
├── docs/                 # running guide, outputs, known limitations
└── results/              # generated (gitignored)
```

---

## Method summary

- **Model** — EA-LSTM (Kratzert et al., 2019): static attributes drive a
  time-invariant input gate; dynamic forcings drive forget / candidate / output.
- **Transfer** — regional pre-train → target fine-tune. Approach A: head-only.
  Approach B: progressive unfreeze of the last LSTM parameter group at a smaller LR.
- **Evaluation** — 2-year warmup, 90-day walk-forward refits, online bias
  correction, validation tail + best-weight restore, `refit_train_start` to keep
  the scarce-data protocol honest.
- **Early warning** — at-site Q5/Q95/Q99 (fit on pre-eval years); Gaussian residual
  mapping; scored with AUC, F1, Brier (and BSS vs DOY climatology in analysis scripts).

---

## Documentation

- [`docs/README.pdf`](docs/README.pdf) — downloadable PDF of this README
- [`docs/RUNNING.md`](docs/RUNNING.md) — detailed run recipes and compute notes
- [`docs/OUTPUTS.md`](docs/OUTPUTS.md) — file formats and how to read them
- [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) — remaining science/UX caveats
- [`data/README.md`](data/README.md) — CAMELS download and layout

## License

MIT — see [`LICENSE`](LICENSE).
