# CAMELS-US data

This repository does **not** bundle CAMELS. Download the dataset and extract it here:

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

**Source:** [CAMELS-US (NCAR)](https://ral.ucar.edu/solutions/products/camels)

**Size:** ~14 GB unpacked (full continental set used by pretrain).

If you have the NCAR zip locally, unpack under `data/` so the layout above matches.
Do not commit large zips to git (`data/downloads/` is gitignored).

**Verify layout** (from repo root):

```bash
python -c "
from pathlib import Path
root = Path('data')
assert (root / 'basin_dataset_public_v1p2').is_dir()
assert (root / 'camels_attributes_v2.0' / 'camels_topo.txt').is_file()
print('CAMELS layout OK')
"
```

Configs set `data.camels_root: data` in `configs/*.yaml`.

`data/readme.txt` is the upstream CAMELS attributes citation/history file
(kept for provenance). Use **this** README for download and layout instructions.

## Mini sample (for quick local tests)

If you do not have full CAMELS yet, generate a tiny CAMELS-compatible archive:

```bash
python scripts/make_sample_camels.py
# → data/sample_camels/   (~8 basins, ~15 years, synthetic)

python scripts/run_sample_pipeline.py
# → results/sample/   (pretrain → fine-tune → walk-forward in a few minutes)
```

This exercises the real `CamelsDataset` loader. It is **not** real CAMELS
observations — only a layout-compatible synthetic stand-in.
