# Paper figures (data only)

PNG assets referenced by `docs/short_paper.tex`. The paper includes only
**data-related** figures and tables---no experimental result plots, since full
runs have not been completed.

| File | Description |
|------|-------------|
| `fig_camels_map.png` | CAMELS-US gauge locations (Figure camels) |

## Refresh

```bash
bash scripts/paper/sync_paper_images.sh
cd docs && zip -r paper_images.zip paper_images/
```

Keep `paper_images/` next to `short_paper.tex` when uploading the project.
