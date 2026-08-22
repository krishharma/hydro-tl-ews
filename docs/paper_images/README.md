# Paper figures

PNG assets referenced by `docs/short_paper.tex` and embedded in
`docs/Short_Paper_Hydro_TL_EWS.docx`.

| File | Panel |
|------|--------|
| `fig_donor_target_map.png` | 17 Midwest donors + Lick Creek target |
| `fig_hydrograph_2011_2014.png` | Observed, walk-forward, local-from-scratch |
| `fig_skill_bars.png` | NSE/KGE vs persistence and local baselines |
| `fig_donor_pruning.png` | 17- vs 8-donor NSE |
| `fig_reliability_q95.png` | Flood Q95 reliability (product vs window-max) |

`fig_camels_map.png` is the old continental map and is no longer used in the paper.

Refresh::

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/analysis/poster_from_csvs.py
python scripts/paper/build_short_paper_docx.py
```
