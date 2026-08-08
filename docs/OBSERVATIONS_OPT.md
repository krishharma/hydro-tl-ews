# Supervisor optimizations on the Lick Creek pilot

**Date:** 2026-08-08  
**Target:** USGS `05507600` (Lick Creek near Perry, MO)  
**Baseline:** `results/midwest_mini/` (17 Midwest donors, Daymet with Tmin+Tmax, forget-gate bias +3.0, seed 42)  
**Runner:** `python scripts/run_midwest_opt.py`

Feedback items tested:

| Item | Change | Status in this study |
|------|--------|----------------------|
| (a) Prune donors | 17 → ~7–8 most-similar Midwest donors | Tested (request 10 nearest → 8 with local files) |
| (b) Forget-gate bias +3.0 | LSTM forget bias init | Already the project default; unchanged |
| (c) Mean temperature | Replace Tmin/Tmax with daily mean T | Tested (`dynamic_feature_set: mean_temp`) |
| (d) Multi-seed | Seeds 42 / 123 / 456; average daily Q predictions | Tested |

Artifacts: `results/midwest_opt/summary.json`, `results/midwest_opt_ablate/prune_fulltemp/summary.json`.

---

## Continuous skill (2011–2014)

| Setting | Donors | Weather | NSE ZS | NSE FT | NSE Local | NSE WF |
|---------|--------|---------|--------|--------|-----------|--------|
| Baseline (`midwest_mini`) | 17 | Tmin+Tmax | 0.150 | **0.160** | 0.009 | **0.179** |
| First combined run\* | 5 | mean T | ~0.10 | ~0.09–0.10 | <0 | ~0.12 |
| Ablation: prune only | 8 | Tmin+Tmax | 0.108 | 0.113 | 0.009 | 0.138 |
| Combined a+c+d (ensemble) | 8 | mean T | 0.123 | 0.133 | −0.097 | 0.165 |

\*Requesting only 7 nearest basins left **5** after the partial CAMELS file filter; later runs use `similar_donor_count: 10` to land ~8 available donors (within the 6–7 guidance after accounting for missing HUC files).

Ensemble = mean of daily predicted streamflow across seeds 42, 123, 456, then NSE/KGE/PBIAS on that series.

### KGE (fine-tune / walk-forward)

| Setting | FT KGE | WF KGE |
|---------|--------|--------|
| Baseline | **0.112** | **0.161** |
| Prune only | 0.024 | 0.080 |
| Combined ensemble | 0.034 | 0.117 |

---

## What helped / what did not

1. **(b) Forget-gate bias +3.0** — already present in the baseline; no further gain available from “turning it on.”
2. **(a) Pruning 17 → 8 donors** — alone **did not** improve continuous skill. Walk-forward NSE fell 0.179 → 0.138. On this partial Midwest extract, the larger similar-donor set still helps.
3. **(c) Mean temperature** — with the same 8 donors, mean-T + multi-seed ensemble beat prune-only (WF NSE 0.165 vs 0.138) but still trailed the 17-donor baseline (0.179). Dimensionality reduction helped relative to the pruned full-feature run, not relative to the richer donor pool.
4. **(d) Multi-seed averaging** — stabilized scores slightly vs any single seed (ensemble FT NSE 0.133 vs per-seed ~0.12) without changing the ranking vs baseline.
5. **Core hypothesis still holds** in every optimized setting: transfer (fine-tune / walk-forward) ≫ local-from-scratch under the 2-year scarce-data protocol.

---

## Donor IDs (8-basin pruned pool)

`05593900`, `05593575`, `05595730`, `05514500`, `05495500`, `03346000`, `06892000`, `06885500`

Selection: static-attribute similarity within Midwest bbox, 50 km target buffer, then keep basins that exist in the local HUC `04/05/07/10` extract.

---

## Recommendation for the short paper / supervisor reply

- Report that the feedback was implemented and stress-tested.
- Keep **`midwest_mini` (17 donors)** as the primary Lick Creek benchmark: best continuous skill and still computationally light (~minutes on Apple M4).
- Treat the optimized package as a **negative control / sensitivity**: smaller donor pools and mean-T did not beat the already-lean Midwest pilot here.
- If pursuing Pool et al.–style pruning further, prefer a fuller CAMELS extract (so the true top-6/7 neighbors are on disk) before concluding that pruning fails for this basin.

---

## How to reproduce

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Combined a+c+d (8 donors, mean T, 3 seeds)
python scripts/run_midwest_opt.py

# Ablation: prune only (full Tmin/Tmax)
python scripts/run_experiment.py --config configs/midwest_opt_ablate/pretrain_prune_fulltemp.yaml
# …then zero_shot / finetune / local / eval / walk_forward configs in that folder
```
