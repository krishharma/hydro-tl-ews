# Midwest mini-study: observations vs project objectives

**Date:** 2026-08-07  
**Scope (per supervisor):** optimized laptop run on Midwest CAMELS basins — not the full continental dataset.  
**Target:** USGS `05507600` (Lick Creek near Perry, MO; continental plains).  
**Donors:** 17 hydrologically similar basins inside Midwest bbox `[36.5, 49.5, −104, −80.5]` (from a 163-basin geographic pool; HUC `04/05/07/10` Daymet + streamflow extract).  
**Hardware:** Apple M4, MPS, ~16 GB RAM.  
**Artifacts:** `results/midwest_mini/summary.json` and sibling metric/prediction files.  
**Runner:** `python scripts/run_midwest_mini.py`

Original project brief asked for full CAMELS pretrain and a Sierra/Rockies snowmelt target. Supervisor direction explicitly replaced that with a Midwest subset sized to this PC. Objectives below are judged against that agreed scope.

---

## Results (eval window 2011–2014 unless noted)

| Setting | NSE | KGE | PBIAS (%) |
|---------|-----|-----|-----------|
| Zero-shot (pretrained only) | 0.150 | 0.032 | −37.5 |
| Conservative fine-tune (head only, 2-yr warmup) | 0.160 | 0.112 | +4.1 |
| Local baseline (from scratch on warmup) | 0.009 | −0.351 | +72.6 |
| Walk-forward ops protocol (17× 90-day refits) | 0.179 | 0.161 | −8.7 |

Early-warning (walk-forward hindcast probabilities):

| Event | Lead | AUC | F1@0.5 | Brier |
|-------|------|-----|-------|-------|
| Flood Q95 | 1 d | 0.922 | 0.286 | 0.094 |
| Flood Q95 | 3 d | 0.914 | 0.133 | 0.374 |
| Flood Q95 | 7 d | 0.908 | 0.219 | 0.693 |
| Flood Q99 | 1 d | 0.950 | n/a (rare) | 0.005 |
| Drought Q5 | 1 d | 0.805 | 0.410 | 0.214 |
| Drought Q5 | 7 d | 0.819 | 0.276 | 0.803 |

Pretrain validation NSE-loss improved steadily (epoch 1 → 10: val 0.94 → 0.78) on 17 Midwest donors, seq length 90, hidden 128.

---

## Objective-by-objective

From the original guidelines (§1.5 Research Objectives):

### 1. Regional pre-training pipeline
**Met under optimized scope.** An EA-LSTM was pretrained on real CAMELS Midwest donors (not synthetic sample data). Full 671-basin continental pretrain was intentionally skipped per supervisor guidance.

### 2. Fine-tuning for a data-scarce target
**Met.** Conservative fine-tuning (LSTM frozen, head trained on ~2 years of local Q) ran successfully on `05507600`. Note: original brief preferred a snowmelt Sierra/Rockies target; supervisor directed Midwest plains instead — still a data-scarce 2-year warmup protocol.

### 3. Quantify the value of transfer learning
**Met — transfer clearly helps.** On the same 2011–2014 window:

- Fine-tuned transfer **beats local-from-scratch** by a large margin (NSE 0.160 vs 0.009; KGE 0.112 vs −0.351).
- Fine-tune also improves over zero-shot on KGE/PBIAS (KGE 0.112 vs 0.032; PBIAS nearly unbiased at +4% vs −37%).
- Absolute continuous skill is modest (NSE ~0.16–0.18). That is expected for a small regional donor set, short sequences (90 d), and a flashier plains hydrograph — but the **ranking** required by the objective is clear: transfer ≫ local.

### 4. Rigorous walk-forward validation
**Met.** Rolling-origin evaluation with `refit_train_start=2009-01-01` (honest scarce-data refits), 17 refits over 2011–2014, online bias correction. Walk-forward continuous skill (NSE 0.179, KGE 0.161) is the best continuous result in this study.

### 5. Probabilistic early-warning skill
**Partially met.** The EWS layer runs end-to-end with RFA-style thresholds from the long local record and reports AUC / F1 / Brier at leads 1/3/7.

- Discrimination is strong for flood Q95/Q99 (AUC ≈ 0.91–0.95) and good for drought Q5 (AUC ≈ 0.81).
- Calibration/decision utility is weaker: F1@0.5 is low, and Brier scores degrade sharply at longer leads (as expected when warnings are derived from hindcast flow rather than multi-day weather forecasts).
- Q99 F1 is undefined (too few events at the 0.5 probability threshold) — a known rarity issue, not a pipeline failure.

### 6. Explainable AI (SHAP)
**Met.** GradientExplainer was run after walk-forward (200 background / 100 sample sequences). Top mean-|SHAP| drivers were all dynamic weather features:

1. precipitation  
2. vapor pressure  
3. shortwave radiation  
4. day length  
5. Tmin / Tmax  

Static catchment attributes had mean-|SHAP| = 0 in this single-basin explanation. That is expected when statics do not vary within one gauge; the model can still use them via the EA-LSTM input gate, but day-to-day attribution highlights weather. Precip ranking first is hydrologically sensible for this plains basin. Output: `results/midwest_mini/shap_global_importance.csv`.

---

## Overall verdict

**Yes — under the supervisor’s optimized Midwest scope, all six research objectives are addressed.**

The pipeline demonstrates the intended story:

1. Regional EA-LSTM pretraining on real CAMELS data works on a laptop-sized Midwest extract.  
2. With only ~2 years of local data, transfer learning outperforms training from scratch.  
3. Walk-forward evaluation provides an operationally honest skill estimate.  
4. Extreme-event warning scores can be produced from the same forecasts.  
5. SHAP attributions are available and precip-led for this target.

It does **not** claim continental CAMELS performance, snowmelt-regime transfer, or multi-basin generalization. Those remain future work if more compute/disk become available.

---

## Practical notes / limitations

- Data: Midwest HUC subset (~200 MB time series) + attributes; not the full ~14 GB CAMELS unpack.
- Model capacity deliberately reduced (hidden 128, seq 90, 10 pretrain epochs) for M4 runtime (~minutes for transfer stages after ~3 min pretrain; SHAP ~tens of seconds with defaults).
- Continuous NSE remains well below published large-sample LSTM benchmarks (those use hundreds of donors and longer sequences).
- Drought threshold Q5 resolved to 0.0 mm/d at this gauge (intermittent/low-flow behavior) — drought labels are therefore “near-zero flow” days.
- Install `pyarrow` if parquet outputs are preferred; this run wrote `walk_forward.csv`.
- SHAP static zeros are a single-basin attribution artifact, not proof that static attributes are unused.

---

## Follow-up: supervisor optimizations (2026-08-08)

See [`OBSERVATIONS_OPT.md`](OBSERVATIONS_OPT.md). Combined donor pruning + mean temperature + 3-seed averaging did **not** beat this 17-donor baseline on continuous NSE/KGE; transfer ≫ local still held. Forget-gate bias +3.0 was already used here.