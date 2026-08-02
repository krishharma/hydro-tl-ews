# Known limitations

This project has undergone several protocol fixes (walk-forward
`refit_train_start`, validation tails, forward-looking EWS labels). Remaining
caveats:

## Science / evaluation

1. **EWS is a post-process on same-day hindcasts**, not a true multi-day
   forecast issued at time *t* without future forcings. Lead windows score
   whether predicted flow in `[t+1, t+L]` crosses thresholds.
2. **Compound warning probabilities** use an independence assumption
   (`1 − ∏(1 − p_i)`), which overstates probability under autocorrelated flow.
3. **Static normalizers** in transfer stages are fit on all CAMELS attributes
   (including held-out targets), not the donor-only normalizer from pretrain.
   Approach A (frozen LSTM) is most sensitive.
4. **Dynamic normalizer periods** still differ slightly between fine-tune
   (warmup only) and walk-forward (forcings through `initial_train_end`).
5. **SHAP** uses a simplified DeepExplainer path (`base_value=0`, mean over
   time) on the pre-walk-forward checkpoint — treat as qualitative.
6. **Pretrain validation years** (2010–2014 donors) overlap the target eval
   climate era (targets themselves are held out).

## Runtime / packaging

1. Paper builders need optional packages (`reportlab`, `python-docx`) and
   populated `results/`; they are not part of the core pipeline.
2. Full CAMELS pretrain is intentionally separate from
   `run_full_pipeline.sh` (which uses `pretrain_subset200` so checkpoint
   names match the single-target YAMLs).
3. `num_workers > 0` can be flaky on macOS DataLoaders; default remains `0`
   except where configs set otherwise for pretrain.

## Intentionally out of scope

- Live operational forecasting service / API
- Docker / cloud IaC
- True multi-site regional frequency analysis (Hosking–Wallis); thresholds are
  at-site long-record quantiles
