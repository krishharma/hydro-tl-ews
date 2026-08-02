#!/usr/bin/env bash
# Run the single-target CAMELS pipeline (Merced / Happy Isles).
# Default chain uses the 200-donor pretrain so downstream configs' checkpoint
# paths match (pretrain_subset200.pt → finetune_*.pt → walk_forward).
# For full CAMELS regional pretrain, run configs/pretrain.yaml separately and
# point later configs at results/checkpoints/pretrain.pt (or use multi-target).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs results/checkpoints results/history

LOG="results/logs/full_pipeline.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== hydro_tl_ews full pipeline started $(date -Iseconds) ==="
python -c "
import torch
from hydro_tl_ews.utils.device import training_device
print('torch', torch.__version__, 'device', training_device())
"

if [[ ! -d data/basin_dataset_public_v1p2 ]]; then
  echo "ERROR: CAMELS not found under data/. See data/README.md"
  exit 1
fi

# Ensure src/ is importable when not installed editable.
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:$PYTHONPATH}"

STAGES=(
  pretrain_subset200
  zero_shot
  finetune_conservative
  finetune_progressive
  local_baseline
  walk_forward
  min_data_sensitivity
)

for stage in "${STAGES[@]}"; do
  echo "=== stage=${stage} $(date -Iseconds) ==="
  python scripts/run_experiment.py --config "configs/${stage}.yaml"
done

echo "=== pipeline finished OK $(date -Iseconds) ==="
