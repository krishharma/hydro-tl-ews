#!/usr/bin/env bash
# Copy data-focused paper figures into docs/paper_images/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/figures"
DST="$ROOT/docs/paper_images"

mkdir -p "$DST"

FILES=(
  fig_camels_map.png
)

for f in "${FILES[@]}"; do
  cp "$SRC/$f" "$DST/$f"
  echo "copied $f"
done

# Remove figures no longer used by the paper
for f in fig1_architecture.png fig2_walk_forward.png fig3_unfreezing.png \
         fig4_rfa_thresholds.png fig_sample_hydrograph.png; do
  rm -f "$DST/$f"
done

echo "Done. Data figures are in docs/paper_images/"
