#!/usr/bin/env bash
# Build docs/short_paper.tex -> docs/Short_Paper_Hydro_TL_EWS.pdf
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/docs"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found. Install a TeX distribution (MacTeX, TeX Live)." >&2
  exit 1
fi

pdflatex -interaction=nonstopmode short_paper.tex >/dev/null
pdflatex -interaction=nonstopmode short_paper.tex >/dev/null

cp -f short_paper.pdf Short_Paper_Hydro_TL_EWS.pdf
echo "Wrote docs/Short_Paper_Hydro_TL_EWS.pdf ($(pdfinfo Short_Paper_Hydro_TL_EWS.pdf 2>/dev/null | awk '/Pages:/ {print $2}') pages)"
