#!/bin/bash
set -euo pipefail

mkdir -p results/validation/D10

R_CONC=2.0
EPS=100

for f in f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20; do
  pkl="experiments/D10/${f}/${f}.pkl"
  outdir="results/validation/D10/${f}"
  
  [[ -f "$pkl" ]] || continue
  mkdir -p "$outdir"
  
  echo "Processing $f (eps=$EPS, r_conc=$R_CONC)..."
  
  python analysis/validate_morse.py --pkl "$pkl" --func "$f" --dim 10 --eps "$EPS" \
    --r_conc "$R_CONC" --window 60 --sum_mode --output "${outdir}/${f}_sum.json"
done

echo "Done."
