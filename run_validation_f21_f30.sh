#!/bin/bash
set -euo pipefail

mkdir -p results/validation/D10

R_CONC=3.0
EPS=500

for f in f21 f22 f23 f24 f25 f26 f27 f28 f29 f30; do
  pkl="experiments/D10/${f}/${f}.pkl"
  outdir="results/validation/D10/${f}"
  
  [[ -f "$pkl" ]] || continue
  mkdir -p "$outdir"
  
  echo "Processing $f (eps=$EPS, r_conc=$R_CONC)..."
  
  python analysis/validate_morse.py --pkl "$pkl" --func "$f" --dim 10 --eps "$EPS" \
    --r_conc "$R_CONC" --sum_mode --output "${outdir}/${f}_sum.json"
done

echo "Done."
