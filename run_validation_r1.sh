#!/bin/bash
set -euo pipefail

rm -rf results/validation/D10
mkdir -p results/validation/D10

R_CONC=1.0

# Function difficulty groups
easy="f1 f2 f3 f4 f6 f9"
medium="f5 f7 f8 f10"

get_eps() {
  local f=$1
  for e in $easy; do [[ "$f" == "$e" ]] && echo 10 && return; done
  for m in $medium; do [[ "$f" == "$m" ]] && echo 100 && return; done
  echo 100
}

for f in f1 f2 f3 f4 f5 f6 f7 f8 f9 f10; do
  pkl="experiments/D10/${f}/${f}.pkl"
  outdir="results/validation/D10/${f}"
  
  [[ -f "$pkl" ]] || continue
  mkdir -p "$outdir"
  
  eps=$(get_eps "$f")
  echo "Processing $f (eps=$eps, r_conc=$R_CONC)..."
  
  python analysis/validate_morse.py --pkl "$pkl" --func "$f" --dim 10 --eps "$eps" \
    --r_conc "$R_CONC" --output "${outdir}/${f}_best.json"
  
  python analysis/validate_morse.py --pkl "$pkl" --func "$f" --dim 10 --eps "$eps" \
    --r_conc "$R_CONC" --set_of_good_slots --output "${outdir}/${f}_good.json"
  
  python analysis/validate_morse.py --pkl "$pkl" --func "$f" --dim 10 --eps "$eps" \
    --r_conc "$R_CONC" --sum_mode --output "${outdir}/${f}_sum.json"
done

echo "Done. Results in results/validation/D10/"
