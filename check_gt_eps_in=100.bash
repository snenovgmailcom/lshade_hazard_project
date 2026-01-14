EPS_IN=100
R_CONC=2.0
OUT="results/witness/D10/all_eps${EPS_IN}_r${R_CONC}"
mkdir -p "$OUT"

for i in $(seq 1 30); do
  f="f${i}"
  echo "=== Running ${f} ==="
  python analysis/check_Gt.py \
    --pkl "experiments/D10/${f}/${f}.pkl" \
    --func "${f}" --dim 10 \
    --eps_in "${EPS_IN}" \
    --r_conc "${R_CONC}" \
    --quiet \
    --outdir "${OUT}/${f}"
done
