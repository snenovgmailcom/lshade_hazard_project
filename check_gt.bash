DIM=10
EPS_IN=10
R_CONC=2.0
BASE="experiments/D10"
OUT="results/witness/D10/all_eps${EPS_IN}_r${R_CONC}"
mkdir -p "$OUT"

for i in $(seq 1 30); do
  f="f${i}"
  echo "=== Running ${f} ==="
  python analysis/check_Gt.py \
    --pkl "${BASE}/${f}/${f}.pkl" \
    --func "${f}" --dim "${DIM}" \
    --eps_in "${EPS_IN}" \
    --r_conc "${R_CONC}" \
    --quiet \
    --outdir "${OUT}/${f}"
done
