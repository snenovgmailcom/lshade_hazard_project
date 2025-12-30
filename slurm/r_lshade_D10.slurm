#!/bin/bash
#SBATCH --job-name=r_D10
#SBATCH --output=slurm/logs/%j_D10.out
#SBATCH --error=slurm/logs/%j_D10.err
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=32G
#SBATCH --partition=compute
#SBATCH --mail-user=nenov@uctm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
set -x  # optional debugging

# Activate environment
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate intelpython

# Navigate to project
cd "$HOME/lshade_hazard_project"

# Prepare directories
mkdir -p slurm/logs
mkdir -p experiments

# Avoid BLAS/OpenMP oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUTDIR="experiments/D10"
mkdir -p "$OUTDIR"

echo "=== R-LSHADE D=10 started at $(date) on $(hostname) ==="
echo "Output dir: $OUTDIR"

CMD=(
  python benchmarks/benchmark.py
  --dim 10
  --functions all
  --runs 51
  --jobs 180
  --max-evals 100000
  --seed-start 42
  --outdir "$OUTDIR"
)

printf '%q ' "${CMD[@]}" > "$OUTDIR/run_command.sh"
echo >> "$OUTDIR/run_command.sh"
chmod +x "$OUTDIR/run_command.sh"

"${CMD[@]}"

echo "=== R-LSHADE D=10 finished at $(date) ==="
