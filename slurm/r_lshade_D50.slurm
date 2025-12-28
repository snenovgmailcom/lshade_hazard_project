#!/bin/bash
#SBATCH --job-name=r_lshade_D50
#SBATCH --output=slurm/logs/r_lshade_D50_%j.out
#SBATCH --error=slurm/logs/r_lshade_D50_%j.err
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=180
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --mail-user=nenov@uctm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
set -x

# Load conda
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate intelpython

# Go to project
cd "$HOME/lshade_project"

# Create directories
mkdir -p slurm/logs
mkdir -p experiments

# Avoid BLAS/OpenMP oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Output folder
OUTDIR="experiments/r_lshade_D50"
mkdir -p "$OUTDIR"

echo "=== R-LSHADE D=50 started at $(date) on $(hostname) ==="
echo "Output dir: $OUTDIR"

CMD=(
  python benchmarks/r_benchmark.py
    --dim 50
    --functions all
    --runs 51
    --jobs 180
    --max-evals 500000
    --seed-start 42
    --outdir "$OUTDIR"
)

# Save invoked command
printf '%q ' "${CMD[@]}" > "$OUTDIR/run_command.sh"
echo >> "$OUTDIR/run_command.sh"
chmod +x "$OUTDIR/run_command.sh"

# Run
"${CMD[@]}"

echo "=== R-LSHADE D=50 finished at $(date) ==="
