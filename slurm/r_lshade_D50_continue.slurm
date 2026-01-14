#!/bin/bash
#SBATCH --job-name=D50_cont
#SBATCH --output=slurm/logs/D50_cont_%j.out
#SBATCH --error=slurm/logs/D50_cont_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --mail-user=nenov@uctm.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate intelpython

cd "$HOME/lshade_hazard_project"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "=== D50 continuation (f12-f30) started at $(date) ==="

python benchmarks/benchmark.py \
    --dim 50 \
    --functions "12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30" \
    --runs 51 \
    --jobs 32 \
    --max-evals 500000 \
    --seed-start 42 \
    --outdir experiments/D50

echo "=== D50 continuation finished at $(date) ==="
