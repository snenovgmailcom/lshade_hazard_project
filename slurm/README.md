# SLURM Job Scripts

This directory contains SLURM batch scripts used to run L-SHADE experiments on HPC clusters.

The scripts are designed for reproducible benchmarking on the CEC2017 test suite.
They are used to generate the experimental data summarized in `experiments/*/summary_lshade.csv`.

---

## Contents

- `r_lshade_D10.slurm`  
  SLURM batch script for running L-SHADE experiments in **dimension D = 10**.


- `r_lshade_D30.slurm`  
  SLURM batch script for running L-SHADE experiments in **dimension D = 30**.

- `r_lshade_D50.slurm`  
  SLURM batch script for running L-SHADE experiments in **dimension D = 50**.

- `r_lshade_D50.slurm`  
  SLURM batch script for running L-SHADE experiments in **dimension D = 100**.

(Older or temporary scripts are intentionally not tracked.)

---

## Typical Usage

Submit a job using:

```bash
sbatch r_lshade_D30.slurm
