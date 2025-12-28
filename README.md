# L-SHADE Hazard Analysis

Hazard-based analysis of first-hitting times in L-SHADE differential evolution.

## Paper

"On the Probability of First Success in Differential Evolution: Hazard Identities and Tail Bounds"

## Project Structure

```
lshade_hazard_project/
├── algorithms/
│   └── r_lshade.py          # L-SHADE implementation with history logging
├── benchmarks/
│   └── r_benchmark.py       # CEC2017 benchmark runner
├── analysis/
│   ├── km_analysis.py       # Kaplan-Meier survival analysis
│   ├── witness_frequency.py # Gamma_t estimation
│   ├── gap_analysis.py      # Theory vs practice comparison
│   └── generate_tables.py   # Generate paper tables/figures
├── experiments/             # Raw results (git-ignored)
├── results/                 # Figures and tables
└── slurm/                   # SLURM job scripts
```

## Quick Start

### 1. Run Benchmarks

```bash
# Single dimension
python benchmarks/r_benchmark.py --dim 10 --functions all --runs 51 --max-evals 100000

# Or submit SLURM jobs
sbatch slurm/r_lshade_D10.slurm
sbatch slurm/r_lshade_D30.slurm
sbatch slurm/r_lshade_D50.slurm
```

### 2. Generate Analysis

```bash
# Single dimension
python -m analysis.generate_tables --pkl experiments/r_lshade_D10/raw_results_lshade.pkl --dim 10

# All dimensions
python -m analysis.generate_tables --all-dims --exp-dir experiments
```

### 3. Output

Results are saved to `results/tables/`:
- `success_rates.csv` - Success rates by function and epsilon
- `km_table_eps{X}.csv` - KM statistics for each epsilon
- `gamma_table_eps{X}.csv` - Witness frequency and gap analysis
- `survival_curves_eps{X}.png` - KM survival plots

## Key Parameters

### L-SHADE (Table II of Tanabe & Fukunaga 2014)
- `N_init = 18 * d`
- `N_min = 4`
- `H = 6` (memory size)
- `p = 0.11` (p-best fraction)
- `arc_rate = 2.6`

### Witness Frequency Thresholds
- `F_minus = 0.1, F_plus = 0.9`
- `g_minus = 0.1` (Cauchy density lower bound)
- `c_cr = 0.5` (CR threshold)
- `q_minus = 0.25` (CR tail probability)

## Citation

```bibtex
@article{nedanovski2025hazard,
  title={On the Probability of First Success in Differential Evolution: 
         Hazard Identities and Tail Bounds},
  author={Nedanovski, S. and Nenov, S. and Pilev, D.},
  year={2025}
}
```
