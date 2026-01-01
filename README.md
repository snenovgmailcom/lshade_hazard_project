# L-SHADE Hazard Analysis

Empirical and theoretical analysis of first-hitting times in L-SHADE differential evolution, supporting the paper:

> **"Conditional Hazard Analysis of First-Hitting Times in Differential Evolution"**  
> S. Nenov, S. Nedanovski, D. Pilev (2025)

## Overview

This project provides:
- **Theoretical framework**: Hazard-based bounds for first-hitting times using the L-SHADE witness event
- **Empirical analysis**: Kaplan-Meier survival curves, witness frequency estimation, and gap analysis
- **Benchmarking**: L-SHADE implementation with history logging on CEC2017 functions

## Project Structure
```
lshade_hazard_project/
├── algorithms/
│   └── lshade.py              # L-SHADE with memory history logging
├── benchmarks/
│   └── benchmark.py           # CEC2017 benchmark runner
├── analysis/
│   ├── __init__.py
│   ├── km_analysis.py         # Kaplan-Meier survival analysis
│   ├── witness_frequency.py   # Gamma estimation (L2, L3 checking)
│   ├── gap_analysis.py        # Theoretical vs empirical comparison
│   └── generate_tables.py     # Generate paper tables/figures
├── experiments/               # Raw PKL results (git-ignored)
├── results/tables/            # Generated CSV/LaTeX tables
├── slurm/                     # SLURM job scripts for HPC
└── requirements.txt
```

## Installation
```bash
git clone https://github.com/snenovgmailcom/lshade_hazard_project.git
cd lshade_hazard_project
pip install -r requirements.txt
```

Developed with Python 3.10+ using [Intel Distribution for Python](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html).

## Usage

### 1. Run Benchmarks
```bash
python benchmarks/benchmark.py --dim 10 --functions all --runs 51 --max-evals 100000

# Or via SLURM on HPC
sbatch slurm/r_lshade_D10.slurm
sbatch slurm/r_lshade_D30.slurm
sbatch slurm/r_lshade_D50.slurm
sbatch slurm/r_lshade_D100.slurm
```

### 2. Generate Analysis
```bash
# Single dimension
python -m analysis.generate_tables --pkl experiments/D10/raw_results_lshade.pkl --dim 10

# All dimensions
python -m analysis.generate_tables --all-dims --exp-dir experiments
```

### 3. Output Files

Generated in `results/tables/`:

| File | Description |
|------|-------------|
| `full_results.csv` | All metrics for all functions |
| `success_rates.csv` | Hit rates by function × ε |
| `km_table_eps{X}.csv` | KM statistics |
| `gamma_table_eps{X}.csv` | Witness frequency and gap analysis |
| `survival_curves_eps{X}.png` | KM survival plots |

## L-SHADE Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| N_init | 18d | Initial population size |
| N_min | 4 | Minimum population size |
| H | 6 | Memory slots |
| p | 0.11 | p-best fraction |
| arc_rate | 2.6 | Archive rate |

## Acknowledgments

We gratefully acknowledge the **Discoverer Petascale Supercomputer** at Sofia Tech Park for providing access to high-performance computing resources.

This study is financed by the European Union–NextGenerationEU through the National Recovery and Resilience Plan of the Republic of Bulgaria, project **BG-RRP-2.004-0002 "BiOrgaMCT"**.

### AI Assistance Disclosure

Portions of the code in this repository were developed with the assistance of AI tools (Claude, Anthropic). All AI-generated code has been reviewed, tested, and validated by the authors to ensure correctness and alignment with the research objectives.

## Citation
```bibtex
@article{nenov2025hazard,
  title={Conditional Hazard Analysis of First-Hitting Times in Differential Evolution},
  author={Nenov, S. and Nedanovski, S. and Pilev, D.},
  year={2025}
}
```

## License

MIT
