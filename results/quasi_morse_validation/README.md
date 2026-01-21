# L-SHADE Hazard Analysis and Quasi-Morse Validation

This repository accompanies a series of theoretical and empirical studies on
**first-hitting times, hazard bounds, and witness regimes in L-SHADE**
(Differential Evolution with Linear Population Size Reduction).

The focus is on understanding *when and why* L-SHADE rapidly converges after a
deep hit into a low-level sublevel set, and on identifying the empirical regimes
under which theoretical success-probability bounds become non-vacuous.

## Repository structure

lshade_hazard_project/
├── algorithms/ # L-SHADE variants and mutation logic
├── benchmarks/ # CEC2017 benchmark wrappers
├── analysis/ # Hazard, KM, cluster, and witness analyses
├── experiments/ # Raw experimental outputs (PKL)
├── results/
│ ├── quasi_morse_validation/ # Supplementary quasi-Morse results (CSV + figures)
│ └── ... # Other result folders
├── validate_quasi_morse.py # Main script for quasi-Morse validation
├── paper/ # LaTeX sources (main + supplementary)
└── README.md # This file

## Context

### Witness regime and hazard bounds
The theory developed in the accompanying papers shows that once L-SHADE reaches
a sufficiently deep sublevel set
\[
A_{\varepsilon_{\mathrm{deep}}} = \{x : f(x) \le f^\star + \varepsilon_{\mathrm{deep}}\},
\]
the probability of further improvement depends critically on:

- geometric concentration of donors around a *witness* individual,
- existence of at least two nearby donors (the $C_2$ event),
- emergence of non-vacuous donor-pair probabilities.

This repository provides both:
1. **Formal hazard-based bounds** (main paper), and
2. **Empirical validation** of the geometric conditions under which those bounds apply
   (supplementary material).

## Quasi-Morse regime

CEC benchmark functions are **not Morse functions** in the strict analytical sense:
they may exhibit flat regions, non-isolated critical points, or numerical artifacts.
Nevertheless, empirical L-SHADE trajectories often display behavior *characteristic*
of Morse functions after a sufficiently deep hit.

We therefore use the term **quasi-Morse regime** to denote an *empirical phase* in which,
after the first entrance into \(A_{\varepsilon_{\mathrm{deep}}}\),

- the population rapidly concentrates around a single witness,
- donor distances stabilize at a small scale,
- $C_2$ and donor-pair events occur with high probability.

All Morse-based arguments are applied **conditionally on the observed occurrence of this
regime**, without assuming global Morse regularity of the objective function.

## Supplementary results: quasi-Morse validation

The directory

results/quasi_morse_validation/

contains CSV summaries and figures validating the quasi-Morse regime on CEC2017 for dimensions \(D = 10, 30, 50\).

For each dimension we report:
- **τ_deep**: first hit into \(A_{\varepsilon_{\mathrm{deep}}}\),
- **τ_C2**: first time at least one donor lies within \(r_{\mathrm{conc}}\),
- **τ_pair**: first time at least two donors lie within \(r_{\mathrm{conc}}\).

Some benchmark functions are **intentionally absent**:
a function is excluded if no valid τ_deep sample exists
(missing history, missing PKL, or no deep hit).
This highlights the limits of the quasi-Morse regime.

Representative convergence plots include:
- **f6, f11, f22** (positive quasi-Morse examples),
- **f10** (counterexample: deep hit without geometric concentration).

These figures are discussed in the supplementary material.

## Reproducibility

### Running quasi-Morse validation

```bash
python validate_quasi_morse.py \
  --dim 10 \
  --func all \
  --base experiments \
  --outdir results/quasi_morse_validation/d10 \
  --plots
Repeat with --dim 30 and --dim 50.

The margin parameter is fixed to margin = 60 across dimensions.
This choice is deliberately conservative and intended to expose difficult cases,
not to optimize detection of concentration.

Notes on environments
Experiments were run using Intel Python / Conda.
Minor numerical differences may occur with system Python due to BLAS/LAPACK
implementations and random-number backends.

Citation
If you use this code or results, please cite the accompanying paper(s)
and the official CEC2017 benchmark suite.

Disclaimer
This repository prioritizes scientific transparency and diagnostic clarity
over benchmark leaderboard performance.
Empirical results are intended to validate theoretical regimes,
not to claim universal dominance of any algorithm variant.
