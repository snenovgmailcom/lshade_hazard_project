# Benchmarks (CEC2017 only)

This directory provides a **single-purpose benchmarking harness** for running
**classical L-SHADE** on the **CEC2017** benchmark suite.

- The benchmark runner is `benchmark.py`.
- The optimizer implementation is imported from `algorithms/lshade.py`.
- This harness is intentionally scoped to **CEC2017 only** (functions f1–f30, included depricated function f2 in dimensions 10, 30 and 50, only).

---

## Benchmark Suite

### Official CEC2017 reference

All function definitions and evaluation criteria follow the official CEC2017 problem definition/evaluation criteria document [1].

### Upstream reference implementation / materials

This project uses the upstream CEC2017 materials and/or compatible wrappers as referenced in [2]

---

## What `benchmark.py` does

`benchmark.py` runs L-SHADE on a selected subset of CEC2017 functions (or all f1–f30),
for a fixed dimension `D`, across `--runs` independent seeds.

Design choices:

- **Functions processed sequentially** (f1, f2, …), but **seeds parallelized per function** using a `ProcessPoolExecutor`.
- Results are persisted **immediately after each function** as a per-function `.pkl`, reducing risk of losing long runs.
- A summary CSV is created at the end, and can also be rebuilt later using `--summary-only`.

---

## Outputs (directory layout)

For dimension `D`, the default output directory is:

- `experiments/D{D}/`

Inside it:

- Per-function pickle:
  - `experiments/D{D}/f{i}/f{i}.pkl`
- Summary CSV (aggregated across seeds):
  - `experiments/D{D}/summary_lshade.csv`

Optional (if `--figs` is enabled):

- Envelope plot:
  - `experiments/D{D}/cec2017_f{i}_lshade_envelope.png`

---

## Command-line options (CLI):

- `--dim INT`  
  Dimension `D` (e.g., 10, 30, 50, 100).

- `--max-evals INT`  
  Maximum number of function evaluations (defaults to `10000 * D` if omitted).

- `--runs INT`  
  Number of independent runs (seeds) per function (default 51).

- `--seed-start INT`  
  First seed; run `r` uses seed `seed_start + r` (default 42).

- `--jobs INT`  
  Parallel worker processes (per function). Actual workers are
  `min(jobs, os.cpu_count())`.

- `--functions STR`  
  Which functions to run:
  - `all` (default): run f1–f30
  - comma-separated list like `f1,f3,f10` (also accepts `1,3,10`)

- `--outdir PATH`  
  Output root directory (default `experiments/D{D}`).

- `--popsize INT`  
  Initial population size. If set to `-1` (default), uses `--init-factor * D`.

- `--init-factor FLOAT`  
  `N_init = init_factor * D` when `--popsize = -1` (default 18.0).

- `--N-min INT`  
  Minimum population size for L-SHADE (default 4).

- `--disp`  
  Verbose printouts from the solver.

- `--figs`  
  Generate convergence envelope plots (median with min–max band).

- `--summary-only`  
  Do not run optimization; only rebuild `summary_lshade.csv` from existing `.pkl` files.

---

## Examples

Run one function:

```bash
python benchmarks/benchmark.py --dim 10 --functions f1 --runs 51 --jobs 32
```

Run all functions

```bash
python benchmarks/benchmark.py --dim 10 --functions all --runs 51 --jobs 160
```

---

## AI Assistance Disclosure

Portions of this repository (including benchmarking utilities and documentation)
were developed with the assistance of AI tools (Claude, Anthropic). All AI-assisted
contributions were reviewed, tested, and validated by the authors to ensure correctness,
reproducibility, and alignment with the research objectives.

---

## REFERENCES

[1] N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu, and P. N. Suganthan,
*Problem Definitions and Evaluation Criteria for the CEC 2017 Special Session and
Competition on Single Objective Real-Parameter Numerical Optimization*,
Technical Report, Nanyang Technological University, Singapore, 2016.

[2] P. N. Suganthan (GitHub), **CEC2017-BoundContrained** repository. 