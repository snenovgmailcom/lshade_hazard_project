# Experiments: L-SHADE on CEC2017

This directory contains **experimental results** for baseline **L-SHADE** on the **CEC2017 benchmark suite**.

For each dimension, results are summarized over **51 independent runs** with adaptive population size (`N_init → N_min = 4`, seed = 42 + independent runs).
Only aggregated statistics are stored here; raw per-run data are not tracked in this repository.

---

## D = 10

| Function | Best (mean ± std) | Error (median) | Runs | Final Pop |
|---------|-------------------|----------------|------|-----------|
| f1  | 100.0 ± 0.0 | 0.0 | 51 | 4 |
| f5  | 502.91 ± 0.88 | 2.99 | 51 | 4 |
| f7  | 712.27 ± 0.69 | 12.37 | 51 | 4 |
| f10 | 1031.94 ± 43.82 | 15.20 | 51 | 4 |
| f12 | 1253.44 ± 63.49 | 11.17 | 51 | 4 |
| f21 | 2248.19 ± 50.18 | 114.46 | 51 | 4 |
| f30 | 3408.49 ± 23.99 | 394.01 | 51 | 4 |

➡ Full results: [`D10/summary_lshade.csv`](D10/summary_lshade.csv)

---

## D = 30

| Function | Best (mean ± std) | Error (median) | Runs | Final Pop |
|---------|-------------------|----------------|------|-----------|
| f1  | 100.0 ± 0.0 | 0.0 | 51 | 4 |
| f5  | 506.60 ± 1.44 | 7.01 | 51 | 4 |
| f7  | 737.63 ± 1.27 | 37.60 | 51 | 4 |
| f10 | 2457.59 ± 185.70 | 1417.15 | 51 | 4 |
| f12 | 2200.04 ± 308.29 | 958.52 | 51 | 4 |
| f21 | 2308.01 ± 1.37 | 208.11 | 51 | 4 |
| f30 | 5085.09 ± 109.83 | 2039.44 | 51 | 4 |

➡ Full results: [`D30/summary_lshade.csv`](D30/summary_lshade.csv)

---

## D = 50

| Function | Best (mean ± std) | Error (median) | Runs | Final Pop |
|---------|-------------------|----------------|------|-----------|
| f13 | 1368.72 ± 32.45 | 55.81 | 51 | 4 |
| f15 | 1546.70 ± 11.82 | 44.90 | 51 | 4 |
| f18 | 1850.73 ± 17.35 | 50.16 | 51 | 4 |
| f21 | 2313.79 ± 2.45 | 213.99 | 51 | 4 |
| f24 | 2910.19 ± 2.62 | 509.84 | 51 | 4 |
| f26 | 3790.52 ± 42.44 | 1192.04 | 51 | 4 |
| f30 | 699410.83 ± 101898.90 | 678570.85 | 51 | 4 |

➡ Full results: [`D50/summary_lshade.csv`](D50/summary_lshade.csv)

---

## Notes

- All experiments use **51 independent runs**.
- Population size adapts from `N_init = 18*D` to `N_min = 4` (standard L-SHADE).
- Reported values follow **CEC2017 conventions**.
- These summaries serve as **baseline reference data** for:
  - hazard estimation,
  - witness regime analysis,
  - first-hitting-time survival curves.

For exact numeric values and all benchmark functions, consult the corresponding `summary_lshade.csv` files.
