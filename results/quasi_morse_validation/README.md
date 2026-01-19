# Quasi-Morse Validation

Empirical validation of the concentration-based hazard bound (Theorem 5) from the paper:

> **"On the Probability of First Success in Differential Evolution: Hazard Identities and Tail Bounds"**  
> D. Nedanovski, S. Nenov, D. Pilev (2025)

## Overview

This module validates the **witness-regime concentration conditions** (C1)–(C2) from `thm:morse-hazard` by tracking three critical times:

| Time | Condition | Meaning |
|------|-----------|---------|
| $\tau_{\mathrm{deep}}$ | (C1) | Witness enters $A_{\varepsilon/4}$ |
| $\tau_{C2}$ | (C2) with $c_1 \geq 1$ | First neighbor within $r_{\mathrm{conc}}$ |
| $\tau_{\mathrm{pair}}$ | $c_1 \geq 2$ | Pair bound $c_{\mathrm{pair}} > 0$ becomes non-vacuous |

The key finding: **convergence alone cannot tell you when certified local concentration begins** — the gap between $\tau_{\mathrm{deep}}$ and $\tau_{\mathrm{pair}}$ reveals the transition from exploration to exploitation.

## Directory Structure

```
lshade_hazard_project/
├── experiments/                          # RAW DATA ONLY (git-ignored)
│   ├── D10/f1/f1.pkl
│   ├── D10/f2/f2.pkl
│   └── ...
├── results/                              # ALL ANALYSIS OUTPUT
│   ├── tables/                           # KM analysis tables
│   └── quasi_morse_validation/           # This module's output
│       ├── d10/
│       │   ├── morse_validate_D10_all.per_run.csv
│       │   ├── morse_validate_D10_all.summary.csv
│       │   ├── morse_validate_D10_all.cpair.png
│       │   ├── morse_validate_D10_all.cpair_taupair.png
│       │   ├── morse_validate_D10_all.zones.png
│       │   ├── convergence_D10_f1.png
│       │   ├── convergence_D10_f11.png
│       │   ├── comparison_three_regimes_D10.png
│       │   └── comparison_comprehensive_D10.png
│       ├── d30/
│       ├── d50/
│       └── README.md
└── analysis/                             # Scripts
    ├── morse_validate.py
    ├── plot_convergence_tau.py
    └── plot_convergence_comparison.py
```

## Running Validation

### Quick Start

```bash
# Single dimension, all functions, with plots
python analysis/morse_validate.py --dim 10 --func all --base experiments \
  --outdir results/quasi_morse_validation/d10 --plots

# Or use the runner script
./scripts/run_quasi_morse_validation.sh --dim 10 --func all --plots
```

### Multiple Dimensions

```bash
./scripts/run_quasi_morse_validation.sh --dim 10,30,50 --func all --plots
```

### Subset of Functions

```bash
python analysis/morse_validate.py --dim 10 --func f1,f3,f5,f11,f22 \
  --base experiments --outdir results/quasi_morse_validation/d10 --plots
```

## Output Columns

### Per-Run CSV

| Column | Description |
|--------|-------------|
| `tau_deep` | Generation when (C1) satisfied |
| `tau_C2` | Generation when (C2) satisfied ($c_1 \geq 1$) |
| `tau_pair` | Generation when $c_1 \geq 2$ |
| `gap_C2` | $\tau_{C2} - \tau_{\mathrm{deep}}$ |
| `gap_pair` | $\tau_{\mathrm{pair}} - \tau_{\mathrm{deep}}$ |
| `C2_achieved` | 1 if (C2) achieved, 0 otherwise |
| `pair_achieved` | 1 if $c_1 \geq 2$ achieved, 0 otherwise |
| `L_hat` | Local curvature estimate at witness |
| `r_conc` | Concentration radius $r_{\mathrm{safe}} / (2(F^- + \Delta_F))$ |
| `c_pair_lb_at_tau_pair` | $c_{\mathrm{pair}}$ lower bound at $\tau_{\mathrm{pair}}$ |
| `beta_at_tau_pair` | Concentration fraction at $\tau_{\mathrm{pair}}$ |

### Summary CSV

| Column | Description |
|--------|-------------|
| `n_runs_ok` | Runs reaching $\tau_{\mathrm{deep}}$ |
| `C2_achieved_frac` | Fraction achieving (C2) |
| `pair_achieved_frac` | Fraction achieving $c_1 \geq 2$ |
| `tau_deep_med` | Median $\tau_{\mathrm{deep}}$ |
| `gap_pair_med` | Median $\tau_{\mathrm{pair}} - \tau_{\mathrm{deep}}$ |
| `c_pair_lb_at_tau_pair_med` | Median $c_{\mathrm{pair}}$ |

## Three Concentration Regimes

| Regime | Functions | C2? | pair? | $c_{\mathrm{pair}}$ | Interpretation |
|--------|-----------|-----|-------|---------------------|----------------|
| **Fast concentration** | f1–f4, f6, f9 | ✅ | ✅ | $\sim 3 \times 10^{-5}$ | Unimodal: bound applies |
| **Slow concentration** | f11–f15, f18, f20 | ✅ | ✅ | $\sim 2 \times 10^{-4}$ | Hybrid: long plateau then collapse |
| **Dispersion** | f5, f7, f10, f17 | ❌ | ❌ | — | Multimodal: population disperses |
| **Exploration failure** | f22, f24 | ✅* | ✅* | $\sim 4 \times 10^{-5}$ | *Only 4–6% find basin |

## Figures

### Convergence Plots (`convergence_D{dim}_f{fid}.png`)

Shows best-so-far fitness with vertical markers:
- **Green dashed**: $\tau_{\mathrm{deep}}$ — witness enters basin
- **Orange dashed**: $\tau_{C2}$ — first concentration
- **Red dashed**: $\tau_{\mathrm{pair}}$ — pair-ready (bound non-vacuous)

### Comparison Plots (`comparison_*.png`)

| File | Contents |
|------|----------|
| `comparison_three_regimes_D{dim}.png` | f1 vs f5 vs f11 — key contrast |
| `comparison_comprehensive_D{dim}.png` | 2×3 grid with 6 representative functions |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `margin` | 60 | $\varepsilon_{\mathrm{in}} = \max(\mathrm{margin}, \mathrm{median} + \mathrm{margin})$ |
| `deep_ratio` | 0.25 | $\varepsilon_{\mathrm{deep}} = 0.25 \cdot \varepsilon_{\mathrm{in}}$ |
| `F_minus` | 0.1 | Lower $F$ bound |
| `Delta_F` | 0.8 | $F$ interval width |
| `L_n_dirs` | 64 | Directions for $L$ estimation |
| `L_quantile` | 0.90 | Quantile for robust $L$ estimate |

## Computing the Hazard Bound

The per-generation hazard bound (Theorem 5) is:

$$\tilde{a}_t = \frac{c_{\mathrm{pair}}}{H} \cdot (g^- \Delta_F) \cdot (q^- \eta_r)$$

With typical values $H=5$, $g^- \Delta_F \approx 0.08$, $q^- \eta_r \approx 0.1$:

$$\tilde{a} \approx c_{\mathrm{pair}} \times 1.6 \times 10^{-3}$$

| Function class | $c_{\mathrm{pair}}$ | $\tilde{a}$ |
|----------------|---------------------|-------------|
| Unimodal (f1–f4) | $3 \times 10^{-5}$ | $5 \times 10^{-8}$ |
| Hybrid (f11–f20) | $2 \times 10^{-4}$ | $3 \times 10^{-7}$ |

## Key Results Table (D=10)

| Function | $\tau_{\mathrm{deep}}$ | $\tau_{C2}$ | $\tau_{\mathrm{pair}}$ | $c_{\mathrm{pair}}$ | $\tilde{a}$ |
|----------|------------------------|-------------|------------------------|---------------------|-------------|
| f1 | 183 | 228 | 249 | $4.2 \times 10^{-5}$ | $6.7 \times 10^{-8}$ |
| f3 | 98 | 150 | 153 | $3.2 \times 10^{-5}$ | $5.2 \times 10^{-8}$ |
| f5 | 220 | — | — | — | — |
| f11 | 82 | 557 | 588 | $1.8 \times 10^{-4}$ | $2.8 \times 10^{-7}$ |
| f22 | 102 | 172 | 174 | $3.2 \times 10^{-5}$ | $5.1 \times 10^{-8}$ |