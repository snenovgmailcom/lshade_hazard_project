#!/usr/bin/env python3
"""
analyze_witness_regime_conc.py  (paper-faithful version)

Goal
----
At τ_deep (first hit into A_{ε_deep} = A_{ε_in/4}), summarize:
  - Mass in zones: A_deep ⊂ A_in ⊂ A_out ⊂ [l,u]^d
  - Geometric concentration proxy aligned with Assumption (concentration):
        r_conc = r_safe / (2 * (F^- + Delta_F))

  Define (paper-faithful):
    - Choose a witness target index i = (one of) individuals in A_deep at τ_deep
    - Choose pbest index b ≠ i (here: best in population excluding i)
    - S^(1) = population indices excluding {i,b}   (size s1 = N - 2)
    - S^(2) = (population indices excluding {i,b}) ∪ archive points
              (theoretical size s2 = N + |A| - 3, per Tanabe–Fukunaga 2014)

    - C1 ⊂ S^(1) are donors within r_conc of x_b
    - C2 ⊂ S^(2) are donors within r_conc of x_b

  Estimate:
    beta1_hat = |C1| / s1
    beta2_hat = |C2| / s2
    c_pair lower bound (TF-exact combinatorial term):
        c_pair_lb_hat = (|C1|/s1) * ((|C2|-1)/(s2-1))   if |C2|>1 and s2>1 else 0

Important differences vs the earlier draft
-----------------------------------------
1) We DO NOT set i=b. We pick i from the deep-hit set at τ_deep, then pick b as pbest excluding i.
2) We DO NOT do extra “-1 / -2” conservative subtractions. (Those make the empirical test stricter
   than the paper and bias results downward.)
3) C1/C2 are computed exactly on S^(1), S^(2) as above.

Data format expectation
-----------------------
Pickle file: base/D{dim}/f{func_id}/f{func_id}.pkl
Contains dict with key "cec2017_f{func_id}" mapping to a list of runs.
Each run is a dict with at least:
  run["history"]["fitness"][t]    -> list/array length N of fitnesses
  run["history"]["positions"][t]  -> array (N,d)

Archive (optional) history keys (any of):
  "archive_positions", "A_positions", "archive_pos", "archive"
and/or size keys:
  "archive_size", "A_size", "archive_len"

Usage examples
--------------
All functions, D=10:
  python analyze_witness_regime_conc.py --dim 10 --func all --base experiments \
      --r_safe 1.0 --F_minus 0.1 --Delta_F 0.8 --outdir out

Single function with margin eps_in and plots:
  python analyze_witness_regime_conc.py --dim 10 --func f1 --base experiments \
      --eps_in_mode margin --margin 5.0 \
      --r_safe 1.0 --F_minus 0.1 --Delta_F 0.8 \
      --outdir out --plots
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib with non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Progress utilities
# ============================================================================

def print_progress(msg: str, end: str = "\n", flush: bool = True) -> None:
    """Print progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", end=end, flush=flush)


def print_run_progress(run_idx: int, n_runs: int, func_id: int, status: str = "") -> None:
    """Print progress for run processing."""
    pct = 100 * (run_idx + 1) / n_runs
    msg = f"  f{func_id}: run {run_idx+1}/{n_runs} ({pct:5.1f}%)"
    if status:
        msg += f" - {status}"
    print(f"\r{msg}", end="", flush=True)


# ============================================================================
# Utilities: loading & f*
# ============================================================================

def get_f_star(func_id: int) -> float:
    """CEC2017: f* = 100 * func_id."""
    return 100.0 * func_id


def load_runs(dim: int, func_id: int, base: Path) -> Optional[List[Dict[str, Any]]]:
    """Load runs from pickle at base/D{dim}/f{func_id}/f{func_id}.pkl"""
    pkl_path = base / f"D{dim}" / f"f{func_id}" / f"f{func_id}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    key = f"cec2017_f{func_id}"
    return data.get(key, None)


# ============================================================================
# eps_in & τ detection
# ============================================================================

def compute_eps_in_rho(runs: List[Dict[str, Any]], f_star: float, rho: float) -> float:
    """ε_in = max{(1+ρ)*median(final_errors), 1}"""
    final_errors = []
    for run in runs:
        curve = run.get("curve", [])
        if curve is not None and len(curve) > 0:
            final_errors.append(float(curve[-1]) - f_star)
    if not final_errors:
        return 1.0
    m = float(np.median(np.asarray(final_errors, dtype=float)))
    return max((1.0 + rho) * m, 1.0)


def compute_eps_in_margin(runs: List[Dict[str, Any]], f_star: float, margin: float) -> float:
    """ε_in = max{median(final_errors) + margin, margin/10}"""
    final_errors = []
    for run in runs:
        curve = run.get("curve", [])
        if curve is not None and len(curve) > 0:
            final_errors.append(float(curve[-1]) - f_star)
    if not final_errors:
        return max(margin / 10.0, 1e-12)
    m = float(np.median(np.asarray(final_errors, dtype=float)))
    return max(m + margin, margin / 10.0)


def find_tau_first_hit(fitness_hist: List[Any], f_star: float, eps: float) -> Optional[int]:
    """First index t where min(f_t) ≤ f* + eps."""
    thr = f_star + eps
    for t, f_t in enumerate(fitness_hist):
        arr = np.asarray(f_t, dtype=float)
        if arr.size > 0 and float(np.min(arr)) <= thr:
            return t
    return None


# ============================================================================
# eps_out via pooled gaps
# ============================================================================

def compute_eps_out_pooled(
    runs: List[Dict[str, Any]], f_star: float, eps_in: float, q: float = 80.0
) -> float:
    """
    eps_out = eps_in + Delta_q pooled across runs at each run's tau_in.
    Delta_q is the q-th percentile of outsider gaps: f - (f* + eps_in).
    """
    boundary = f_star + eps_in
    pooled_gaps = []
    for run in runs:
        hist = run.get("history", {})
        fitness_hist = hist.get("fitness", [])
        if not fitness_hist:
            continue
        tau_in = find_tau_first_hit(fitness_hist, f_star, eps_in)
        if tau_in is None:
            continue
        f_at = np.asarray(fitness_hist[tau_in], dtype=float)
        gaps = f_at[f_at > boundary] - boundary
        if gaps.size:
            pooled_gaps.extend(gaps.tolist())
    if not pooled_gaps:
        return eps_in
    Delta_q = float(np.percentile(np.asarray(pooled_gaps, dtype=float), q))
    return eps_in + max(0.0, Delta_q)


# ============================================================================
# Archive extraction (robust to different logging formats)
# ============================================================================

def extract_archive_positions(hist: Dict[str, Any], t: int) -> Optional[np.ndarray]:
    """
    Extract archive positions at time t from history.
    Tries common key names used in various L-SHADE implementations.
    Returns array shape (M, d) or None.
    """
    for key in ("archive_positions", "A_positions", "archive_pos", "archive"):
        seq = hist.get(key)
        if isinstance(seq, list) and t < len(seq):
            try:
                A_t = np.asarray(seq[t], dtype=float)
                if A_t.ndim == 2 and A_t.shape[0] > 0:
                    return A_t
            except (ValueError, TypeError):
                pass
    return None


def extract_archive_size(hist: Dict[str, Any], t: int, X_arch: Optional[np.ndarray]) -> int:
    """
    Extract archive size at time t.
    Prefer actual positions length if available.
    """
    if isinstance(X_arch, np.ndarray):
        return int(X_arch.shape[0])

    for key in ("archive_size", "A_size", "archive_len"):
        seq = hist.get(key)
        if isinstance(seq, list) and t < len(seq):
            try:
                return int(seq[t])
            except (TypeError, ValueError):
                pass
        if isinstance(seq, (int, float)):
            return int(seq)

    return 0


# ============================================================================
# Witness index selection at τ_deep (paper-faithful)
# ============================================================================

def choose_i_b_at_tau_deep(
    f_t: np.ndarray, f_star: float, eps_deep: float
) -> Optional[Tuple[int, int]]:
    """
    Choose witness target i and pbest index b at τ_deep.

    - i: one index in A_deep at τ_deep. We pick the BEST among deep hitters.
    - b: pbest index, chosen as the best in the population excluding i (b != i).

    Returns (i, b) or None.
    """
    thr_deep = f_star + eps_deep
    deep_idx = np.where(f_t <= thr_deep)[0]
    if deep_idx.size == 0:
        return None

    # i = best among deep hitters
    i = int(deep_idx[np.argmin(f_t[deep_idx])])

    # b = best among population excluding i
    mask = np.ones_like(f_t, dtype=bool)
    mask[i] = False
    if not np.any(mask):
        return None
    b = int(np.argmin(np.where(mask, f_t, np.inf)))
    if b == i:
        return None
    return i, b


# ============================================================================
# Concentration metrics (Tanabe-Fukunaga aligned, paper-faithful)
# ============================================================================

def concentration_metrics_tanabe_fukunaga(
    X_pop: np.ndarray,
    f_pop: np.ndarray,
    r_conc: float,
    i: int,
    b: int,
    arch_M: int = 0,
    X_arch: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Paper-faithful concentration proxy for current-to-pbest/1.

    S^(1) = population indices excluding {i,b}; s1 = N - 2
    S^(2) = (population indices excluding {i,b}) ∪ archive points; theoretical s2 = N + |A| - 3

    C1 = {x in S^(1): ||x - x_b|| <= r_conc}
    C2 = {x in S^(2): ||x - x_b|| <= r_conc}

    beta1_hat = |C1|/s1
    beta2_hat = |C2|/s2

    TF-exact combinatorial lower bound used in your Proposition:
      c_pair_lb_hat = (|C1|/s1) * ((|C2|-1)/(s2-1))   when |C2|>1 and s2>1 else 0
    """
    N = int(f_pop.size)
    nan_result = {
        "s1": float("nan"), "s2": float("nan"),
        "c1": float("nan"), "c2": float("nan"),
        "beta1_hat": float("nan"), "beta2_hat": float("nan"),
        "c_pair_lb_hat": float("nan"),
    }

    if X_pop.ndim != 2 or X_pop.shape[0] != N or N < 4:
        return nan_result
    if not (0 <= i < N and 0 <= b < N) or i == b:
        return nan_result

    x_b = X_pop[b]

    # S^(1): population excluding {i,b}
    mask_S1 = np.ones(N, dtype=bool)
    mask_S1[i] = False
    mask_S1[b] = False
    X_S1 = X_pop[mask_S1]  # shape (N-2, d)
    s1 = int(X_S1.shape[0])

    # S^(2) theoretical pool size per Tanabe–Fukunaga:
    # exclude i, pbest=b, and r1 from union(pop ∪ arch)
    # => s2 = N + |A| - 3
    s2 = max(N + int(arch_M) - 3, 0)

    if s1 <= 0 or s2 <= 1:
        return nan_result

    # C1 count
    d_S1 = np.linalg.norm(X_S1 - x_b, axis=1)
    c1 = int(np.sum(d_S1 <= r_conc))
    c1 = min(c1, s1)

    # C2 count: (population excluding {i,b}) plus archive points
    c2 = c1
    if X_arch is not None and X_arch.ndim == 2 and X_arch.shape[0] > 0:
        d_arch = np.linalg.norm(X_arch - x_b, axis=1)
        c2 += int(np.sum(d_arch <= r_conc))

    c2 = min(c2, s2)

    beta1_hat = c1 / s1
    beta2_hat = c2 / s2

    if s2 > 1 and c2 > 1:
        c_pair_lb_hat = (c1 / s1) * ((c2 - 1) / (s2 - 1))
    else:
        c_pair_lb_hat = 0.0

    return {
        "s1": float(s1),
        "s2": float(s2),
        "c1": float(c1),
        "c2": float(c2),
        "beta1_hat": float(beta1_hat),
        "beta2_hat": float(beta2_hat),
        "c_pair_lb_hat": float(c_pair_lb_hat),
    }


# ============================================================================
# Diameter computation (no SciPy dependency)
# ============================================================================

def diam_pairwise_numpy(X: np.ndarray) -> float:
    """Max pairwise distance using Gram trick. O(n^2) memory."""
    n = X.shape[0]
    if n <= 1:
        return 0.0
    G = X @ X.T
    sq = np.diag(G)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    D2[D2 < 0] = 0.0
    return float(np.sqrt(np.max(D2)))


def diam_to_best(X: np.ndarray, x_b: np.ndarray) -> float:
    """Max distance from any point to x_b."""
    if X.shape[0] == 0:
        return 0.0
    return float(np.max(np.linalg.norm(X - x_b, axis=1)))


# ============================================================================
# Per-run analysis at τ_deep
# ============================================================================

def analyze_run_at_tau_deep(
    run: Dict[str, Any],
    f_star: float,
    eps_in: float,
    eps_out: float,
    r_conc: float,
) -> Optional[Dict[str, Any]]:
    """
    Compute zone counts and concentration metrics at τ_deep.
    τ_deep is first hit into A_{eps_in/4}.
    """
    hist = run.get("history", {})
    fitness_hist = hist.get("fitness", [])
    positions_hist = hist.get("positions", [])

    if not fitness_hist or not positions_hist:
        return None

    eps_deep = eps_in / 4.0
    tau_deep = find_tau_first_hit(fitness_hist, f_star, eps_deep)
    if tau_deep is None:
        return None

    # Population snapshot at τ_deep
    f_t = np.asarray(fitness_hist[tau_deep], dtype=float)
    X_t = np.asarray(positions_hist[tau_deep], dtype=float)

    if f_t.size == 0 or X_t.ndim != 2 or X_t.shape[0] != f_t.size:
        return None

    N = int(f_t.size)
    if N < 4:
        return None

    # Choose witness indices (i, b) paper-faithfully
    ib = choose_i_b_at_tau_deep(f_t, f_star, eps_deep)
    if ib is None:
        return None
    i, b = ib
    x_b = X_t[b]

    # ---- Zone counts ----
    thr_deep = f_star + eps_deep
    thr_in = f_star + eps_in
    thr_out = f_star + eps_out

    n_deep = int(np.sum(f_t <= thr_deep))
    n_in = int(np.sum(f_t <= thr_in))
    n_out = int(np.sum(f_t <= thr_out))

    n_in_shell = n_in - n_deep
    n_out_shell = n_out - n_in
    n_outside = N - n_out

    # ---- Diameters for A_out members ----
    in_A_out = f_t <= thr_out
    X_out = X_t[in_A_out]

    diam_max = float("nan")
    diam_best = float("nan")
    if X_out.shape[0] >= 1:
        diam_max = diam_pairwise_numpy(X_out)
        diam_best = diam_to_best(X_out, x_b)

    # ---- Archive extraction ----
    X_arch = extract_archive_positions(hist, tau_deep)
    arch_M = extract_archive_size(hist, tau_deep, X_arch)

    # ---- Concentration metrics (Tanabe-Fukunaga aligned) ----
    conc = concentration_metrics_tanabe_fukunaga(
        X_pop=X_t,
        f_pop=f_t,
        r_conc=r_conc,
        i=i,
        b=b,
        arch_M=arch_M,
        X_arch=X_arch,
    )

    return {
        "tau_deep": int(tau_deep),
        "N": int(N),
        "i": int(i),
        "b": int(b),
        "eps_in": float(eps_in),
        "eps_out": float(eps_out),
        "n_deep": int(n_deep),
        "n_in_shell": int(n_in_shell),
        "n_out_shell": int(n_out_shell),
        "n_outside": int(n_outside),
        "diam_max": float(diam_max),
        "diam_best": float(diam_best),
        "r_conc": float(r_conc),
        "arch_M": int(arch_M),
        **conc,
    }


# ============================================================================
# Function-level processing
# ============================================================================

def process_function(
    dim: int,
    func_id: int,
    base: Path,
    eps_in_mode: str,
    rho: float,
    margin: float,
    out_q: float,
    r_safe: float,
    F_minus: float,
    Delta_F: float,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Process one function, return median summary."""
    if verbose:
        print_progress(f"Loading f{func_id}...", end=" ")

    runs = load_runs(dim, func_id, base)
    if not runs:
        if verbose:
            print("not found")
        return None

    if verbose:
        print(f"loaded {len(runs)} runs")

    f_star = get_f_star(func_id)

    # eps_in
    if eps_in_mode == "rho":
        eps_in = compute_eps_in_rho(runs, f_star, rho)
    elif eps_in_mode == "margin":
        eps_in = compute_eps_in_margin(runs, f_star, margin)
    else:
        raise ValueError("eps_in_mode must be 'rho' or 'margin'")

    # eps_out pooled across runs at tau_in
    eps_out = compute_eps_out_pooled(runs, f_star, eps_in, q=out_q)

    # theorem-inspired radius: r_conc = r_safe / (2*(F^- + Delta_F))
    denom = 2.0 * (F_minus + Delta_F)
    r_conc = (r_safe / denom) if denom > 0 else float("inf")

    if verbose:
        print_progress(f"  eps_in={eps_in:.4f}, eps_out={eps_out:.4f}, r_conc={r_conc:.4f}")

    results = []
    n_runs = len(runs)

    for k, run in enumerate(runs):
        if verbose and (k % 10 == 0 or k == n_runs - 1):
            print_run_progress(k, n_runs, func_id)

        res = analyze_run_at_tau_deep(run, f_star, eps_in, eps_out, r_conc)
        if res is not None:
            results.append(res)

    if verbose:
        print()  # newline after progress
        print_progress(f"  Valid runs: {len(results)}/{n_runs}")

    if not results:
        return None

    # ---- Compute medians ----
    def med(key: str) -> float:
        vals = [r.get(key, np.nan) for r in results]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    def med_int(key: str) -> int:
        vals = [r.get(key, np.nan) for r in results]
        vals = [v for v in vals if np.isfinite(v)]
        return int(np.median(vals)) if vals else 0

    return {
        "Func": f"f{func_id}",
        "n_runs_ok": len(results),
        "n_runs": n_runs,
        "N_med": med_int("N"),
        "tau_deep_med": med_int("tau_deep"),
        "eps_in": float(eps_in),
        "eps_out": float(eps_out),
        "#A_deep": med_int("n_deep"),
        "#A_in\\A_deep": med_int("n_in_shell"),
        "#A_out\\A_in": med_int("n_out_shell"),
        "#outside": med_int("n_outside"),
        "diam(A_out)": med("diam_max"),
        "diam_best": med("diam_best"),
        "r_conc": float(r_conc),
        "s1": med("s1"),
        "s2": med("s2"),
        "c1": med("c1"),
        "c2": med("c2"),
        "beta1": med("beta1_hat"),
        "beta2": med("beta2_hat"),
        "c_pair_lb": med("c_pair_lb_hat"),
        "arch_M": med_int("arch_M"),
    }


# ============================================================================
# Plotting functions
# ============================================================================

def plot_zone_mass(df: pd.DataFrame, outpath: Path, title: str = "") -> None:
    """
    Figure A: Stacked horizontal bar chart showing population mass in zones
    at τ_deep (normalized by N_med).
    """
    funcs = df["Func"].tolist()
    N = df["N_med"].astype(float).values
    N = np.where(N > 0, N, 1.0)

    deep = df["#A_deep"].astype(float).values / N
    in_shell = df["#A_in\\A_deep"].astype(float).values / N
    out_shell = df["#A_out\\A_in"].astype(float).values / N
    outside = df["#outside"].astype(float).values / N

    y = np.arange(len(funcs))[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(funcs) + 1)))
    left = np.zeros_like(deep)

    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#7f7f7f"]

    ax.barh(y, deep, left=left, color=colors[0], label=r"$A_{\mathrm{deep}}$")
    left += deep
    ax.barh(y, in_shell, left=left, color=colors[1], label=r"$A_{\varepsilon}\setminus A_{\mathrm{deep}}$")
    left += in_shell
    ax.barh(y, out_shell, left=left, color=colors[2], label=r"$A_{\mathrm{out}}\setminus A_{\varepsilon}$")
    left += out_shell
    ax.barh(y, outside, left=left, color=colors[3], label=r"outside $A_{\mathrm{out}}$")

    ax.set_yticks(y)
    ax.set_yticklabels(funcs)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(r"fraction of population at $t=\tau_{\mathrm{deep}}$")
    ax.set_title(title if title else r"Population mass in zones at $\tau_{\mathrm{deep}}$")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_witness_strength(df: pd.DataFrame, outpath: Path, title: str = "") -> None:
    """
    Figure B: Scatter plot of c_pair_lb per function.
    Marker size proportional to beta1.
    """
    funcs = df["Func"].tolist()
    y = np.arange(len(funcs))[::-1]

    cpair = df["c_pair_lb"].astype(float).values
    beta1 = df["beta1"].astype(float).values

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(funcs) + 1)))

    sizes = 50 + 400 * np.clip(np.nan_to_num(beta1, nan=0.0), 0, 1)
    scatter = ax.scatter(
        cpair, y, s=sizes, c=cpair, cmap="viridis",
        edgecolors="black", linewidths=0.5, vmin=0
    )

    ax.set_yticks(y)
    ax.set_yticklabels(funcs)
    ax.set_xlabel(r"lower bound $\widehat{c}_{\mathrm{pair}}$ at $t=\tau_{\mathrm{deep}}$")
    ax.set_title(title if title else r"Witness regime strength at $\tau_{\mathrm{deep}}$")
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(r"$\widehat{c}_{\mathrm{pair}}$")

    ax.text(
        0.98, 0.02, r"marker size $\propto \hat\beta_1$",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        style="italic", color="gray"
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_beta_scatter(df: pd.DataFrame, outpath: Path, title: str = "") -> None:
    """
    Figure C: Scatter plot of beta1 vs beta2 per function.
    """
    funcs = df["Func"].tolist()
    beta1 = df["beta1"].astype(float).values
    beta2 = df["beta2"].astype(float).values
    cpair = df["c_pair_lb"].astype(float).values

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        beta1, beta2, s=100, c=cpair, cmap="viridis",
        edgecolors="black", linewidths=0.5, vmin=0
    )

    for i, func in enumerate(funcs):
        ax.annotate(func, (beta1[i], beta2[i]), fontsize=8, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(r"$\hat\beta_1 = |C_1| / s_1$")
    ax.set_ylabel(r"$\hat\beta_2 = |C_2| / s_2$")
    ax.set_title(title if title else r"Concentration fractions at $\tau_{\mathrm{deep}}$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label=r"$\beta_1 = \beta_2$")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(r"$\widehat{c}_{\mathrm{pair}}$")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Witness regime concentration analysis (Tanabe-Fukunaga aligned, paper-faithful)"
    )
    ap.add_argument("--dim", type=int, required=True, help="Dimension (e.g., 10, 30, 50)")
    ap.add_argument("--func", type=str, required=True, help="f1..f30 or 'all'")
    ap.add_argument("--base", type=str, default="experiments", help="Base directory for data")

    ap.add_argument("--eps_in_mode", type=str, default="rho", choices=["rho", "margin"],
                    help="Method for computing eps_in")
    ap.add_argument("--rho", type=float, default=0.5, help="Multiplier for rho mode")
    ap.add_argument("--margin", type=float, default=5.0, help="Additive margin for margin mode")

    ap.add_argument("--out_q", type=float, default=80.0,
                    help="Percentile for outsider gaps at tau_in (pooled)")

    ap.add_argument("--r_safe", type=float, required=True,
                    help="r_safe from theorem (e.g., sqrt(eps/(2L)))")
    ap.add_argument("--F_minus", type=float, required=True,
                    help="F^- lower bound (e.g., 0.1)")
    ap.add_argument("--Delta_F", type=float, required=True,
                    help="Delta_F success interval width (e.g., 0.8)")

    ap.add_argument("--outdir", type=str, default=None, help="Output directory for CSV/plots")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    ap.add_argument("--plots", action="store_true", help="Save summary plots")

    args = ap.parse_args()

    base = Path(args.base)
    verbose = not args.quiet

    # Parse function list
    if args.func.lower() == "all":
        func_ids = list(range(1, 31))
    else:
        func_ids = []
        for part in args.func.replace(" ", "").split(","):
            func_ids.append(int(part.lower().replace("f", "")))

    # r_conc preview
    denom = 2.0 * (args.F_minus + args.Delta_F)
    r_conc_preview = (args.r_safe / denom) if denom > 0 else float("inf")

    if verbose:
        print("=" * 70)
        print("Witness Regime Concentration Analysis (paper-faithful)")
        print("=" * 70)
        print(f"Dimension:    D = {args.dim}")
        print(f"Functions:    {func_ids}")
        print(f"Data base:    {base}")
        print(f"eps_in mode:  {args.eps_in_mode} (rho={args.rho}, margin={args.margin})")
        print(f"eps_out q:    {args.out_q}th percentile")
        print(f"r_safe:       {args.r_safe}")
        print(f"F^-:          {args.F_minus}")
        print(f"Delta_F:      {args.Delta_F}")
        print(f"r_conc:       {r_conc_preview:.4g} = r_safe / (2*(F^- + Delta_F))")
        print(f"Plots:        {'enabled' if args.plots else 'disabled'}")
        print("=" * 70)
        print()

    rows = []
    t_start = time.time()

    for idx, fid in enumerate(func_ids):
        if verbose:
            print(f"\n[{idx+1}/{len(func_ids)}] Processing f{fid}")
            print("-" * 40)

        row = process_function(
            dim=args.dim,
            func_id=fid,
            base=base,
            eps_in_mode=args.eps_in_mode,
            rho=args.rho,
            margin=args.margin,
            out_q=args.out_q,
            r_safe=args.r_safe,
            F_minus=args.F_minus,
            Delta_F=args.Delta_F,
            verbose=verbose,
        )
        if row is not None:
            rows.append(row)

    elapsed = time.time() - t_start

    if verbose:
        print()
        print("=" * 70)
        print(f"Completed in {elapsed:.1f}s")
        print("=" * 70)

    if not rows:
        print("No data found / no valid runs.")
        return

    df = pd.DataFrame(rows)
    preferred = [
        "Func", "n_runs_ok", "n_runs", "N_med", "tau_deep_med",
        "eps_in", "eps_out",
        "#A_deep", "#A_in\\A_deep", "#A_out\\A_in", "#outside",
        "diam(A_out)", "diam_best",
        "r_conc", "s1", "s2", "c1", "c2",
        "beta1", "beta2", "c_pair_lb",
        "arch_M",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    print()
    print("Summary Table:")
    print("-" * 70)
    print(df.to_string(index=False))
    print()

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        func_str = args.func.replace(",", "_")

        csv_path = outdir / f"witness_conc_D{args.dim}_{func_str}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        if args.plots:
            print()
            print("Generating plots...")

            plot_zone_mass(
                df,
                outdir / f"zones_D{args.dim}_{func_str}.png",
                title=rf"Population zones at $\tau_{{\mathrm{{deep}}}}$ (D={args.dim})"
            )

            plot_witness_strength(
                df,
                outdir / f"cpair_D{args.dim}_{func_str}.png",
                title=rf"Witness strength (D={args.dim}, $r_{{\mathrm{{conc}}}}$={r_conc_preview:.3g})"
            )

            plot_beta_scatter(
                df,
                outdir / f"beta_scatter_D{args.dim}_{func_str}.png",
                title=rf"Concentration fractions (D={args.dim})"
            )


if __name__ == "__main__":
    main()
