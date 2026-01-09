#!/usr/bin/env python3
"""
extended_analysis.py

Extended per-function diagnostics that rely on full logging (positions, fitness,
pop_size, successful_F, etc.). This script is intended to support the empirical
tables in the paper:

- N_tau: population size at hitting time (useful because L-SHADE shrinks N_t).
- Success-F interval: empirical support/quantiles of successful F samples.
- Concentration diagnostics at hitting time: n_clust, beta1, diameter for various r_conc.

Notes on definitions (aligned with the paper):
- tau_eps = inf{ t : f_best(t) <= f* + eps }.
- n_clust(t) = #{ i : ||x_i^{(t)} - x_best^{(t)}|| <= r_conc }.

This script focuses on *at-hitting* summaries (one time per run, per eps).
For per-generation witness-regime proxy frequencies, see combined_analysis.py
or build on analysis/cluster_analysis.py and analysis/witness_frequency.py.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from km_analysis import compute_hitting_times
from cluster_analysis import cluster_stats_at_hitting


def f_star_from_func(func: str) -> float:
    func = func.strip()
    if func.startswith("cec2017_"):
        func = func.replace("cec2017_", "")
    if not (func.startswith("f") and func[1:].isdigit()):
        raise ValueError(f"Cannot parse f* from func={func}")
    return float(100 * int(func[1:]))


def compute_N_at_hitting(run: Dict, tau: int) -> Optional[int]:
    hist = run.get("history", {})
    pop_hist = hist.get("pop_size", None)
    if pop_hist is None:
        return None
    if tau < 0 or tau >= len(pop_hist):
        return None
    try:
        return int(pop_hist[tau])
    except Exception:
        return None


def compute_success_F_interval(run: Dict, tau: int, q_lo: float = 0.05, q_hi: float = 0.95) -> Tuple[float, float]:
    """
    Empirical "success-F interval" from logged successful_F values up to time tau.
    We take (q_lo, q_hi) quantiles of the pooled successful F samples.

    Returns (F_minus, F_plus). If no samples, returns (nan,nan).
    """
    hist = run.get("history", {})
    succ = hist.get("successful_F", None)
    if succ is None:
        return (np.nan, np.nan)
    tau = int(tau)
    tau = min(tau, len(succ) - 1)
    vals = []
    for t in range(0, tau + 1):
        ft = succ[t]
        if ft is None:
            continue
        if isinstance(ft, (list, tuple, np.ndarray)):
            vals.extend([float(x) for x in ft if np.isfinite(float(x))])
    if len(vals) == 0:
        return (np.nan, np.nan)
    v = np.asarray(vals, dtype=float)
    return (float(np.quantile(v, q_lo)), float(np.quantile(v, q_hi)))


def analyze_function_extended(
    runs: List[Dict],
    f_star: float,
    eps_values: List[float],
    r_conc_values: List[float],
) -> pd.DataFrame:
    curves = [np.asarray(r["curve"], dtype=float) for r in runs]
    B = int(max(len(c) for c in curves) - 1)

    rows = []
    for eps in eps_values:
        taus = compute_hitting_times(curves, f_star=f_star, eps=eps)
        finite = taus[np.isfinite(taus)]
        tau_median = float(np.median(finite)) if finite.size else np.nan

        # N_tau
        N_list = []
        F_minus_list = []
        F_plus_list = []
        for run, tau in zip(runs, taus):
            if not np.isfinite(tau):
                continue
            tau = int(tau)
            N = compute_N_at_hitting(run, tau)
            if N is not None:
                N_list.append(N)
            fmin, fplus = compute_success_F_interval(run, tau)
            if np.isfinite(fmin):
                F_minus_list.append(fmin)
            if np.isfinite(fplus):
                F_plus_list.append(fplus)

        N_med = float(np.median(N_list)) if N_list else np.nan
        N_min = float(np.min(N_list)) if N_list else np.nan
        N_max = float(np.max(N_list)) if N_list else np.nan

        F_minus_med = float(np.median(F_minus_list)) if F_minus_list else np.nan
        F_plus_med = float(np.median(F_plus_list)) if F_plus_list else np.nan

        base = {
            "eps": float(eps),
            "n_hits": int(np.isfinite(taus).sum()),
            "tau_median": tau_median,
            "N_tau_median": N_med,
            "N_tau_min": N_min,
            "N_tau_max": N_max,
            "F_minus_med": F_minus_med,
            "F_plus_med": F_plus_med,
        }

        # cluster stats at hitting for each r_conc
        for r_conc in r_conc_values:
            stats = []
            for run, tau in zip(runs, taus):
                if not np.isfinite(tau):
                    continue
                st = cluster_stats_at_hitting(run, tau=int(tau), r_conc=float(r_conc))
                if st is not None:
                    stats.append(st)
            if stats:
                df_st = pd.DataFrame(stats)
                base[f"n_clust_r{r_conc:g}"] = float(df_st["n_clust"].median())
                base[f"beta1_r{r_conc:g}"] = float(df_st["beta1"].median())
                base[f"diameter_r{r_conc:g}"] = float(df_st["diameter"].median())
            else:
                base[f"n_clust_r{r_conc:g}"] = np.nan
                base[f"beta1_r{r_conc:g}"] = np.nan
                base[f"diameter_r{r_conc:g}"] = np.nan

        rows.append(base)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extended analysis with full logging")
    ap.add_argument("--pkl", required=True, help="Path to raw_results_lshade.pkl")
    ap.add_argument("--func", required=True, help="Function name (e.g., f1, f11) or cec2017_f1")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="results/tables")
    ap.add_argument("--eps", type=str, default="0.01,1,10,100,400")
    ap.add_argument("--r_conc", type=str, default="1,10", help="comma-separated r_conc values")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    func_key = args.func if args.func.startswith("cec2017_") else f"cec2017_{args.func}"
    if func_key not in data:
        raise KeyError(f"Function {func_key} not found in PKL keys.")

    runs = data[func_key]
    f_star = f_star_from_func(args.func)

    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
    r_conc_values = [float(x) for x in args.r_conc.split(",") if x.strip()]

    outdir = Path(args.outdir) / f"D{args.dim}" / args.func.replace("cec2017_", "")
    outdir.mkdir(parents=True, exist_ok=True)

    df = analyze_function_extended(runs, f_star=f_star, eps_values=eps_values, r_conc_values=r_conc_values)
    out_csv = outdir / "extended_results.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
