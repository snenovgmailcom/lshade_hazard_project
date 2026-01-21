#!/usr/bin/env python3
"""
generate_tables.py

Batch driver to generate per-function CSV tables from a single PKL file.

This script is intended to support the paper tables and to provide a clean
single-entry workflow:

- Loads PKL once.
- For each function and each tolerance eps:
  * KM statistics (tau_first/median/max, pooled hazard p_hat, envelope diagnostics)
  * Memory-condition pooled rates (L2, L3, gamma)
  * Optional gap analysis against a Morse-style bound
    (requires a concentration proxy c_pair; we approximate it via beta1*beta2,
     where beta1 is measured from a ball around the best at hitting time.)

Outputs:
- results/tables/D{dim}/summary_all_functions.csv
- results/tables/D{dim}/{fX}/full_results.csv (per function)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from km_analysis import compute_hitting_times, compute_km_statistics
from witness_frequency import compute_witness_indicators, estimate_gamma, GAMMA_THRESHOLDS
from cluster_analysis import summarize_cluster_at_hitting
from gap_analysis import compute_gap_analysis


def list_function_keys(data: Dict) -> List[str]:
    keys = [k for k in data.keys() if "f" in k]
    import re
    def fnum(k: str) -> int:
        m = re.search(r"f(\d+)", k)
        return int(m.group(1)) if m else 999
    return sorted(keys, key=fnum)


def f_star_from_key(key: str) -> float:
    import re
    m = re.search(r"f(\d+)", key)
    if not m:
        raise ValueError(f"Cannot parse function id from key={key}")
    return float(100 * int(m.group(1)))


def pop_size_median_at_hitting(runs: List[Dict], taus: np.ndarray) -> float:
    Ns = []
    for run, tau in zip(runs, taus):
        if not np.isfinite(tau):
            continue
        tau = int(tau)
        hist = run.get("history", {})
        pop = hist.get("pop_size", None)
        if pop is None or tau >= len(pop):
            continue
        try:
            Ns.append(int(pop[tau]))
        except Exception:
            pass
    return float(np.median(Ns)) if Ns else float("nan")


def analyze_single_function(
    key: str,
    runs: List[Dict],
    dim: int,
    eps_values: Sequence[float],
    r_conc_for_gap: float = 10.0,
    thresholds: Dict[str, float] = GAMMA_THRESHOLDS,
) -> pd.DataFrame:
    f_star = f_star_from_key(key)
    curves = [np.asarray(r["curve"], dtype=float) for r in runs]
    B = int(max(len(c) for c in curves) - 1)

    # precompute witness indicators per run
    indicators = []
    for r in runs:
        hist = r.get("history", {})
        mf = hist.get("memory_f", [])
        mcr = hist.get("memory_cr", [])
        indicators.append(compute_witness_indicators(mf, mcr, thresholds=thresholds) if (mf and mcr)
                          else compute_witness_indicators([], [], thresholds=thresholds))

    rows = []
    for eps in eps_values:
        eps = float(eps)
        taus = compute_hitting_times(curves, f_star=f_star, eps=eps)
        km_stats = compute_km_statistics(taus, B=B, tail_start=None, alpha=0.05)

        gamma_stats = estimate_gamma(
            taus=taus,
            indicators=indicators,
            B=B,
            start_gen=0,
            end_gen=None,
            alive_strict=True,
        )

        # concentration proxy at hitting (beta1 used in c_pair proxy)
        conc = summarize_cluster_at_hitting(runs, taus=taus, r_conc=r_conc_for_gap)
        beta1 = float(conc.get("beta1_median", np.nan))
        N_tau_median = pop_size_median_at_hitting(runs, taus)

        gap = compute_gap_analysis(
            km_stats=km_stats,
            gamma_stats=gamma_stats,
            N_tau_median=N_tau_median,
            d=dim,
            beta1=beta1 if np.isfinite(beta1) else None,
            beta2=None,
            thresholds=thresholds,
            r=None,
        )

        row = {
            "function": key,
            "eps": eps,
            "n_hits": int(km_stats["n_hits"]),
            "tau_first": km_stats["tau_first"],
            "tau_median": km_stats["tau_median"],
            "tau_max": km_stats["tau_max"],
            "tail_start": km_stats["tail_start"],
            "p_hat": km_stats["p_hat"],
            "p_lcb": km_stats["p_lcb"],
            "a_env": km_stats["a_env"],
            "clustering": km_stats["clustering"],
            "L2_rate": gamma_stats["l2_rate"],
            "L3_rate": gamma_stats["l3_rate"],
            "gamma": gamma_stats["gamma"],
            "N_tau_median": N_tau_median,
            "beta1_med_r_gap": beta1,
            "tilde_a": gap["tilde_a"],
            "p_lower": gap["p_lower"],
            "p_lower_gamma": gap["p_lower_gamma"],
            "gap_ratio": gap["gap_ratio"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate summary tables from PKL")
    ap.add_argument("--pkl", required=True, help="Path to raw_results_lshade.pkl")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="results/tables")
    ap.add_argument("--eps", type=str, default="0.01,1,10,100,400")
    ap.add_argument("--funcs", type=str, default="", help="comma-separated list like f1,f5,f11; empty = all")
    ap.add_argument("--r_gap", type=float, default=10.0, help="r_conc used for beta1 proxy in gap analysis")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]
    outroot = Path(args.outdir) / f"D{args.dim}"
    outroot.mkdir(parents=True, exist_ok=True)

    if args.funcs.strip():
        funcs = [f.strip() for f in args.funcs.split(",") if f.strip()]
        keys = [f"cec2017_{f}" if not f.startswith("cec2017_") else f for f in funcs]
    else:
        keys = list_function_keys(data)

    all_rows = []
    for key in keys:
        if key not in data:
            print(f"[WARN] missing key: {key}")
            continue
        df = analyze_single_function(
            key=key,
            runs=data[key],
            dim=args.dim,
            eps_values=eps_values,
            r_conc_for_gap=args.r_gap,
        )
        fshort = key.replace("cec2017_", "")
        fdir = outroot / fshort
        fdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(fdir / "full_results.csv", index=False)
        all_rows.append(df)

    if all_rows:
        summary = pd.concat(all_rows, ignore_index=True)
        summary.to_csv(outroot / "summary_all_functions.csv", index=False)
        print(f"\nSaved: {outroot / 'summary_all_functions.csv'}")


if __name__ == "__main__":
    main()
