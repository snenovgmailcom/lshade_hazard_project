#!/usr/bin/env python3
"""
combined_analysis.py

Single-entry analysis driver: reads the PKL once and produces per-function
CSV + plots, aligned with the math in the paper.

This script unifies the main functionality of:
- km_analysis.py (KM + hazard)
- witness_frequency.py (L2/L3/gamma memory diagnostics)
- cluster_analysis.py (concentration diagnostics)

Typical usage:
  python analysis/combined_analysis.py --pkl raw_results_lshade.pkl --dim 10 --outdir results/tables

Outputs (per function, under outdir/D{dim}/fX/):
- full_results.csv               (one row per eps)
- survival_eps{eps}.png          (KM survival curve; includes geometric envelope)
- distance_ecdf_eps{eps}.png     (ECDF of distances-to-best at hitting time)
- extended_cluster_at_hitting.csv (optional detailed cluster stats per r_conc)

Also produces:
- outdir/D{dim}/summary_all_functions.csv

If your PKL has missing logs (positions, fitness, memory arrays), the script
will skip the corresponding diagnostics and continue.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from km_analysis import (
    compute_hitting_times,
    kaplan_meier_discrete,
    compute_km_statistics,
    geometric_envelope_rate,
)
from witness_frequency import (
    compute_witness_indicators,
    estimate_gamma,
    GAMMA_THRESHOLDS,
)
from cluster_analysis import (
    summarize_cluster_at_hitting,
    plot_distance_ecdf_at_hitting,
)


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


def _plot_survival(km, stats: Dict[str, float], out_png: Path, title: str) -> None:
    t = km.times
    # KM survival can hit exact zeros (e.g., all runs hit at t=0 for very loose eps).
    # For log-scale plotting we clip to a tiny positive value for display only.
    S = np.clip(km.S, 1e-12, 1.0)
    T = int(stats["tail_start"])
    a_env = float(stats["a_env"])

    plt.figure()
    plt.step(t, S, where="post", label="KM $\\hat S(t)$")

    if 0 <= T < t[-1] and a_env > 0 and S[T] > 0:
        env = np.ones_like(S)
        env[:T] = np.nan
        env[T:] = S[T] * (1.0 - a_env) ** (t[T:] - T)
        env = np.clip(env, 1e-12, 1.0)
        plt.plot(t, env, linestyle="--", label=f"Geom envelope (a_env={a_env:.3g})")

    plt.yscale("log")
    plt.xlabel("generation t")
    plt.ylabel("survival $\\hat S(t)$ (log scale)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def analyze_function(
    key: str,
    runs: List[Dict],
    dim: int,
    eps_values: Sequence[float],
    outdir: Path,
    r_conc_for_cluster: float = 10.0,
    thresholds: Dict[str, float] = GAMMA_THRESHOLDS,
) -> pd.DataFrame:
    f_star = f_star_from_key(key)
    curves = [np.asarray(r["curve"], dtype=float) for r in runs]
    B = int(max(len(c) for c in curves) - 1)

    # report availability
    hist0 = runs[0].get("history", {}) if runs else {}
    has_positions = ("positions" in hist0) and ("fitness" in hist0)
    has_memory = ("memory_f" in hist0) and ("memory_cr" in hist0)
    print(f"Data availability: positions={'YES' if has_positions else 'NO'}, memory={'YES' if has_memory else 'NO'}")

    # precompute witness indicators per run (if memory is present)
    indicators = []
    if has_memory:
        for r in runs:
            hist = r.get("history", {})
            indicators.append(compute_witness_indicators(hist.get("memory_f", []), hist.get("memory_cr", []), thresholds=thresholds))
    else:
        indicators = [compute_witness_indicators([], [], thresholds=thresholds) for _ in runs]

    rows = []
    for eps in eps_values:
        eps = float(eps)
        taus = compute_hitting_times(curves, f_star=f_star, eps=eps)
        km = kaplan_meier_discrete(taus, B=B)
        km_stats = compute_km_statistics(taus, B=B, tail_start=None, alpha=0.05)

        gamma_stats = estimate_gamma(
            taus=taus,
            indicators=indicators,
            B=B,
            start_gen=0,
            end_gen=None,
            alive_strict=True,
        )

        # concentration proxy at hitting (median)
        conc = summarize_cluster_at_hitting(runs, taus=taus, r_conc=r_conc_for_cluster)
        beta1_med = float(conc.get("beta1_median", np.nan))
        n_clust_med = float(conc.get("n_clust_median", np.nan))
        diam_med = float(conc.get("diameter_median", np.nan))
        N_tau_med = pop_size_median_at_hitting(runs, taus)

        row = {
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
            "N_tau_median": N_tau_med,
            f"n_clust_med_r{r_conc_for_cluster:g}": n_clust_med,
            f"beta1_med_r{r_conc_for_cluster:g}": beta1_med,
            f"diameter_med_r{r_conc_for_cluster:g}": diam_med,
        }
        rows.append(row)

        # plots
        surv_png = outdir / f"survival_eps{eps:g}.png"
        _plot_survival(km, km_stats, surv_png, title=f"{key} (eps={eps:g})")

        if has_positions:
            dist_png = outdir / f"distance_ecdf_eps{eps:g}.png"
            plot_distance_ecdf_at_hitting(runs, taus=taus, out_png=dist_png)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "full_results.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified analysis driver (read PKL once)")
    ap.add_argument("--pkl", required=True, help="Path to raw_results_lshade.pkl")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="results/tables")
    ap.add_argument("--eps", type=str, default="0.01,1,10,100,400")
    ap.add_argument("--funcs", type=str, default="", help="comma-separated list like f1,f5,f11; empty = all")
    ap.add_argument("--r_conc", type=float, default=10.0, help="r_conc used for at-hitting concentration summaries")
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    eps_values = [float(x) for x in args.eps.split(",") if x.strip()]

    if args.funcs.strip():
        funcs = [f.strip() for f in args.funcs.split(",") if f.strip()]
        keys = [f"cec2017_{f}" if not f.startswith("cec2017_") else f for f in funcs]
    else:
        keys = list_function_keys(data)

    outroot = Path(args.outdir) / f"D{args.dim}"
    outroot.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for key in keys:
        if key not in data:
            print(f"[WARN] missing key: {key}")
            continue
        fshort = key.replace("cec2017_", "")
        fdir = outroot / fshort
        fdir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print(f"ANALYZING {key}  (f*={f_star_from_key(key):g})")
        print(f"Output dir: {fdir}")
        print("=" * 70)

        df = analyze_function(
            key=key,
            runs=data[key],
            dim=args.dim,
            eps_values=eps_values,
            outdir=fdir,
            r_conc_for_cluster=args.r_conc,
        )
        df.insert(0, "function", key)
        all_dfs.append(df)

    if all_dfs:
        summary = pd.concat(all_dfs, ignore_index=True)
        summary.to_csv(outroot / "summary_all_functions.csv", index=False)
        print(f"\nSaved: {outroot / 'summary_all_functions.csv'}")


if __name__ == "__main__":
    main()
