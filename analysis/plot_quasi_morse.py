#!/usr/bin/env python3
"""
plot_convergence_tau.py

Plot convergence curves (best fitness over generations) with vertical lines
marking critical times: τ_deep, τ_C2, τ_pair.

Usage:
  python plot_convergence_tau.py --dim 10 --func f1 --base experiments \
    --per_run_csv out_fix123/morse_validate_D10_f1.per_run.csv \
    --outdir plots_convergence
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_f_star_cec2017(fid: int) -> float:
    return 100.0 * fid


def load_runs(dim: int, fid: int, base: Path) -> Optional[List[Dict[str, Any]]]:
    pkl = base / f"D{dim}" / f"f{fid}" / f"f{fid}.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return data.get(f"cec2017_f{fid}", None)


def extract_best_curve(run: Dict[str, Any], f_star: float) -> np.ndarray:
    """Extract best-so-far curve from run history."""
    # Try curve first
    curve = run.get("curve", None)
    if curve is not None and len(curve) > 0:
        arr = np.asarray(curve, dtype=float) - f_star
        return np.maximum(arr, 1e-16)  # avoid log(0)
    
    # Fallback to fitness history
    hist = run.get("history", {})
    fitness_hist = hist.get("fitness", [])
    if not fitness_hist:
        return np.array([])
    
    best_curve = []
    best_so_far = np.inf
    for f_t in fitness_hist:
        arr = np.asarray(f_t, dtype=float)
        if arr.size:
            best_so_far = min(best_so_far, float(np.min(arr)))
        best_curve.append(best_so_far - f_star)
    
    return np.maximum(np.asarray(best_curve, dtype=float), 1e-16)


def plot_single_run(
    ax: plt.Axes,
    curve: np.ndarray,
    tau_deep: float,
    tau_C2: float,
    tau_pair: float,
    eps_in: float,
    eps_deep: float,
    run_id: int,
    alpha: float = 1.0,
):
    """Plot a single run's convergence curve with tau markers."""
    generations = np.arange(len(curve))
    
    # Convergence curve
    ax.semilogy(generations, curve, 'b-', alpha=alpha, linewidth=0.8)
    
    # Horizontal lines for eps levels
    ax.axhline(y=eps_in, color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=eps_deep, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    # Vertical lines for tau times
    if np.isfinite(tau_deep):
        ax.axvline(x=tau_deep, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    if np.isfinite(tau_C2):
        ax.axvline(x=tau_C2, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
    if np.isfinite(tau_pair):
        ax.axvline(x=tau_pair, color='red', linestyle='--', alpha=0.7, linewidth=1.5)


def plot_multi_runs(
    ax: plt.Axes,
    curves: List[np.ndarray],
    tau_deeps: List[float],
    tau_C2s: List[float],
    tau_pairs: List[float],
    eps_in: float,
    eps_deep: float,
    show_median: bool = True,
):
    """Plot multiple runs with median and individual curves."""
    # Plot individual curves (faint)
    max_len = max(len(c) for c in curves)
    for curve in curves:
        generations = np.arange(len(curve))
        ax.semilogy(generations, curve, 'b-', alpha=0.15, linewidth=0.5)
    
    # Compute and plot median curve
    if show_median:
        # Pad curves to same length
        padded = []
        for c in curves:
            if len(c) < max_len:
                c = np.concatenate([c, np.full(max_len - len(c), c[-1] if len(c) > 0 else np.nan)])
            padded.append(c)
        stacked = np.vstack(padded)
        median_curve = np.nanmedian(stacked, axis=0)
        q25 = np.nanpercentile(stacked, 25, axis=0)
        q75 = np.nanpercentile(stacked, 75, axis=0)
        
        generations = np.arange(max_len)
        ax.semilogy(generations, median_curve, 'b-', linewidth=2, label='Median')
        ax.fill_between(generations, q25, q75, alpha=0.2, color='blue')
    
    # Horizontal lines for eps levels
    ax.axhline(y=eps_in, color='green', linestyle=':', alpha=0.7, linewidth=1.5, label=r'$\varepsilon_{\mathrm{in}}$')
    ax.axhline(y=eps_deep, color='orange', linestyle=':', alpha=0.7, linewidth=1.5, label=r'$\varepsilon_{\mathrm{deep}}$')
    
    # Median tau times
    tau_deep_med = np.nanmedian([t for t in tau_deeps if np.isfinite(t)])
    tau_C2_med = np.nanmedian([t for t in tau_C2s if np.isfinite(t)])
    tau_pair_med = np.nanmedian([t for t in tau_pairs if np.isfinite(t)])
    
    if np.isfinite(tau_deep_med):
        ax.axvline(x=tau_deep_med, color='green', linestyle='--', linewidth=2, label=r'$\tau_{\mathrm{deep}}$')
    if np.isfinite(tau_C2_med):
        ax.axvline(x=tau_C2_med, color='orange', linestyle='--', linewidth=2, label=r'$\tau_{C2}$')
    if np.isfinite(tau_pair_med):
        ax.axvline(x=tau_pair_med, color='red', linestyle='--', linewidth=2, label=r'$\tau_{\mathrm{pair}}$')


def main():
    ap = argparse.ArgumentParser(description="Plot convergence curves with tau markers")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--func", type=str, required=True, help="e.g., f1 or f1,f3,f5")
    ap.add_argument("--base", type=str, default="experiments")
    ap.add_argument("--per_run_csv", type=str, required=True, help="Path to per_run CSV from morse_validate")
    ap.add_argument("--outdir", type=str, default="plots")
    ap.add_argument("--single_run", type=int, default=None, help="Plot single run by ID")
    ap.add_argument("--max_gen", type=int, default=None, help="Max generation to show")
    
    args = ap.parse_args()
    
    # Parse function list
    func_str = args.func.strip().lower()
    if func_str == "all":
        fids = list(range(1, 31))
    else:
        parts = [p.strip() for p in func_str.replace(" ", "").split(",") if p]
        fids = [int(p.replace("f", "")) for p in parts]
    
    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load per-run CSV
    df = pd.read_csv(args.per_run_csv)
    
    for fid in fids:
        f_star = get_f_star_cec2017(fid)
        runs = load_runs(args.dim, fid, base)
        
        if runs is None:
            print(f"f{fid}: No data found")
            continue
        
        # Filter CSV for this function
        df_func = df[df["fid"] == fid].copy()
        if df_func.empty:
            print(f"f{fid}: No rows in CSV")
            continue
        
        eps_in = df_func["eps_in"].iloc[0]
        eps_deep = df_func["eps_deep"].iloc[0]
        
        # Match runs to CSV rows by run_id
        curves = []
        tau_deeps = []
        tau_C2s = []
        tau_pairs = []
        
        for _, row in df_func.iterrows():
            rid = int(row["run_id"])
            if rid < 0 or rid >= len(runs):
                continue
            
            curve = extract_best_curve(runs[rid], f_star)
            if curve.size == 0:
                continue
            
            curves.append(curve)
            tau_deeps.append(float(row["tau_deep"]))
            tau_C2s.append(float(row["tau_C2"]) if pd.notna(row["tau_C2"]) else np.nan)
            tau_pairs.append(float(row["tau_pair"]) if pd.notna(row["tau_pair"]) else np.nan)
        
        if not curves:
            print(f"f{fid}: No valid curves")
            continue
        
        # Single run plot
        if args.single_run is not None:
            idx = args.single_run
            if idx >= len(curves):
                print(f"f{fid}: Run {idx} not found")
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_single_run(
                ax, curves[idx],
                tau_deeps[idx], tau_C2s[idx], tau_pairs[idx],
                eps_in, eps_deep, idx
            )
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel(r"$f_{\mathrm{best}} - f^*$", fontsize=12)
            ax.set_title(f"f{fid} (D={args.dim}), Run {idx}", fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            if args.max_gen:
                ax.set_xlim(0, args.max_gen)
            
            out_path = outdir / f"convergence_D{args.dim}_f{fid}_run{idx}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")
        
        else:
            # Multi-run plot
            fig, ax = plt.subplots(figsize=(12, 7))
            plot_multi_runs(
                ax, curves,
                tau_deeps, tau_C2s, tau_pairs,
                eps_in, eps_deep
            )
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel(r"$f_{\mathrm{best}} - f^*$", fontsize=12)
            
            # Count achievements
            n_C2 = sum(1 for t in tau_C2s if np.isfinite(t))
            n_pair = sum(1 for t in tau_pairs if np.isfinite(t))
            n_total = len(curves)
            
            ax.set_title(
                f"f{fid} (D={args.dim}): {n_total} runs, "
                f"C2={n_C2}/{n_total} ({100*n_C2/n_total:.0f}%), "
                f"pair={n_pair}/{n_total} ({100*n_pair/n_total:.0f}%)",
                fontsize=14
            )
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if args.max_gen:
                ax.set_xlim(0, args.max_gen)
            
            out_path = outdir / f"convergence_D{args.dim}_f{fid}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
