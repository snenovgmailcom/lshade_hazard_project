#!/usr/bin/env python3
"""
plot_convergence_comparison.py

Create a multi-panel comparison figure showing different concentration regimes:
  - f1/f3: Fast concentration (small gap between τ_deep and τ_pair)
  - f5/f10: Never concentrates (dispersion)
  - f11: Long plateau then rapid descent

Usage:
  python plot_convergence_comparison.py --dim 10 --base experiments \
    --per_run_csv out_fix123/morse_validate_D10_all.per_run.csv \
    --outdir plots_convergence
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
    curve = run.get("curve", None)
    if curve is not None and len(curve) > 0:
        arr = np.asarray(curve, dtype=float) - f_star
        return np.maximum(arr, 1e-16)
    
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


def get_function_data(
    fid: int, dim: int, base: Path, df: pd.DataFrame
) -> Tuple[List[np.ndarray], List[float], List[float], List[float], float, float]:
    """Get curves and tau values for a function."""
    f_star = get_f_star_cec2017(fid)
    runs = load_runs(dim, fid, base)
    
    if runs is None:
        return [], [], [], [], np.nan, np.nan
    
    df_func = df[df["fid"] == fid].copy()
    if df_func.empty:
        return [], [], [], [], np.nan, np.nan
    
    eps_in = df_func["eps_in"].iloc[0]
    eps_deep = df_func["eps_deep"].iloc[0]
    
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
    
    return curves, tau_deeps, tau_C2s, tau_pairs, eps_in, eps_deep


def plot_panel(
    ax: plt.Axes,
    curves: List[np.ndarray],
    tau_deeps: List[float],
    tau_C2s: List[float],
    tau_pairs: List[float],
    eps_in: float,
    eps_deep: float,
    title: str,
    max_gen: Optional[int] = None,
    show_legend: bool = True,
):
    """Plot a single panel with convergence curves and tau markers."""
    if not curves:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    # Plot individual curves (faint)
    max_len = max(len(c) for c in curves)
    for curve in curves:
        generations = np.arange(len(curve))
        ax.semilogy(generations, curve, 'b-', alpha=0.15, linewidth=0.5)
    
    # Compute and plot median curve
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
    ax.semilogy(generations, median_curve, 'b-', linewidth=2.5, label='Median')
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
    
    # Statistics
    n_C2 = sum(1 for t in tau_C2s if np.isfinite(t))
    n_pair = sum(1 for t in tau_pairs if np.isfinite(t))
    n_total = len(curves)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Generation", fontsize=10)
    ax.set_ylabel(r"$f_{\mathrm{best}} - f^*$", fontsize=10)
    
    # Add stats text
    stats_text = f"C2: {n_C2}/{n_total} ({100*n_C2/n_total:.0f}%)\npair: {n_pair}/{n_total} ({100*n_pair/n_total:.0f}%)"
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if max_gen:
        ax.set_xlim(0, max_gen)
    
    ax.grid(True, alpha=0.3)
    
    if show_legend:
        ax.legend(loc='lower left', fontsize=8)


def main():
    ap = argparse.ArgumentParser(description="Plot convergence comparison")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--base", type=str, default="experiments")
    ap.add_argument("--per_run_csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="plots")
    ap.add_argument("--max_gen", type=int, default=None)
    
    args = ap.parse_args()
    
    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.per_run_csv)
    
    # Define comparison groups
    comparisons = [
        {
            "name": "three_regimes",
            "functions": [
                (1, "f1: Unimodal (fast concentration)"),
                (5, "f5: Multimodal (dispersion)"),
                (11, "f11: Hybrid (long plateau)"),
            ],
            "figsize": (15, 5),
            "layout": (1, 3),
        },
        {
            "name": "unimodal_comparison",
            "functions": [
                (1, "f1: Bent Cigar"),
                (3, "f3: Zakharov"),
                (4, "f4: Rosenbrock"),
            ],
            "figsize": (15, 5),
            "layout": (1, 3),
        },
        {
            "name": "multimodal_comparison",
            "functions": [
                (5, "f5: Rastrigin"),
                (7, "f7: Griewank"),
                (9, "f9: Rastrigin (shifted)"),
            ],
            "figsize": (15, 5),
            "layout": (1, 3),
        },
        {
            "name": "hybrid_comparison",
            "functions": [
                (11, "f11: Hybrid 1"),
                (14, "f14: Hybrid 4"),
                (17, "f17: Hybrid 7"),
            ],
            "figsize": (15, 5),
            "layout": (1, 3),
        },
    ]
    
    for comp in comparisons:
        fig, axes = plt.subplots(comp["layout"][0], comp["layout"][1], 
                                  figsize=comp["figsize"])
        if comp["layout"][0] == 1:
            axes = [axes] if comp["layout"][1] == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        for idx, (fid, title) in enumerate(comp["functions"]):
            curves, tau_deeps, tau_C2s, tau_pairs, eps_in, eps_deep = get_function_data(
                fid, args.dim, base, df
            )
            
            plot_panel(
                axes[idx], curves, tau_deeps, tau_C2s, tau_pairs,
                eps_in, eps_deep, title,
                max_gen=args.max_gen,
                show_legend=(idx == 0)  # Legend only on first panel
            )
        
        fig.suptitle(f"CEC2017 Convergence Comparison (D={args.dim})", fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        out_path = outdir / f"comparison_{comp['name']}_D{args.dim}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
    
    # Also create a single 2x3 comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    all_functions = [
        (1, "f1: Unimodal (Bent Cigar)"),
        (3, "f3: Unimodal (Zakharov)"),
        (5, "f5: Multimodal (Rastrigin)"),
        (11, "f11: Hybrid 1"),
        (17, "f17: Hybrid 7 (dispersion)"),
        (22, "f22: Composition"),
    ]
    
    for idx, (fid, title) in enumerate(all_functions):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        curves, tau_deeps, tau_C2s, tau_pairs, eps_in, eps_deep = get_function_data(
            fid, args.dim, base, df
        )
        
        plot_panel(
            ax, curves, tau_deeps, tau_C2s, tau_pairs,
            eps_in, eps_deep, title,
            max_gen=args.max_gen,
            show_legend=(idx == 0)
        )
    
    fig.suptitle(f"CEC2017 Concentration Regimes (D={args.dim})", fontsize=16, fontweight='bold')
    
    out_path = outdir / f"comparison_comprehensive_D{args.dim}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
