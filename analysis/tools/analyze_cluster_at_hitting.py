#!/usr/bin/env python3
"""
analyze_witness_regime.py

At τ_deep (first hit into A_{ε/4}), count mass in 4 zones:
  A_deep ⊂ A_in ⊂ A_out ⊂ [l,u]^D

Output: median counts across runs per function.
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_f_star(func_id: int) -> float:
    """CEC2017: f* = 100 * func_id."""
    return 100.0 * func_id


def load_runs(dim: int, func_id: int, base: Path) -> Optional[List[Dict]]:
    """Load runs from pickle."""
    pkl_path = base / f"D{dim}" / f"f{func_id}" / f"f{func_id}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    key = f"cec2017_f{func_id}"
    return data.get(key, None)


def compute_eps_in(runs: List[Dict], f_star: float, rho: float) -> float:
    """ε_in = max{(1+ρ)*m, 1} where m = median(final_errors)."""
    final_errors = []
    for run in runs:
        curve = run.get("curve", [])
        if curve is not None and len(curve) > 0:
            final_errors.append(float(curve[-1]) - f_star)
    if len(final_errors) == 0:
        return 1.0
    m = float(np.median(final_errors))
    return max((1 + rho) * m, 1.0)


def find_tau(fitness_hist: List, f_star: float, eps: float) -> Optional[int]:
    """First t where min(f) ≤ f* + eps."""
    for t, f_t in enumerate(fitness_hist):
        f_arr = np.asarray(f_t, dtype=float)
        if f_arr.size > 0 and float(np.min(f_arr)) <= f_star + eps:
            return t
    return None


def compute_delta_q80(f_t: np.ndarray, f_star: float, eps_in: float) -> float:
    """80th percentile gap of outsiders at τ_in."""
    boundary = f_star + eps_in
    gaps = f_t[f_t > boundary] - boundary
    if gaps.size == 0:
        return 0.0
    return float(np.percentile(gaps, 80))


def compute_diameters(X: np.ndarray) -> Tuple[float, float]:
    """Compute diam(A_out) and diam_center(A_out).
    
    Returns:
        diam_max: max pairwise distance
        diam_center: 2 × max distance to centroid
    """
    if X.shape[0] == 0:
        return np.nan, np.nan
    if X.shape[0] == 1:
        return 0.0, 0.0
    
    # diam_center: 2 × max distance to centroid
    centroid = np.mean(X, axis=0)
    dists_to_center = np.linalg.norm(X - centroid, axis=1)
    diam_center = 2.0 * float(np.max(dists_to_center))
    
    # diam_max: max pairwise distance
    from scipy.spatial.distance import pdist
    if X.shape[0] > 1:
        diam_max = float(np.max(pdist(X)))
    else:
        diam_max = 0.0
    
    return diam_max, diam_center


def analyze_run(run: Dict, f_star: float, eps_in: float, eps_deep: float) -> Optional[Dict]:
    """Analyze one run. Returns counts at τ_deep."""
    hist = run.get("history", {})
    fitness_hist = hist.get("fitness", [])
    positions_hist = hist.get("positions", [])
    if len(fitness_hist) == 0:
        return None

    # τ_in: first hit into A_in
    tau_in = find_tau(fitness_hist, f_star, eps_in)
    if tau_in is None:
        return None

    # Δ_0.8 at τ_in → ε_out
    f_at_tau_in = np.asarray(fitness_hist[tau_in], dtype=float)
    delta_q80 = compute_delta_q80(f_at_tau_in, f_star, eps_in)
    eps_out = eps_in + delta_q80

    # τ_deep: first hit into A_deep
    tau_deep = find_tau(fitness_hist, f_star, eps_deep)
    if tau_deep is None:
        return None

    # Counts at τ_deep
    f_t = np.asarray(fitness_hist[tau_deep], dtype=float)
    N = f_t.size

    n_deep = int(np.sum(f_t <= f_star + eps_deep))
    n_in = int(np.sum(f_t <= f_star + eps_in))
    n_out = int(np.sum(f_t <= f_star + eps_out))

    # Diameters of A_out at τ_deep
    diam_max, diam_center = np.nan, np.nan
    if tau_deep < len(positions_hist):
        X_t = np.asarray(positions_hist[tau_deep], dtype=float)
        if X_t.ndim == 2 and X_t.shape[0] == N:
            in_A_out = f_t <= f_star + eps_out
            X_out = X_t[in_A_out]
            diam_max, diam_center = compute_diameters(X_out)

    return {
        "tau_deep": tau_deep,
        "eps_out": eps_out,
        "n_deep": n_deep,
        "n_in_shell": n_in - n_deep,
        "n_out_shell": n_out - n_in,
        "n_outside": N - n_out,
        "diam_max": diam_max,
        "diam_center": diam_center,
    }


def process_function(dim: int, func_id: int, rho: float, base: Path) -> Optional[Dict]:
    """Process one function, return median summary."""
    runs = load_runs(dim, func_id, base)
    if runs is None or len(runs) == 0:
        return None

    f_star = get_f_star(func_id)
    eps_in = compute_eps_in(runs, f_star, rho)
    eps_deep = eps_in / 4.0

    results = []
    for run in runs:
        res = analyze_run(run, f_star, eps_in, eps_deep)
        if res is not None:
            results.append(res)

    if len(results) == 0:
        return None

    return {
        "Func": f"f{func_id}",
        "τ_deep": int(np.median([r["tau_deep"] for r in results])),
        "ε_in": eps_in,
        "#(A_deep)": int(np.median([r["n_deep"] for r in results])),
        "#(A_in \\ A_deep)": int(np.median([r["n_in_shell"] for r in results])),
        "ε_out": float(np.median([r["eps_out"] for r in results])),
        "#(A_out \\ A_in)": int(np.median([r["n_out_shell"] for r in results])),
        "diam(A_out)": float(np.median([r["diam_max"] for r in results])),
        "diam_center(A_out)": float(np.median([r["diam_center"] for r in results])),
        "#([l,u]^D \\ A_out)": int(np.median([r["n_outside"] for r in results])),
    }


def plot_cloud_from_summary(rows: List[Dict], dim: int, rho: float, outdir: Path):
    """Create cloud plot from summary data for all functions."""
    import matplotlib.patches as mpatches
    
    n_funcs = len(rows)
    if n_funcs == 0:
        return
    
    fig, ax = plt.subplots(figsize=(16, n_funcs * 0.6 + 2))
    y_positions = np.arange(n_funcs)[::-1]  # Reverse so f1 is at top

    for i, d in enumerate(rows):
        y = y_positions[i]
        func = d['Func']
        
        eps_in = d['ε_in']
        eps_out = d['ε_out']
        eps_deep = eps_in / 4
        n_deep = d['#(A_deep)']
        n_in = d['#(A_in \\ A_deep)']
        n_out = d['#(A_out \\ A_in)']
        
        # Normalize to [0, 1] where 1 = eps_out
        scale = eps_out
        
        x_deep = eps_deep / scale
        x_in = eps_in / scale
        x_out = 1.0
        
        # Draw zone backgrounds
        ax.axhspan(y - 0.35, y + 0.35, xmin=0, xmax=x_deep/1.1, color='red', alpha=0.15)
        ax.axhspan(y - 0.35, y + 0.35, xmin=x_deep/1.1, xmax=x_in/1.1, color='orange', alpha=0.15)
        ax.axhspan(y - 0.35, y + 0.35, xmin=x_in/1.1, xmax=x_out/1.1, color='steelblue', alpha=0.15)
        
        # Draw boundary lines
        ax.plot([0, 0], [y - 0.3, y + 0.3], 'g-', linewidth=2)
        ax.plot([x_deep, x_deep], [y - 0.3, y + 0.3], 'r--', linewidth=1.5, alpha=0.7)
        ax.plot([x_in, x_in], [y - 0.3, y + 0.3], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot([x_out, x_out], [y - 0.3, y + 0.3], 'b--', linewidth=1.5, alpha=0.7)
        
        # Place points as dots
        np.random.seed(42 + i)
        
        # A_deep points (red)
        if n_deep > 0:
            x_pts = np.random.uniform(0, x_deep, n_deep)
            y_pts = np.random.uniform(y - 0.2, y + 0.2, n_deep)
            ax.scatter(x_pts, y_pts, c='red', s=50, marker='o', edgecolors='darkred', zorder=5)
        
        # A_in \ A_deep points (orange)
        if n_in > 0:
            x_pts = np.random.uniform(x_deep, x_in, n_in)
            y_pts = np.random.uniform(y - 0.25, y + 0.25, n_in)
            ax.scatter(x_pts, y_pts, c='orange', s=15, marker='o', alpha=0.7, zorder=4)
        
        # A_out \ A_in points (blue)
        if n_out > 0:
            x_pts = np.random.uniform(x_in, x_out, n_out)
            y_pts = np.random.uniform(y - 0.25, y + 0.25, n_out)
            ax.scatter(x_pts, y_pts, c='steelblue', s=15, marker='o', alpha=0.7, zorder=4)
            # Median marker
            median_x = (x_in + x_out) / 2
            ax.scatter([median_x], [y], c='blue', s=100, marker='D', edgecolors='darkblue', linewidths=1.5, zorder=6)
        
        # Function label
        ax.text(-0.08, y, func, ha='right', va='center', fontsize=10, fontweight='bold')
        
        # Count labels
        ax.text(1.02, y, f'({n_deep} | {n_in} | {n_out})', ha='left', va='center', fontsize=8, color='gray')

    # Labels at top
    ax.text(0, n_funcs + 0.3, 'f*', ha='center', fontsize=9, color='green', fontweight='bold')
    ax.text(0.05, n_funcs + 0.3, 'ε/4', ha='center', fontsize=9, color='red')
    ax.text(0.15, n_funcs + 0.3, 'ε_in', ha='center', fontsize=9, color='orange')
    ax.text(0.91, n_funcs + 0.3, 'ε_out', ha='center', fontsize=9, color='steelblue')

    ax.set_xlim(-0.12, 1.15)
    ax.set_ylim(-0.8, n_funcs + 0.7)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('error (normalized to ε_out)', fontsize=11)
    ax.set_title(f'Population distribution at τ_deep: A_deep | A_in\\A_deep | A_out\\A_in (D={dim}, ρ={rho})\n(counts shown as #deep | #in | #out)', fontsize=13)

    # Legend
    legend_elements = [
        mpatches.Patch(color='red', alpha=0.5, label='A_deep'),
        mpatches.Patch(color='orange', alpha=0.5, label='A_in \\ A_deep'),
        mpatches.Patch(color='steelblue', alpha=0.5, label='A_out \\ A_in'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', markersize=10, label='median(A_out\\A_in)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    outpath = outdir / f"witness_cloud_D{dim}.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--func", type=str, required=True, help="f1..f30 or all")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--base", type=str, default="experiments")
    args = ap.parse_args()

    base = Path(args.base)
    func_ids = list(range(1, 31)) if args.func.lower() == "all" else [int(args.func.lower().replace("f", ""))]

    outdir = Path(f"results/witness_regime/D{args.dim}_rho{args.rho}")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fid in func_ids:
        row = process_function(args.dim, fid, args.rho, base)
        if row is not None:
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        
        # Save CSV
        csv_path = outdir / "summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
        
        # Save cloud figure
        plot_cloud_from_summary(rows, args.dim, args.rho, outdir)


if __name__ == "__main__":
    main()
