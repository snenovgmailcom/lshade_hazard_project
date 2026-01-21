#!/usr/bin/env python3
"""
analyze_cluster_at_hitting.py

Analyze where the population is when hitting occurs.

Key questions:
1. First moment when 1 agent enters A_in: Where are OTHER agents?
2. First moment when K agents are in A_in: Where are OTHER agents?

eps_in: Derived from error distribution (error_median + margin)
eps_out: COMPUTED as mean distance of other agents to A_in (not a parameter!)

Output:
- Distance distribution of "other" agents at hitting time
- How many generations until K agents are in A_in
- Cluster spread relative to A_in boundary

Usage:
    python analysis/analyze_cluster_at_hitting.py \
        --pkl experiments/D10/f1/f1.pkl \
        --func f1 --dim 10 \
        --outdir results/cluster_hitting/D10/f1
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pkl(pkl_path: Path) -> Dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_func_key(data: Dict, func: str) -> str:
    func = func.strip()
    if func in data:
        return func
    if func.startswith("f") and func[1:].isdigit():
        key = f"cec2017_{func}"
        if key in data:
            return key
    for k in data.keys():
        if k.endswith(f"_{func}") or k == func:
            return k
    raise KeyError(f"Function {func} not found")


def f_star_from_func(func: str) -> float:
    import re
    m = re.search(r"f(\d+)", func)
    if not m:
        raise ValueError(f"Cannot parse function id from {func}")
    return float(100 * int(m.group(1)))


def compute_eps_in_from_errors(runs: List[Dict], f_star: float, margin: float = 5.0) -> Tuple[float, Dict]:
    """
    Compute eps_in based on final error distribution.
    
    eps_in = error_median + margin
    
    For perfect functions (error=0), use small positive value.
    """
    final_errors = []
    for run in runs:
        curve = run.get("curve", [])
        if len(curve) > 0:
            final_errors.append(curve[-1] - f_star)
    
    final_errors = np.array(final_errors)
    
    error_median = np.median(final_errors)
    error_mean = np.mean(final_errors)
    error_min = np.min(final_errors)
    error_max = np.max(final_errors)
    
    # eps_in = median error + margin (but at least margin/10 for perfect functions)
    eps_in = max(error_median + margin, margin / 10)
    
    stats = {
        "error_min": float(error_min),
        "error_median": float(error_median),
        "error_mean": float(error_mean),
        "error_max": float(error_max),
        "margin": margin,
        "eps_in": float(eps_in),
    }
    
    return eps_in, stats


def analyze_run_cluster_at_hitting(
    run: Dict,
    f_star: float,
    eps_in: float,
    K_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """
    Analyze cluster position at different hitting thresholds.
    
    For each K in K_values:
    - Find tau_K = first time when at least K agents are in A_in
    - Compute distances of OTHER agents to A_in boundary
    
    Returns detailed analysis including:
    - tau_K for each K
    - Distance distribution of "others" at each tau_K
    - Implied eps_out (mean distance of others)
    """
    hist = run.get("history", {})
    curve = np.asarray(run.get("curve", []), dtype=float)
    
    positions = hist.get("positions", [])
    fitness = hist.get("fitness", [])
    
    n_gens = min(len(positions), len(fitness))
    
    results = {
        "f_star": f_star,
        "eps_in": eps_in,
    }
    
    for K in K_values:
        # Find tau_K: first time with at least K agents in A_in
        tau_K = None
        for t in range(n_gens):
            f_t = np.asarray(fitness[t], dtype=float)
            in_Ain = f_t <= f_star + eps_in
            if in_Ain.sum() >= K:
                tau_K = t
                break
        
        if tau_K is None:
            # Never reached K agents in A_in
            results[f"tau_{K}"] = np.inf
            results[f"n_in_Ain_at_tau_{K}"] = 0
            results[f"others_dist_mean_{K}"] = np.nan
            results[f"others_dist_median_{K}"] = np.nan
            results[f"others_dist_min_{K}"] = np.nan
            results[f"others_dist_max_{K}"] = np.nan
            results[f"implied_eps_out_{K}"] = np.nan
            continue
        
        # Analyze population at tau_K
        f_tau = np.asarray(fitness[tau_K], dtype=float)
        X_tau = np.asarray(positions[tau_K], dtype=float)
        N = len(f_tau)
        
        in_Ain = f_tau <= f_star + eps_in
        n_in_Ain = int(in_Ain.sum())
        
        # Distance to A_in boundary for OTHERS (not in A_in)
        # Distance = f - (f* + eps_in) for agents outside A_in
        # (negative means inside A_in)
        distances_to_Ain = f_tau - (f_star + eps_in)
        
        others_mask = ~in_Ain
        n_others = int(others_mask.sum())
        
        if n_others > 0:
            others_dist = distances_to_Ain[others_mask]
            others_dist_mean = float(np.mean(others_dist))
            others_dist_median = float(np.median(others_dist))
            others_dist_min = float(np.min(others_dist))
            others_dist_max = float(np.max(others_dist))
            others_dist_std = float(np.std(others_dist))
            others_dist_q25 = float(np.percentile(others_dist, 25))
            others_dist_q75 = float(np.percentile(others_dist, 75))
        else:
            # All agents in A_in
            others_dist_mean = 0.0
            others_dist_median = 0.0
            others_dist_min = 0.0
            others_dist_max = 0.0
            others_dist_std = 0.0
            others_dist_q25 = 0.0
            others_dist_q75 = 0.0
        
        # Implied eps_out: eps_in + mean distance of others
        implied_eps_out = eps_in + max(0, others_dist_mean)
        
        # Also compute: distances of those IN A_in (how deep are they?)
        if n_in_Ain > 0:
            in_depths = -(distances_to_Ain[in_Ain])  # Positive = deeper in A_in
            in_depth_mean = float(np.mean(in_depths))
            in_depth_max = float(np.max(in_depths))
        else:
            in_depth_mean = 0.0
            in_depth_max = 0.0
        
        # Spatial clustering: how spread out are agents in A_in?
        if n_in_Ain > 1:
            X_in = X_tau[in_Ain]
            centroid = X_in.mean(axis=0)
            spatial_spread = np.mean(np.linalg.norm(X_in - centroid, axis=1))
        else:
            spatial_spread = 0.0
        
        results[f"tau_{K}"] = tau_K
        results[f"n_in_Ain_at_tau_{K}"] = n_in_Ain
        results[f"n_others_{K}"] = n_others
        results[f"N_at_tau_{K}"] = N
        
        # Distances of others
        results[f"others_dist_mean_{K}"] = others_dist_mean
        results[f"others_dist_median_{K}"] = others_dist_median
        results[f"others_dist_min_{K}"] = others_dist_min
        results[f"others_dist_max_{K}"] = others_dist_max
        results[f"others_dist_std_{K}"] = others_dist_std
        results[f"others_dist_q25_{K}"] = others_dist_q25
        results[f"others_dist_q75_{K}"] = others_dist_q75
        
        # Implied eps_out
        results[f"implied_eps_out_{K}"] = implied_eps_out
        
        # Depth of those in A_in
        results[f"in_depth_mean_{K}"] = in_depth_mean
        results[f"in_depth_max_{K}"] = in_depth_max
        
        # Spatial spread
        results[f"spatial_spread_{K}"] = spatial_spread
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze cluster at hitting time")
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--func", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--margin", type=float, default=5.0, 
                        help="Margin to add to error_median for eps_in")
    parser.add_argument("--eps_in", type=float, default=None,
                        help="Override eps_in (skip adaptive)")
    parser.add_argument("--K_values", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="K values for K-agent hitting analysis")
    parser.add_argument("--outdir", required=True)
    
    args = parser.parse_args()
    
    # Load data
    data = load_pkl(Path(args.pkl))
    func_key = get_func_key(data, args.func)
    runs = data[func_key]
    f_star = f_star_from_func(func_key)
    
    print(f"Function: {func_key}, f*={f_star}, runs={len(runs)}")
    
    # Compute eps_in
    if args.eps_in is not None:
        eps_in = args.eps_in
        eps_stats = {"eps_in": eps_in, "source": "override"}
    else:
        eps_in, eps_stats = compute_eps_in_from_errors(runs, f_star, args.margin)
    
    print(f"\nError distribution:")
    print(f"  min:    {eps_stats.get('error_min', 'N/A')}")
    print(f"  median: {eps_stats.get('error_median', 'N/A')}")
    print(f"  mean:   {eps_stats.get('error_mean', 'N/A')}")
    print(f"  max:    {eps_stats.get('error_max', 'N/A')}")
    print(f"\nUsing eps_in = {eps_in:.6g}")
    print(f"K values: {args.K_values}")
    
    # Analyze each run
    results = []
    for i, run in enumerate(runs):
        res = analyze_run_cluster_at_hitting(run, f_star, eps_in, args.K_values)
        res["run"] = i
        results.append(res)
    
    df = pd.DataFrame(results)
    
    # Output
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(outdir / "cluster_at_hitting.csv", index=False)
    
    # Summary for each K
    print(f"\n{'='*70}")
    print("CLUSTER ANALYSIS AT HITTING TIME")
    print(f"{'='*70}")
    
    summary = {
        "function": func_key,
        "f_star": f_star,
        "n_runs": len(runs),
        "eps_in": eps_in,
        **eps_stats,
    }
    
    for K in args.K_values:
        tau_col = f"tau_{K}"
        successful = df[np.isfinite(df[tau_col])]
        n_success = len(successful)
        
        print(f"\n--- K = {K} (at least {K} agent(s) in A_in) ---")
        print(f"Runs reaching K={K}: {n_success}/{len(runs)} ({n_success/len(runs):.1%})")
        
        summary[f"n_success_K{K}"] = n_success
        summary[f"success_rate_K{K}"] = n_success / len(runs)
        
        if n_success > 0:
            tau_med = successful[tau_col].median()
            others_mean_med = successful[f"others_dist_mean_{K}"].median()
            others_median_med = successful[f"others_dist_median_{K}"].median()
            implied_eps_out = successful[f"implied_eps_out_{K}"].median()
            
            print(f"  τ_{K} (median):           {tau_med:.0f}")
            print(f"  n_in_Ain at τ_{K}:        {successful[f'n_in_Ain_at_tau_{K}'].median():.0f}")
            print(f"  n_others at τ_{K}:        {successful[f'n_others_{K}'].median():.0f}")
            print(f"  Others dist mean:         {others_mean_med:.2f}")
            print(f"  Others dist median:       {others_median_med:.2f}")
            print(f"  Others dist min:          {successful[f'others_dist_min_{K}'].median():.2f}")
            print(f"  Others dist max:          {successful[f'others_dist_max_{K}'].median():.2f}")
            print(f"  => Implied eps_out:       {implied_eps_out:.2f}")
            
            summary[f"tau_{K}_median"] = float(tau_med)
            summary[f"others_dist_mean_K{K}"] = float(others_mean_med)
            summary[f"others_dist_median_K{K}"] = float(others_median_med)
            summary[f"implied_eps_out_K{K}"] = float(implied_eps_out)
            
            # Classification: How close are others?
            very_close = successful[successful[f"others_dist_mean_{K}"] < eps_in]
            close = successful[(successful[f"others_dist_mean_{K}"] >= eps_in) & 
                              (successful[f"others_dist_mean_{K}"] < eps_in * 10)]
            far = successful[successful[f"others_dist_mean_{K}"] >= eps_in * 10]
            
            print(f"\n  Classification (others_dist_mean):")
            print(f"    Very close (<eps_in):     {len(very_close)} ({len(very_close)/n_success:.1%})")
            print(f"    Close (eps_in to 10×):    {len(close)} ({len(close)/n_success:.1%})")
            print(f"    Far (>10× eps_in):        {len(far)} ({len(far)/n_success:.1%})")
    
    # Save summary
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {outdir / 'summary.json'}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    K_main = args.K_values[0]  # Usually K=1
    successful = df[np.isfinite(df[f"tau_{K_main}"])]
    
    if len(successful) > 0:
        # Panel 1: Distribution of others_dist_mean
        ax1 = axes[0, 0]
        others_dist = successful[f"others_dist_mean_{K_main}"].dropna()
        if len(others_dist) > 0:
            ax1.hist(others_dist, bins=30, edgecolor='black', alpha=0.7)
            ax1.axvline(others_dist.median(), color='red', linestyle='--', 
                        label=f'Median={others_dist.median():.1f}')
            ax1.axvline(eps_in, color='green', linestyle=':', linewidth=2,
                        label=f'eps_in={eps_in:.2g}')
        ax1.set_xlabel(f'Mean distance of others to A_in (K={K_main})')
        ax1.set_ylabel('Count')
        ax1.set_title(f'How far are "others" when first {K_main} agent(s) hit?')
        ax1.legend()
        
        # Panel 2: τ_K for different K
        ax2 = axes[0, 1]
        tau_medians = []
        K_labels = []
        for K in args.K_values:
            succ_K = df[np.isfinite(df[f"tau_{K}"])]
            if len(succ_K) > 0:
                tau_medians.append(succ_K[f"tau_{K}"].median())
                K_labels.append(f"K={K}")
        if tau_medians:
            ax2.bar(K_labels, tau_medians, alpha=0.7)
            ax2.set_ylabel('Median τ_K')
            ax2.set_title('Time to reach K agents in A_in')
        
        # Panel 3: Others distance distribution (boxplot for different K)
        ax3 = axes[1, 0]
        box_data = []
        box_labels = []
        for K in args.K_values:
            succ_K = df[np.isfinite(df[f"tau_{K}"])]
            if len(succ_K) > 0:
                box_data.append(succ_K[f"others_dist_mean_{K}"].dropna().values)
                box_labels.append(f"K={K}")
        if box_data:
            ax3.boxplot(box_data, labels=box_labels)
            ax3.axhline(eps_in, color='green', linestyle=':', label=f'eps_in={eps_in:.2g}')
            ax3.set_ylabel('Mean distance of others to A_in')
            ax3.set_title('Others distance by K')
            ax3.legend()
        
        # Panel 4: Implied eps_out
        ax4 = axes[1, 1]
        eps_out_data = []
        for K in args.K_values:
            succ_K = df[np.isfinite(df[f"tau_{K}"])]
            if len(succ_K) > 0:
                eps_out_data.append(succ_K[f"implied_eps_out_{K}"].median())
            else:
                eps_out_data.append(0)
        ax4.bar([f"K={K}" for K in args.K_values], eps_out_data, alpha=0.7, color='orange')
        ax4.axhline(eps_in, color='green', linestyle=':', linewidth=2, label=f'eps_in={eps_in:.2g}')
        ax4.set_ylabel('Implied eps_out (median)')
        ax4.set_title('Implied A_out boundary')
        ax4.legend()
    
    plt.suptitle(f'{func_key}: Cluster at Hitting (eps_in={eps_in:.4g})', fontsize=14)
    plt.tight_layout()
    plt.savefig(outdir / 'cluster_at_hitting.png', dpi=200)
    plt.close()
    print(f"Saved: {outdir / 'cluster_at_hitting.png'}")


if __name__ == "__main__":
    main()
