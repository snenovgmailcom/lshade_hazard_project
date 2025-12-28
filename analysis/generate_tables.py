#!/usr/bin/env python3
"""
Generate all tables and figures for the L-SHADE hazard paper.

Usage:
    python -m analysis.generate_tables --dim 10 --pkl experiments/r_lshade_D10/raw_results_lshade.pkl
    
    # Or analyze all dimensions:
    python -m analysis.generate_tables --all
"""

import argparse
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional

from .km_analysis import (
    CEC2017_OPTIMA,
    compute_hitting_times,
    compute_km_statistics,
    kaplan_meier,
)
from .witness_frequency import (
    estimate_gamma,
    compute_theoretical_a_t,
)
from .gap_analysis import (
    compute_gap_analysis,
    format_gap_table_row,
)


# Default epsilon values
EPS_VALUES = [0.01, 1, 10, 100, 400]

# Default thresholds for gamma estimation
GAMMA_THRESHOLDS = {
    'F_minus': 0.1,
    'F_plus': 0.9,
    'g_minus': 0.1,
    'c_cr': 0.5,
    'q_minus': 0.25,
    'sigma_f': 0.1,
    'sigma_cr': 0.1,
}


def load_results(pkl_path: str) -> Dict[str, List[Dict]]:
    """Load raw results from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def analyze_single_function(
    runs: List[Dict],
    fname: str,
    f_global: float,
    d: int,
    eps_values: List[float] = EPS_VALUES,
    gamma_thresholds: Dict = GAMMA_THRESHOLDS,
) -> List[Dict]:
    """
    Full analysis for a single function.
    
    Returns list of result dictionaries, one per epsilon value.
    """
    curves = [np.array(r['curve']) for r in runs]
    hitting_times = compute_hitting_times(curves, f_global, eps_values)
    
    results = []
    
    for eps in eps_values:
        taus = hitting_times[eps]
        
        # KM analysis
        km_stats = compute_km_statistics(taus)
        
        # Gamma estimation (only if hits exist and history available)
        if km_stats['n_hits'] > 0 and 'history' in runs[0]:
            gamma_stats = estimate_gamma(runs, taus, **gamma_thresholds)
        else:
            gamma_stats = {'gamma_mean': None, 'l2_mean': None, 'l3_mean': None}
        
        # Theoretical a_t
        theoretical = compute_theoretical_a_t(d, **{
            'g_minus': gamma_thresholds['g_minus'],
            'q_minus': gamma_thresholds['q_minus'],
            'c_cr': gamma_thresholds['c_cr'],
        })
        
        # Gap analysis
        gap = compute_gap_analysis(km_stats, gamma_stats, theoretical)
        
        results.append({
            'function': fname,
            'd': d,
            'eps': eps,
            'n_runs': km_stats['n_runs'],
            'n_hits': km_stats['n_hits'],
            'hit_rate': km_stats['hit_rate'],
            'T_first': km_stats['T_first'],
            'tau_min': km_stats.get('tau_min'),
            'tau_med': km_stats['tau_med'],
            'tau_max': km_stats['tau_max'],
            'p_hat': km_stats['p_hat'],
            'a_valid': km_stats['a_valid'],
            'clustering': km_stats['clustering'],
            'gamma_mean': gamma_stats.get('gamma_mean'),
            'l2_mean': gamma_stats.get('l2_mean'),
            'l3_mean': gamma_stats.get('l3_mean'),
            'a_t': theoretical['a_t'],
            'predicted_hazard': gap.get('predicted_hazard'),
            'gap_factor': gap.get('gap_factor'),
            'km_times': km_stats.get('km_times'),
            'km_survival': km_stats.get('km_survival'),
        })
    
    return results


def analyze_all_functions(
    raw_results: Dict[str, List[Dict]],
    d: int,
    eps_values: List[float] = EPS_VALUES,
) -> pd.DataFrame:
    """
    Analyze all functions and return results as DataFrame.
    """
    all_results = []
    
    for fname, runs in raw_results.items():
        if fname not in CEC2017_OPTIMA:
            print(f"Warning: {fname} not in CEC2017_OPTIMA, skipping")
            continue
        
        f_global = CEC2017_OPTIMA[fname]
        print(f"Analyzing {fname} (f* = {f_global})...")
        
        results = analyze_single_function(runs, fname, f_global, d, eps_values)
        all_results.extend(results)
    
    return pd.DataFrame(all_results)


# =============================================================================
# Table Generation
# =============================================================================

def generate_success_rate_table(df: pd.DataFrame, output_dir: str):
    """
    Generate success rate table by function and epsilon.
    """
    # Pivot table: rows = functions, columns = eps values
    pivot = df.pivot_table(
        index='function',
        columns='eps',
        values='n_hits',
        aggfunc='first'
    )
    
    # Format as "hits/51"
    n_runs = df['n_runs'].iloc[0]
    pivot_formatted = pivot.applymap(lambda x: f"{int(x)}/{n_runs}" if pd.notna(x) else "--")
    
    # Save CSV
    pivot_formatted.to_csv(f"{output_dir}/success_rates.csv")
    
    # Generate LaTeX
    latex = pivot_formatted.to_latex(
        caption="Success rates (hits/51 runs) by function and tolerance $\\varepsilon$",
        label="tab:success-rates"
    )
    with open(f"{output_dir}/success_rates.tex", 'w') as f:
        f.write(latex)
    
    print(f"Saved: {output_dir}/success_rates.csv")
    print(f"Saved: {output_dir}/success_rates.tex")
    
    return pivot_formatted


def generate_km_table(df: pd.DataFrame, eps: float, output_dir: str):
    """
    Generate detailed KM statistics table for a specific epsilon.
    """
    df_eps = df[df['eps'] == eps].copy()
    
    # Select and format columns
    cols = ['function', 'n_hits', 'T_first', 'tau_med', 'tau_max', 'p_hat', 'a_valid', 'clustering']
    df_table = df_eps[cols].copy()
    
    # Format numeric columns
    df_table['hits'] = df_table['n_hits'].apply(lambda x: f"{x}/51")
    df_table['p_hat'] = df_table['p_hat'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "--")
    df_table['a_valid'] = df_table['a_valid'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "--")
    df_table['clustering'] = df_table['clustering'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "--")
    df_table['T_first'] = df_table['T_first'].apply(lambda x: str(int(x)) if pd.notna(x) else "--")
    df_table['tau_med'] = df_table['tau_med'].apply(lambda x: str(int(x)) if pd.notna(x) else "--")
    df_table['tau_max'] = df_table['tau_max'].apply(lambda x: str(int(x)) if pd.notna(x) else "--")
    
    # Reorder columns
    df_table = df_table[['function', 'hits', 'T_first', 'tau_med', 'tau_max', 'p_hat', 'a_valid', 'clustering']]
    
    # Save
    df_table.to_csv(f"{output_dir}/km_table_eps{eps}.csv", index=False)
    
    latex = df_table.to_latex(
        index=False,
        caption=f"Kaplan-Meier analysis for $\\varepsilon = {eps}$",
        label=f"tab:km-eps{eps}"
    )
    with open(f"{output_dir}/km_table_eps{eps}.tex", 'w') as f:
        f.write(latex)
    
    print(f"Saved: {output_dir}/km_table_eps{eps}.csv")
    
    return df_table


def generate_gamma_table(df: pd.DataFrame, eps: float, output_dir: str):
    """
    Generate witness frequency and gap analysis table.
    """
    df_eps = df[(df['eps'] == eps) & (df['n_hits'] > 0)].copy()
    
    if len(df_eps) == 0:
        print(f"No hits for eps={eps}, skipping gamma table")
        return None
    
    cols = ['function', 'n_hits', 'gamma_mean', 'l2_mean', 'l3_mean', 'a_t', 'p_hat', 'gap_factor']
    df_table = df_eps[cols].copy()
    
    # Format
    df_table['hits'] = df_table['n_hits'].apply(lambda x: f"{x}/51")
    df_table['gamma_mean'] = df_table['gamma_mean'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    df_table['l2_mean'] = df_table['l2_mean'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    df_table['l3_mean'] = df_table['l3_mean'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    df_table['a_t'] = df_table['a_t'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "--")
    df_table['p_hat'] = df_table['p_hat'].apply(lambda x: f"{x:.2e}" if pd.notna(x) else "--")
    df_table['gap_factor'] = df_table['gap_factor'].apply(
        lambda x: f"{x:.1e}" if pd.notna(x) and np.isfinite(x) else "--"
    )
    
    df_table = df_table[['function', 'hits', 'gamma_mean', 'l2_mean', 'l3_mean', 'a_t', 'p_hat', 'gap_factor']]
    
    df_table.to_csv(f"{output_dir}/gamma_table_eps{eps}.csv", index=False)
    print(f"Saved: {output_dir}/gamma_table_eps{eps}.csv")
    
    return df_table


def generate_dimension_scaling_table(
    results_by_dim: Dict[int, pd.DataFrame],
    eps: float,
    output_dir: str
):
    """
    Generate dimension scaling comparison table.
    """
    rows = []
    
    for d, df in sorted(results_by_dim.items()):
        df_eps = df[df['eps'] == eps]
        
        for _, row in df_eps.iterrows():
            rows.append({
                'function': row['function'],
                'd': d,
                'hits': f"{row['n_hits']}/51",
                'T_first': row['T_first'],
                'tau_med': row['tau_med'],
                'p_hat': row['p_hat'],
                'clustering': row['clustering'],
                'gamma_mean': row.get('gamma_mean'),
            })
    
    df_table = pd.DataFrame(rows)
    
    # Pivot by dimension
    pivot = df_table.pivot_table(
        index='function',
        columns='d',
        values=['hits', 'p_hat', 'clustering'],
        aggfunc='first'
    )
    
    pivot.to_csv(f"{output_dir}/dimension_scaling_eps{eps}.csv")
    print(f"Saved: {output_dir}/dimension_scaling_eps{eps}.csv")
    
    return pivot


# =============================================================================
# Figure Generation
# =============================================================================

def plot_survival_curves(
    df: pd.DataFrame,
    eps: float,
    output_dir: str,
    max_plots: int = 12
):
    """
    Plot KM survival curves for functions with hits.
    """
    df_eps = df[(df['eps'] == eps) & (df['n_hits'] > 0)].copy()
    
    if len(df_eps) == 0:
        print(f"No hits for eps={eps}, skipping survival plots")
        return
    
    # Sort by hit rate descending
    df_eps = df_eps.sort_values('hit_rate', ascending=False).head(max_plots)
    
    n_funcs = len(df_eps)
    n_cols = min(4, n_funcs)
    n_rows = (n_funcs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_funcs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (_, row) in enumerate(df_eps.iterrows()):
        ax = axes[idx]
        
        times = row['km_times']
        survival = row['km_survival']
        
        if times is not None and survival is not None:
            ax.step(times, survival, where='post', linewidth=1.5)
            if row['T_first'] is not None:
                ax.axvline(row['T_first'], color='red', linestyle='--', alpha=0.7)
        
        ax.set_title(f"{row['function']}\n{row['n_hits']}/51, C={row['clustering']:.2f}" 
                     if row['clustering'] else f"{row['function']}\n{row['n_hits']}/51")
        ax.set_xlabel('Generation')
        ax.set_ylabel('$\\hat{S}(t)$')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_funcs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Kaplan-Meier Survival Curves ($\\varepsilon = {eps}$)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/survival_curves_eps{eps}.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_dir}/survival_curves_eps{eps}.png")


def plot_clustering_distribution(df: pd.DataFrame, eps: float, output_dir: str):
    """
    Plot distribution of clustering indices.
    """
    df_eps = df[(df['eps'] == eps) & (df['clustering'].notna())].copy()
    
    if len(df_eps) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    clustering = df_eps['clustering'].values
    
    ax.hist(clustering, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(clustering.mean(), color='red', linestyle='--', 
               label=f'Mean = {clustering.mean():.2f}')
    ax.axvline(0.5, color='green', linestyle=':', label='C = 0.5 (threshold)')
    
    ax.set_xlabel('Clustering Index $\\mathcal{C} = a_{valid}/\\hat{p}_T$')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Clustering Index ($\\varepsilon = {eps}$)\n'
                 f'C < 0.5: clustered, C ≈ 1: geometric')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/clustering_dist_eps{eps}.png", dpi=150)
    plt.close()
    
    print(f"Saved: {output_dir}/clustering_dist_eps{eps}.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate tables and figures for L-SHADE paper")
    
    parser.add_argument('--pkl', type=str, help='Path to raw_results_lshade.pkl')
    parser.add_argument('--dim', type=int, help='Dimension (extracted from data if not provided)')
    parser.add_argument('--outdir', type=str, default='results/tables', help='Output directory')
    parser.add_argument('--eps', type=str, default='0.01,1,10,100,400',
                        help='Comma-separated epsilon values')
    parser.add_argument('--all-dims', action='store_true',
                        help='Analyze all dimensions (looks for D10, D30, D50 directories)')
    parser.add_argument('--exp-dir', type=str, default='experiments',
                        help='Experiments base directory')
    
    args = parser.parse_args()
    
    eps_values = [float(x) for x in args.eps.split(',')]
    os.makedirs(args.outdir, exist_ok=True)
    
    if args.all_dims:
        # Analyze all dimensions
        results_by_dim = {}
        
        for d in [10, 30, 50]:
            # Try different directory patterns
            patterns = [
                f"{args.exp_dir}/r_lshade_D{d}/raw_results_lshade.pkl",
                f"{args.exp_dir}/r_lshade_D{d}_nfev_*/raw_results_lshade.pkl",
            ]
            
            pkl_path = None
            for pattern in patterns:
                import glob
                matches = glob.glob(pattern)
                if matches:
                    pkl_path = matches[0]
                    break
            
            if pkl_path and os.path.exists(pkl_path):
                print(f"\n{'='*60}")
                print(f"Analyzing D={d} from {pkl_path}")
                print(f"{'='*60}")
                
                raw_results = load_results(pkl_path)
                df = analyze_all_functions(raw_results, d, eps_values)
                results_by_dim[d] = df
                
                # Save per-dimension results
                dim_outdir = f"{args.outdir}/D{d}"
                os.makedirs(dim_outdir, exist_ok=True)
                
                df.to_csv(f"{dim_outdir}/full_results.csv", index=False)
                
                generate_success_rate_table(df, dim_outdir)
                
                for eps in eps_values:
                    generate_km_table(df, eps, dim_outdir)
                    generate_gamma_table(df, eps, dim_outdir)
                    plot_survival_curves(df, eps, dim_outdir)
                    plot_clustering_distribution(df, eps, dim_outdir)
            else:
                print(f"Warning: No results found for D={d}")
        
        # Generate cross-dimension tables
        if len(results_by_dim) > 1:
            for eps in eps_values:
                generate_dimension_scaling_table(results_by_dim, eps, args.outdir)
    
    elif args.pkl:
        # Analyze single file
        print(f"Loading: {args.pkl}")
        raw_results = load_results(args.pkl)
        
        # Infer dimension from first run
        first_runs = list(raw_results.values())[0]
        d = args.dim or len(first_runs[0]['curve'])  # Approximate from curve length
        
        print(f"Dimension: {d}")
        print(f"Functions: {len(raw_results)}")
        print(f"Epsilon values: {eps_values}")
        
        df = analyze_all_functions(raw_results, d, eps_values)
        
        # Save full results
        df.to_csv(f"{args.outdir}/full_results.csv", index=False)
        print(f"\nSaved: {args.outdir}/full_results.csv")
        
        # Generate tables
        generate_success_rate_table(df, args.outdir)
        
        for eps in eps_values:
            generate_km_table(df, eps, args.outdir)
            generate_gamma_table(df, eps, args.outdir)
            plot_survival_curves(df, eps, args.outdir)
            plot_clustering_distribution(df, eps, args.outdir)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        for eps in eps_values:
            df_eps = df[df['eps'] == eps]
            n_with_hits = (df_eps['n_hits'] > 0).sum()
            n_full_success = (df_eps['n_hits'] == 51).sum()
            print(f"eps={eps}: {n_with_hits}/30 functions with hits, {n_full_success} with 100% success")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
