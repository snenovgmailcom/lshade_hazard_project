#!/usr/bin/env python3
"""
Kaplan-Meier Survival Analysis for L-SHADE First-Hitting Times
===============================================================

This script analyzes first-hitting time data from L-SHADE experiments
and produces:
1. Kaplan-Meier survival estimates
2. Empirical hazard rates
3. Tightest geometric envelope bounds
4. LaTeX tables for the paper

Usage:
    python km_analysis.py --data-dir experiments/r_lshade_D10_nfev_100000 --epsilon 0.01
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class KMResult:
    """Kaplan-Meier analysis result for a single function."""
    func_id: int
    n_runs: int
    n_hits: int
    hit_rate: float
    times: np.ndarray          # Unique event times
    n_risk: np.ndarray         # Number at risk at each time
    n_events: np.ndarray       # Number of events at each time
    survival: np.ndarray       # KM survival estimate
    hazard: np.ndarray         # Empirical hazard at each time
    variance: np.ndarray       # Greenwood variance estimate
    ci_lower: np.ndarray       # 95% CI lower bound
    ci_upper: np.ndarray       # 95% CI upper bound
    median_time: Optional[float]  # Median survival time (if estimable)
    mean_time: Optional[float]    # Mean hitting time (among hits)
    a_valid: Optional[float]      # Tightest geometric envelope rate
    T_burnin: int                 # Burn-in start time for envelope


@dataclass 
class RunData:
    """Data from a single L-SHADE run."""
    func_id: int
    run_id: int
    hit_time: Optional[int]    # None if censored
    censored: bool
    best_fitness: float
    n_evals: int


# =============================================================================
# Data Loading
# =============================================================================

def load_pkl_data(pkl_path: Path) -> dict:
    """Load a pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def extract_hitting_times(data_dir: Path, epsilon: float = 0.01, 
                          f_star_offset: float = 100.0) -> Dict[int, List[RunData]]:
    """
    Extract first-hitting times from all pkl files in a directory.
    
    For CEC2017, f* = 100 * func_id, so target is f* + epsilon.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing pkl files
    epsilon : float
        Success threshold (target = f* + epsilon)
    f_star_offset : float
        f* = f_star_offset * func_id for CEC2017
        
    Returns
    -------
    Dict mapping func_id -> list of RunData
    """
    results = {}
    
    # Find all pkl files
    pkl_files = sorted(data_dir.glob("*.pkl"))
    
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in {data_dir}")
    
    print(f"Found {len(pkl_files)} pkl files in {data_dir}")
    
    for pkl_path in pkl_files:
        try:
            data = load_pkl_data(pkl_path)
        except Exception as e:
            print(f"  Warning: Could not load {pkl_path}: {e}")
            continue
            
        # Try to extract function ID from filename or data
        # Common patterns: "F1_run0.pkl", "func1_seed42.pkl", etc.
        func_id = None
        run_id = None
        
        # Try to get from data dict
        if isinstance(data, dict):
            func_id = data.get('func_id') or data.get('function_id') or data.get('fid')
            run_id = data.get('run_id') or data.get('seed') or data.get('run')
            
        # Try to parse from filename
        if func_id is None:
            name = pkl_path.stem.lower()
            import re
            # Match patterns like "f1", "func1", "function_1"
            match = re.search(r'f(?:unc(?:tion)?)?[_]?(\d+)', name)
            if match:
                func_id = int(match.group(1))
            # Match "run0", "seed42"
            match = re.search(r'(?:run|seed)[_]?(\d+)', name)
            if match:
                run_id = int(match.group(1))
                
        if func_id is None:
            print(f"  Warning: Could not determine func_id for {pkl_path}")
            continue
            
        if run_id is None:
            run_id = 0
            
        # Extract hitting time
        f_star = f_star_offset * func_id
        target = f_star + epsilon
        
        hit_time = None
        best_fitness = None
        n_evals = None
        
        if isinstance(data, dict):
            # Try various common keys
            history = (data.get('convergence') or data.get('history') or 
                      data.get('best_history') or data.get('fitness_history'))
            
            if history is not None:
                history = np.array(history)
                # Find first time below target
                hits = np.where(history <= target)[0]
                if len(hits) > 0:
                    hit_time = int(hits[0]) + 1  # 1-indexed
                best_fitness = float(np.min(history))
                n_evals = len(history)
            else:
                # Maybe just final result
                best_fitness = data.get('best') or data.get('best_fitness') or data.get('result')
                n_evals = data.get('nfev') or data.get('n_evals') or data.get('evaluations')
                if best_fitness is not None and best_fitness <= target:
                    # We know it hit but not when - use n_evals as upper bound
                    hit_time = n_evals
                    
        # Store result
        if func_id not in results:
            results[func_id] = []
            
        results[func_id].append(RunData(
            func_id=func_id,
            run_id=run_id,
            hit_time=hit_time,
            censored=(hit_time is None),
            best_fitness=best_fitness if best_fitness is not None else float('inf'),
            n_evals=n_evals if n_evals is not None else 0
        ))
        
    # Sort runs within each function
    for func_id in results:
        results[func_id].sort(key=lambda r: r.run_id)
        
    return results


def load_logged_runs(data_dir: Path, epsilon: float = 0.01) -> Dict[int, List[RunData]]:
    """
    Load data from logged runs (like logged_runs_D10.pkl format).
    
    This handles the format from r_benchmark_logged.py which stores
    detailed per-generation data.
    """
    results = {}
    
    # Look for the consolidated logged runs file
    logged_file = data_dir / "logged_runs_D10.pkl"
    if not logged_file.exists():
        # Try other patterns
        for pattern in ["logged_runs*.pkl", "*logged*.pkl"]:
            files = list(data_dir.glob(pattern))
            if files:
                logged_file = files[0]
                break
                
    if not logged_file.exists():
        return results
        
    print(f"Loading logged runs from {logged_file}")
    
    with open(logged_file, 'rb') as f:
        data = pickle.load(f)
        
    # Expected format: list of dicts with 'func_id', 'run_id', 'generations', etc.
    if isinstance(data, list):
        for run in data:
            func_id = run.get('func_id', 1)
            run_id = run.get('run_id', run.get('seed', 0))
            
            # Get convergence history
            generations = run.get('generations', [])
            if generations:
                # Each generation has 'best' fitness
                history = [g.get('best', float('inf')) for g in generations]
                f_star = 100.0 * func_id
                target = f_star + epsilon
                
                hits = [i for i, f in enumerate(history) if f <= target]
                hit_time = hits[0] + 1 if hits else None
                
                if func_id not in results:
                    results[func_id] = []
                    
                results[func_id].append(RunData(
                    func_id=func_id,
                    run_id=run_id,
                    hit_time=hit_time,
                    censored=(hit_time is None),
                    best_fitness=min(history) if history else float('inf'),
                    n_evals=run.get('nfev', len(history))
                ))
                
    return results


# =============================================================================
# Kaplan-Meier Estimation
# =============================================================================

def kaplan_meier(hit_times: List[Optional[int]], 
                 max_time: int,
                 burnin_frac: float = 0.1,
                 min_risk: int = 10) -> KMResult:
    """
    Compute Kaplan-Meier survival estimate with geometric envelope.
    
    Parameters
    ----------
    hit_times : list
        List of hitting times (None for censored runs)
    max_time : int
        Maximum observation time (censoring time for non-hits)
    burnin_frac : float
        Fraction of max_time to use as burn-in for geometric envelope
    min_risk : int
        Minimum number at risk for envelope calculation
        
    Returns
    -------
    KMResult with all computed quantities
    """
    n_runs = len(hit_times)
    n_hits = sum(1 for t in hit_times if t is not None)
    
    # Convert to observed times and event indicators
    observed_times = []
    event_indicators = []
    
    for t in hit_times:
        if t is not None:
            observed_times.append(t)
            event_indicators.append(1)  # Event (hit)
        else:
            observed_times.append(max_time)
            event_indicators.append(0)  # Censored
            
    observed_times = np.array(observed_times)
    event_indicators = np.array(event_indicators)
    
    # Get unique event times (only where events occurred)
    event_times = np.unique(observed_times[event_indicators == 1])
    
    if len(event_times) == 0:
        # No events observed
        return KMResult(
            func_id=0,
            n_runs=n_runs,
            n_hits=0,
            hit_rate=0.0,
            times=np.array([max_time]),
            n_risk=np.array([n_runs]),
            n_events=np.array([0]),
            survival=np.array([1.0]),
            hazard=np.array([0.0]),
            variance=np.array([0.0]),
            ci_lower=np.array([1.0]),
            ci_upper=np.array([1.0]),
            median_time=None,
            mean_time=None,
            a_valid=None,
            T_burnin=max_time
        )
    
    # Compute at-risk counts and event counts at each event time
    n_risk = []
    n_events = []
    
    for t in event_times:
        # Number at risk just before time t
        at_risk = np.sum(observed_times >= t)
        n_risk.append(at_risk)
        
        # Number of events at time t
        events_at_t = np.sum((observed_times == t) & (event_indicators == 1))
        n_events.append(events_at_t)
        
    n_risk = np.array(n_risk)
    n_events = np.array(n_events)
    
    # Compute KM survival estimate: S(t) = prod_{s<=t} (1 - d_s/Y_s)
    hazard = n_events / n_risk  # Empirical hazard at each event time
    survival = np.cumprod(1 - hazard)
    
    # Greenwood's variance formula
    # Var(S(t)) = S(t)^2 * sum_{s<=t} d_s / (Y_s * (Y_s - d_s))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        greenwood_terms = n_events / (n_risk * (n_risk - n_events))
        greenwood_terms = np.where(np.isfinite(greenwood_terms), greenwood_terms, 0)
        
    variance = survival**2 * np.cumsum(greenwood_terms)
    
    # 95% confidence intervals using log-log transform
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        se = np.sqrt(variance)
        # Log-log transform for better CI coverage
        log_log_se = se / (survival * np.abs(np.log(survival + 1e-10)))
        z = 1.96
        
        ci_lower = survival ** np.exp(z * log_log_se)
        ci_upper = survival ** np.exp(-z * log_log_se)
        
        # Clip to [0, 1]
        ci_lower = np.clip(ci_lower, 0, 1)
        ci_upper = np.clip(ci_upper, 0, 1)
        
    # Median survival time
    median_idx = np.where(survival <= 0.5)[0]
    median_time = float(event_times[median_idx[0]]) if len(median_idx) > 0 else None
    
    # Mean hitting time (among hits only)
    actual_hits = [t for t in hit_times if t is not None]
    mean_time = float(np.mean(actual_hits)) if actual_hits else None
    
    # Compute tightest geometric envelope after burn-in
    T_burnin = max(1, int(burnin_frac * max_time))
    
    # Find first event time >= T_burnin with sufficient risk
    valid_idx = np.where((event_times >= T_burnin) & (n_risk >= min_risk))[0]
    
    a_valid = None
    if len(valid_idx) > 0:
        start_idx = valid_idx[0]
        T_burnin = int(event_times[start_idx])
        S_T = survival[start_idx]
        
        if S_T > 0:
            # For each time n >= T, find the root (S(n)/S(T))^{1/(n-T+1)}
            max_root = 0.0
            for i in range(start_idx, len(event_times)):
                if survival[i] > 0:
                    n = event_times[i]
                    exponent = n - T_burnin + 1
                    if exponent > 0:
                        root = (survival[i] / S_T) ** (1.0 / exponent)
                        max_root = max(max_root, root)
                        
            a_valid = 1.0 - max_root if max_root < 1.0 else None
    
    return KMResult(
        func_id=0,
        n_runs=n_runs,
        n_hits=n_hits,
        hit_rate=n_hits / n_runs,
        times=event_times,
        n_risk=n_risk,
        n_events=n_events,
        survival=survival,
        hazard=hazard,
        variance=variance,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        median_time=median_time,
        mean_time=mean_time,
        a_valid=a_valid,
        T_burnin=T_burnin
    )


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_all_functions(run_data: Dict[int, List[RunData]], 
                          max_time: int = None) -> Dict[int, KMResult]:
    """Run Kaplan-Meier analysis for all functions."""
    results = {}
    
    for func_id, runs in sorted(run_data.items()):
        if not runs:
            continue
            
        hit_times = [r.hit_time for r in runs]
        
        # Determine max observation time
        if max_time is None:
            obs_max = max(r.n_evals for r in runs if r.n_evals)
            if obs_max == 0:
                obs_max = 100000  # Default
        else:
            obs_max = max_time
            
        km = kaplan_meier(hit_times, obs_max)
        km.func_id = func_id
        results[func_id] = km
        
    return results


def classify_convergence(km: KMResult) -> str:
    """
    Classify convergence behavior based on survival curve shape.
    
    Returns: 'clustered', 'geometric', 'moderate', or 'no_hits'
    """
    if km.n_hits == 0:
        return 'no_hits'
        
    if km.n_hits < 5:
        return 'insufficient'
        
    # Compute ratio of a_valid to average hazard
    if km.a_valid is None or km.a_valid <= 0:
        return 'moderate'
        
    avg_hazard = np.mean(km.hazard) if len(km.hazard) > 0 else 0
    
    if avg_hazard <= 0:
        return 'moderate'
        
    ratio = km.a_valid / avg_hazard
    
    if ratio < 0.15:
        return 'clustered'  # Hits are bunched together
    elif ratio > 0.7:
        return 'geometric'  # Constant-rate like
    else:
        return 'moderate'


# =============================================================================
# Output Generation
# =============================================================================

def generate_latex_table(results: Dict[int, KMResult], caption: str = "") -> str:
    """Generate LaTeX table of KM results."""
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\small",
        r"\begin{tabular}{@{}rrrrrrrrl@{}}",
        r"\toprule",
        r"$f$ & Runs & Hits & Hit\% & $\bar{\tau}$ & $\tilde{\tau}$ & $T$ & $a_{\mathrm{valid}}$ & Type \\",
        r"\midrule"
    ]
    
    for func_id, km in sorted(results.items()):
        hit_pct = f"{100*km.hit_rate:.1f}"
        mean_t = f"{km.mean_time:.0f}" if km.mean_time else "--"
        median_t = f"{km.median_time:.0f}" if km.median_time else "--"
        T_str = f"{km.T_burnin}"
        a_str = f"{km.a_valid:.4f}" if km.a_valid else "--"
        conv_type = classify_convergence(km)
        
        lines.append(
            f"F{func_id} & {km.n_runs} & {km.n_hits} & {hit_pct} & "
            f"{mean_t} & {median_t} & {T_str} & {a_str} & {conv_type} \\\\"
        )
        
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:km-results}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)


def generate_survival_data_table(results: Dict[int, KMResult]) -> str:
    """Generate detailed survival data for selected functions."""
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering", 
        r"\caption{Kaplan--Meier survival estimates for selected CEC2017 functions (D=10)}",
        r"\small",
        r"\begin{tabular}{@{}rrrrrr@{}}",
        r"\toprule",
        r"Function & $t$ & $Y_t$ & $d_t$ & $\hat{h}_t$ & $\hat{S}(t)$ \\",
        r"\midrule"
    ]
    
    # Select functions with interesting behavior
    selected = [fid for fid, km in results.items() 
                if km.n_hits >= 10 and len(km.times) >= 3][:5]
    
    for func_id in selected:
        km = results[func_id]
        
        # Show first few and last few time points
        n_show = min(5, len(km.times))
        indices = list(range(n_show))
        if len(km.times) > n_show:
            indices = [0, 1, 2, -2, -1]
            
        first = True
        for i in indices:
            if i < 0:
                i = len(km.times) + i
            if i >= len(km.times):
                continue
                
            func_str = f"F{func_id}" if first else ""
            first = False
            
            t = int(km.times[i])
            Y = int(km.n_risk[i])
            d = int(km.n_events[i])
            h = km.hazard[i]
            S = km.survival[i]
            
            lines.append(f"{func_str} & {t} & {Y} & {d} & {h:.4f} & {S:.4f} \\\\")
            
        lines.append(r"\midrule")
        
    lines[-1] = r"\bottomrule"  # Replace last midrule
    
    lines.extend([
        r"\end{tabular}",
        r"\label{tab:km-survival}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)


def generate_summary_stats(results: Dict[int, KMResult]) -> str:
    """Generate summary statistics text."""
    
    n_funcs = len(results)
    n_with_hits = sum(1 for km in results.values() if km.n_hits > 0)
    
    # Classification counts
    types = [classify_convergence(km) for km in results.values()]
    type_counts = {t: types.count(t) for t in set(types)}
    
    # Average hit rate among functions with hits
    hit_rates = [km.hit_rate for km in results.values() if km.n_hits > 0]
    avg_hit_rate = np.mean(hit_rates) if hit_rates else 0
    
    # a_valid statistics
    a_vals = [km.a_valid for km in results.values() if km.a_valid is not None]
    
    lines = [
        "=" * 60,
        "SUMMARY STATISTICS",
        "=" * 60,
        f"Total functions analyzed: {n_funcs}",
        f"Functions with hits: {n_with_hits}",
        f"Average hit rate (among hitters): {100*avg_hit_rate:.1f}%",
        "",
        "Convergence type distribution:",
    ]
    
    for t, count in sorted(type_counts.items()):
        lines.append(f"  {t}: {count} functions")
        
    if a_vals:
        lines.extend([
            "",
            f"Geometric envelope rate a_valid:",
            f"  Min:  {min(a_vals):.6f}",
            f"  Max:  {max(a_vals):.6f}",
            f"  Mean: {np.mean(a_vals):.6f}",
        ])
        
    return "\n".join(lines)


def plot_survival_curves(results: Dict[int, KMResult], output_path: Path):
    """Generate survival curve plots (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
        
    # Select functions with sufficient hits
    selected = [(fid, km) for fid, km in sorted(results.items()) 
                if km.n_hits >= 10]
    
    if not selected:
        print("No functions with sufficient hits for plotting")
        return
        
    # Create figure with subplots
    n_funcs = min(6, len(selected))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    for idx, (func_id, km) in enumerate(selected[:n_funcs]):
        ax = axes[idx]
        
        # Plot KM estimate as step function
        # Extend times to start at 0 with S=1
        times_ext = np.concatenate([[0], km.times])
        surv_ext = np.concatenate([[1.0], km.survival])
        
        ax.step(times_ext, surv_ext, where='post', linewidth=2, label='KM estimate')
        
        # Plot confidence bands
        ci_lower_ext = np.concatenate([[1.0], km.ci_lower])
        ci_upper_ext = np.concatenate([[1.0], km.ci_upper])
        ax.fill_between(times_ext, ci_lower_ext, ci_upper_ext, 
                       alpha=0.3, step='post', label='95% CI')
        
        # Plot geometric envelope if available
        if km.a_valid is not None and km.a_valid > 0:
            T = km.T_burnin
            t_env = np.arange(T, int(km.times[-1]) + 1)
            S_T = km.survival[np.searchsorted(km.times, T, side='right') - 1]
            if S_T > 0:
                S_env = S_T * (1 - km.a_valid) ** (t_env - T + 1)
                ax.plot(t_env, S_env, 'r--', linewidth=1.5, 
                       label=f'Geom. env. (a={km.a_valid:.4f})')
        
        ax.set_xlabel('Evaluations')
        ax.set_ylabel('Survival probability')
        ax.set_title(f'F{func_id} (n={km.n_runs}, hits={km.n_hits})')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
    # Hide unused subplots
    for idx in range(n_funcs, len(axes)):
        axes[idx].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved survival curves to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kaplan-Meier survival analysis for L-SHADE experiments"
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing pkl files')
    parser.add_argument('--epsilon', type=float, default=0.01,
                       help='Success threshold (default: 0.01)')
    parser.add_argument('--max-evals', type=int, default=None,
                       help='Maximum evaluations (censoring time)')
    parser.add_argument('--output-dir', type=str, default='results/km_analysis',
                       help='Output directory')
    parser.add_argument('--plot', action='store_true',
                       help='Generate survival curve plots')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {data_dir}")
    print(f"Success threshold: epsilon = {args.epsilon}")
    
    # Try different loading methods
    run_data = {}
    
    # Method 1: Individual pkl files
    try:
        run_data = extract_hitting_times(data_dir, epsilon=args.epsilon)
    except Exception as e:
        print(f"Could not load individual pkl files: {e}")
        
    # Method 2: Logged runs format
    if not run_data:
        try:
            run_data = load_logged_runs(data_dir, epsilon=args.epsilon)
        except Exception as e:
            print(f"Could not load logged runs: {e}")
            
    if not run_data:
        print("ERROR: Could not load any data!")
        return
        
    # Print summary of loaded data
    print(f"\nLoaded data for {len(run_data)} functions:")
    for func_id, runs in sorted(run_data.items()):
        n_hits = sum(1 for r in runs if r.hit_time is not None)
        print(f"  F{func_id}: {len(runs)} runs, {n_hits} hits")
        
    # Run KM analysis
    print("\nRunning Kaplan-Meier analysis...")
    results = analyze_all_functions(run_data, max_time=args.max_evals)
    
    # Generate outputs
    print("\nGenerating outputs...")
    
    # Summary statistics
    summary = generate_summary_stats(results)
    print("\n" + summary)
    
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSaved summary to {summary_path}")
    
    # LaTeX tables
    latex_table = generate_latex_table(
        results, 
        caption="Kaplan--Meier survival analysis for CEC2017 (D=10, $\\varepsilon=0.01$)"
    )
    
    latex_path = output_dir / "km_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to {latex_path}")
    
    # Detailed survival table
    detail_table = generate_survival_data_table(results)
    detail_path = output_dir / "survival_detail.tex"
    with open(detail_path, 'w') as f:
        f.write(detail_table)
    print(f"Saved detailed table to {detail_path}")
    
    # Save raw results as pickle
    results_path = output_dir / "km_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {results_path}")
    
    # Generate plots if requested
    if args.plot:
        plot_path = output_dir / "survival_curves.png"
        plot_survival_curves(results, plot_path)
        
    print("\nDone!")


if __name__ == '__main__':
    main()
