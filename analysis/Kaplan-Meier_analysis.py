#!/usr/bin/env python3
"""
Kaplan-Meier Survival Analysis for L-SHADE CEC2017 Results
===========================================================

Adapted for the format:
- data['cec2017_f1'] = list of 51 run dicts
- Each run has: 'function', 'seed', 'best', 'nfev', 'nit', 'wall', 'curve', ...
- 'curve' is ndarray with convergence history (best-so-far per generation)

For CEC2017: f* = 100 * func_id, target = f* + epsilon

Usage:
    python km_cec2017.py --data-dir experiments/r_lshade_D10_nfev_100000 --epsilon 0.01
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class KMResult:
    """Kaplan-Meier result for one function."""
    func_id: int
    n_runs: int
    n_hits: int
    hit_rate: float
    hit_times: List[Optional[int]]  # Original data (None = censored)
    times: np.ndarray               # Unique event times
    n_risk: np.ndarray              # At risk at each time
    n_events: np.ndarray            # Events at each time
    survival: np.ndarray            # KM survival estimate
    hazard: np.ndarray              # Empirical hazard
    mean_time: Optional[float]      # Mean among hits
    median_time: Optional[float]    # Median among hits
    a_valid: Optional[float]        # Geometric envelope rate
    T_burnin: Optional[int]         # Burn-in time for envelope


# =============================================================================
# Data Loading
# =============================================================================

def load_cec2017_data(pkl_path: Path, epsilon: float = 0.01) -> Dict[int, List[Optional[int]]]:
    """
    Load CEC2017 results and extract first-hitting times.
    
    Parameters
    ----------
    pkl_path : Path
        Path to raw_results_lshade.pkl
    epsilon : float
        Success threshold (target = f* + epsilon)
        
    Returns
    -------
    Dict mapping func_id -> list of hitting times (None if censored)
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    results = {}
    
    for key, runs in data.items():
        # Extract func_id from key like 'cec2017_f1'
        match = re.search(r'f(\d+)', key)
        if not match:
            continue
        func_id = int(match.group(1))
        
        # For CEC2017: f* = 100 * func_id
        f_star = 100.0 * func_id
        target = f_star + epsilon
        
        hit_times = []
        for run in runs:
            curve = run['curve']  # Convergence history
            
            # Find first generation where best <= target
            hits = np.where(curve <= target)[0]
            if len(hits) > 0:
                # +1 because generation 0 is initialization
                hit_time = int(hits[0])
            else:
                hit_time = None  # Censored
                
            hit_times.append(hit_time)
            
        results[func_id] = hit_times
        
    return results


# =============================================================================
# Kaplan-Meier Estimation
# =============================================================================

def kaplan_meier(hit_times: List[Optional[int]], max_gen: int) -> KMResult:
    """
    Compute Kaplan-Meier survival estimate.
    
    Parameters
    ----------
    hit_times : list
        List of hitting times (None for censored runs)
    max_gen : int
        Maximum generation (censoring time)
        
    Returns
    -------
    KMResult with all computed quantities
    """
    n_runs = len(hit_times)
    n_hits = sum(1 for t in hit_times if t is not None)
    
    # Handle no-hit case
    if n_hits == 0:
        return KMResult(
            func_id=0, n_runs=n_runs, n_hits=0, hit_rate=0.0,
            hit_times=hit_times,
            times=np.array([max_gen]),
            n_risk=np.array([n_runs]),
            n_events=np.array([0]),
            survival=np.array([1.0]),
            hazard=np.array([0.0]),
            mean_time=None, median_time=None,
            a_valid=None, T_burnin=None
        )
    
    # Convert to observed times and event indicators
    obs_times = []
    events = []
    for t in hit_times:
        if t is not None:
            obs_times.append(t)
            events.append(1)
        else:
            obs_times.append(max_gen)
            events.append(0)
    
    obs_times = np.array(obs_times)
    events = np.array(events)
    
    # Unique event times (only where events occurred)
    event_times = np.unique(obs_times[events == 1])
    
    # Compute KM quantities at each event time
    times_list = []
    n_risk_list = []
    n_events_list = []
    hazard_list = []
    survival_list = []
    
    S = 1.0
    for t in event_times:
        Y = np.sum(obs_times >= t)  # At risk just before t
        d = np.sum((obs_times == t) & (events == 1))  # Events at t
        h = d / Y  # Empirical hazard
        S = S * (1 - h)  # Survival
        
        times_list.append(t)
        n_risk_list.append(Y)
        n_events_list.append(d)
        hazard_list.append(h)
        survival_list.append(S)
    
    # Convert to arrays
    times = np.array(times_list)
    n_risk = np.array(n_risk_list)
    n_events = np.array(n_events_list)
    hazard = np.array(hazard_list)
    survival = np.array(survival_list)
    
    # Mean and median among actual hits
    actual_hits = [t for t in hit_times if t is not None]
    mean_time = float(np.mean(actual_hits))
    median_time = float(np.median(actual_hits))
    
    # Compute geometric envelope
    a_valid, T_burnin = compute_geometric_envelope(times, survival, n_risk)
    
    return KMResult(
        func_id=0, n_runs=n_runs, n_hits=n_hits,
        hit_rate=n_hits / n_runs,
        hit_times=hit_times,
        times=times,
        n_risk=n_risk,
        n_events=n_events,
        survival=survival,
        hazard=hazard,
        mean_time=mean_time,
        median_time=median_time,
        a_valid=a_valid,
        T_burnin=T_burnin
    )


def compute_geometric_envelope(times: np.ndarray, survival: np.ndarray, 
                                n_risk: np.ndarray,
                                burnin_frac: float = 0.1,
                                min_risk: int = 5) -> Tuple[Optional[float], Optional[int]]:
    """
    Compute tightest valid geometric envelope rate.
    
    Finds largest a such that: S(n) <= S(T) * (1-a)^{n-T+1} for all n >= T
    
    Parameters
    ----------
    times : array
        Event times
    survival : array
        KM survival at each event time
    n_risk : array
        Number at risk at each event time
    burnin_frac : float
        Fraction of max time for burn-in
    min_risk : int
        Minimum at-risk count for envelope computation
        
    Returns
    -------
    (a_valid, T_burnin) or (None, None) if not computable
    """
    if len(times) == 0 or survival[-1] >= 1.0:
        return None, None
    
    # Burn-in: skip early transient
    max_t = times[-1]
    T_target = max(times[0], int(burnin_frac * max_t))
    
    # Find first valid point after burn-in with sufficient risk
    valid_mask = (times >= T_target) & (n_risk >= min_risk)
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return None, None
    
    start_idx = valid_idx[0]
    T = int(times[start_idx])
    S_T = survival[start_idx]
    
    if S_T <= 0:
        return None, None
    
    # Find maximum root: max over n >= T of (S(n)/S(T))^{1/(n-T+1)}
    max_root = 0.0
    for i in range(start_idx, len(times)):
        if survival[i] > 0:
            n = times[i]
            exponent = n - T + 1
            if exponent > 0:
                root = (survival[i] / S_T) ** (1.0 / exponent)
                max_root = max(max_root, root)
    
    if max_root >= 1.0:
        return None, T
    
    a_valid = 1.0 - max_root
    return a_valid, T


def classify_convergence(km: KMResult) -> str:
    """Classify convergence behavior."""
    if km.n_hits == 0:
        return "no_hits"
    if km.n_hits < 5:
        return "few_hits"
    if km.a_valid is None:
        return "moderate"
    
    avg_hazard = np.mean(km.hazard)
    if avg_hazard <= 0:
        return "moderate"
    
    ratio = km.a_valid / avg_hazard
    
    if ratio < 0.15:
        return "clustered"
    elif ratio > 0.7:
        return "geometric"
    else:
        return "moderate"


# =============================================================================
# Output Generation
# =============================================================================

def print_km_table(km: KMResult, func_id: int, max_rows: int = 15):
    """Print detailed KM table for one function."""
    print(f"\n{'='*70}")
    print(f"Function F{func_id}")
    print(f"{'='*70}")
    print(f"Runs: {km.n_runs}, Hits: {km.n_hits}, Hit rate: {100*km.hit_rate:.1f}%")
    
    if km.n_hits > 0:
        print(f"Mean hitting time: {km.mean_time:.1f} generations")
        print(f"Median hitting time: {km.median_time:.1f} generations")
        actual = [t for t in km.hit_times if t is not None]
        print(f"Range: [{min(actual)}, {max(actual)}]")
    
    if km.a_valid is not None:
        print(f"\nGeometric envelope (T = {km.T_burnin}):")
        print(f"  a_valid = {km.a_valid:.6f}")
        print(f"  Classification: {classify_convergence(km)}")
    
    print(f"\nKaplan-Meier survival table:")
    print(f"{'t':>8} {'Y_t':>8} {'d_t':>6} {'h_t':>10} {'S(t)':>10}")
    print("-" * 50)
    
    n = len(km.times)
    if n <= max_rows:
        indices = range(n)
    else:
        # Show first 10 and last 5
        indices = list(range(10)) + [-5, -4, -3, -2, -1]
    
    prev_idx = -1
    for idx in indices:
        if idx < 0:
            idx = n + idx
        if idx <= prev_idx:
            continue
        if prev_idx >= 0 and idx > prev_idx + 1:
            print("     ...")
        prev_idx = idx
        
        print(f"{km.times[idx]:8.0f} {km.n_risk[idx]:8.0f} {km.n_events[idx]:6.0f} "
              f"{km.hazard[idx]:10.6f} {km.survival[idx]:10.6f}")


def generate_latex_table(results: Dict[int, KMResult]) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Kaplan--Meier survival analysis for CEC2017 (D=10, $\varepsilon=0.01$). "
        r"$\bar{\tau}$: mean hitting time; $\tilde{\tau}$: median; "
        r"$T$: burn-in; $a_{\mathrm{valid}}$: geometric envelope rate.}",
        r"\small",
        r"\begin{tabular}{@{}rrrrrrrrl@{}}",
        r"\toprule",
        r"$f$ & Runs & Hits & Hit\% & $\bar{\tau}$ & $\tilde{\tau}$ & $T$ & $a_{\mathrm{valid}}$ & Type \\",
        r"\midrule"
    ]
    
    for func_id in sorted(results.keys()):
        km = results[func_id]
        
        hit_pct = f"{100*km.hit_rate:.1f}"
        mean_t = f"{km.mean_time:.0f}" if km.mean_time else "--"
        med_t = f"{km.median_time:.0f}" if km.median_time else "--"
        T_str = f"{km.T_burnin}" if km.T_burnin else "--"
        a_str = f"{km.a_valid:.4f}" if km.a_valid else "--"
        ctype = classify_convergence(km)
        
        lines.append(
            f"F{func_id} & {km.n_runs} & {km.n_hits} & {hit_pct} & "
            f"{mean_t} & {med_t} & {T_str} & {a_str} & {ctype} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:km-cec2017}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)


def generate_summary(results: Dict[int, KMResult]) -> str:
    """Generate summary statistics."""
    lines = [
        "=" * 60,
        "SUMMARY STATISTICS",
        "=" * 60,
        f"Functions analyzed: {len(results)}",
    ]
    
    # Count by classification
    types = {fid: classify_convergence(km) for fid, km in results.items()}
    type_counts = {}
    for t in types.values():
        type_counts[t] = type_counts.get(t, 0) + 1
    
    lines.append("\nConvergence classification:")
    for t, count in sorted(type_counts.items()):
        funcs = [f"F{fid}" for fid, tt in types.items() if tt == t]
        lines.append(f"  {t}: {count} functions ({', '.join(funcs[:5])}{'...' if len(funcs) > 5 else ''})")
    
    # a_valid statistics
    a_vals = [(fid, km.a_valid) for fid, km in results.items() if km.a_valid is not None]
    if a_vals:
        vals = [a for _, a in a_vals]
        lines.extend([
            "",
            f"Geometric envelope rate a_valid ({len(a_vals)} functions):",
            f"  Min:  {min(vals):.6f} (F{min(a_vals, key=lambda x: x[1])[0]})",
            f"  Max:  {max(vals):.6f} (F{max(a_vals, key=lambda x: x[1])[0]})",
            f"  Mean: {np.mean(vals):.6f}",
        ])
    
    # Hit rate statistics
    hit_rates = [km.hit_rate for km in results.values()]
    lines.extend([
        "",
        "Hit rates:",
        f"  100% hit rate: {sum(1 for h in hit_rates if h == 1.0)} functions",
        f"  >50% hit rate: {sum(1 for h in hit_rates if h > 0.5)} functions",
        f"  0% hit rate:   {sum(1 for h in hit_rates if h == 0)} functions",
    ])
    
    return "\n".join(lines)


def plot_survival_curves(results: Dict[int, KMResult], output_path: Path):
    """Generate survival curve plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    # Select functions with sufficient hits for interesting plots
    selected = [(fid, km) for fid, km in sorted(results.items()) if km.n_hits >= 10]
    
    if not selected:
        print("No functions with >= 10 hits for plotting")
        return
    
    n_plots = min(9, len(selected))
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, (func_id, km) in enumerate(selected[:n_plots]):
        ax = axes[idx]
        
        # Extend to start at t=0 with S=1
        times_ext = np.concatenate([[0], km.times])
        surv_ext = np.concatenate([[1.0], km.survival])
        
        # KM step function
        ax.step(times_ext, surv_ext, where='post', linewidth=2, 
                color='blue', label='KM estimate')
        
        # Geometric envelope
        if km.a_valid is not None and km.T_burnin is not None:
            T = km.T_burnin
            # Find S(T)
            idx_T = np.searchsorted(km.times, T)
            if idx_T > 0:
                S_T = km.survival[idx_T - 1]
            else:
                S_T = 1.0
            
            t_env = np.arange(T, int(km.times[-1]) + 1)
            S_env = S_T * (1 - km.a_valid) ** (t_env - T + 1)
            ax.plot(t_env, S_env, 'r--', linewidth=1.5,
                   label=f'Envelope (a={km.a_valid:.4f})')
            ax.axvline(T, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('P(τ > t)')
        ax.set_title(f'F{func_id} (n={km.n_runs}, hits={km.n_hits})')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KM analysis for CEC2017 L-SHADE results")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--output-dir', type=str, default='results/km_analysis')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the pkl file
    pkl_files = list(data_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"ERROR: No pkl files in {data_dir}")
        return
    
    pkl_path = pkl_files[0]
    print(f"Loading: {pkl_path}")
    print(f"Epsilon: {args.epsilon}")
    
    # Load data
    hit_data = load_cec2017_data(pkl_path, epsilon=args.epsilon)
    print(f"Loaded {len(hit_data)} functions")
    
    # Get max generations from data
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    max_gen = max(run['nit'] for runs in raw_data.values() for run in runs)
    print(f"Max generations: {max_gen}")
    
    # Run KM analysis
    results = {}
    for func_id, hit_times in sorted(hit_data.items()):
        km = kaplan_meier(hit_times, max_gen)
        km.func_id = func_id
        results[func_id] = km
        
        n_hits = sum(1 for t in hit_times if t is not None)
        print(f"  F{func_id}: {len(hit_times)} runs, {n_hits} hits ({100*n_hits/len(hit_times):.0f}%)")
    
    # Print detailed tables for selected functions
    if args.verbose:
        for func_id in [1, 4, 10, 20]:
            if func_id in results:
                print_km_table(results[func_id], func_id)
    
    # Generate outputs
    print("\n" + "=" * 60)
    summary = generate_summary(results)
    print(summary)
    
    # Save summary
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary)
    
    # LaTeX table
    latex = generate_latex_table(results)
    latex_path = output_dir / "km_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"\nSaved LaTeX table to {latex_path}")
    
    # Print LaTeX
    print("\n" + "=" * 60)
    print("LATEX TABLE")
    print("=" * 60)
    print(latex)
    
    # Save results
    results_path = output_dir / "km_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_path}")
    
    # Plot
    if args.plot:
        plot_path = output_dir / "survival_curves.png"
        plot_survival_curves(results, plot_path)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
