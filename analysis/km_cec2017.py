#!/usr/bin/env python3
"""
Kaplan-Meier Survival Analysis for L-SHADE CEC2017 Results
===========================================================
FIXED: Burn-in now based on first-hit time, not max generations.

Usage:
    python km_cec2017_v2.py --data-dir experiments/r_lshade_D10_nfev_100000 --epsilon 0.01
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
    hit_times: List[Optional[int]]
    times: np.ndarray
    n_risk: np.ndarray
    n_events: np.ndarray
    survival: np.ndarray
    hazard: np.ndarray
    mean_time: Optional[float]
    median_time: Optional[float]
    min_time: Optional[int]
    max_time: Optional[int]
    a_valid: Optional[float]
    T_burnin: Optional[int]
    avg_hazard: Optional[float]


# =============================================================================
# Data Loading
# =============================================================================

def load_cec2017_data(pkl_path: Path, epsilon: float = 0.01) -> Dict[int, List[Optional[int]]]:
    """Load CEC2017 results and extract first-hitting times."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    results = {}
    
    for key, runs in data.items():
        match = re.search(r'f(\d+)', key)
        if not match:
            continue
        func_id = int(match.group(1))
        
        f_star = 100.0 * func_id
        target = f_star + epsilon
        
        hit_times = []
        for run in runs:
            curve = run['curve']
            hits = np.where(curve <= target)[0]
            hit_time = int(hits[0]) if len(hits) > 0 else None
            hit_times.append(hit_time)
            
        results[func_id] = hit_times
        
    return results


# =============================================================================
# Kaplan-Meier Estimation
# =============================================================================

def kaplan_meier(hit_times: List[Optional[int]], max_gen: int) -> KMResult:
    """Compute Kaplan-Meier survival estimate."""
    n_runs = len(hit_times)
    n_hits = sum(1 for t in hit_times if t is not None)
    
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
            min_time=None, max_time=None,
            a_valid=None, T_burnin=None, avg_hazard=None
        )
    
    # Convert to arrays
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
    
    # Unique event times
    event_times = np.unique(obs_times[events == 1])
    
    # Compute KM
    times_list = []
    n_risk_list = []
    n_events_list = []
    hazard_list = []
    survival_list = []
    
    S = 1.0
    for t in event_times:
        Y = np.sum(obs_times >= t)
        d = np.sum((obs_times == t) & (events == 1))
        h = d / Y
        S = S * (1 - h)
        
        times_list.append(t)
        n_risk_list.append(Y)
        n_events_list.append(d)
        hazard_list.append(h)
        survival_list.append(S)
    
    times = np.array(times_list)
    n_risk = np.array(n_risk_list)
    n_events = np.array(n_events_list)
    hazard = np.array(hazard_list)
    survival = np.array(survival_list)
    
    # Statistics
    actual_hits = [t for t in hit_times if t is not None]
    mean_time = float(np.mean(actual_hits))
    median_time = float(np.median(actual_hits))
    min_time = int(min(actual_hits))
    max_time_hit = int(max(actual_hits))
    avg_hazard = float(np.mean(hazard))
    
    # Compute geometric envelope - FIXED burn-in
    a_valid, T_burnin = compute_geometric_envelope(times, survival, n_risk, min_time)
    
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
        min_time=min_time,
        max_time=max_time_hit,
        a_valid=a_valid,
        T_burnin=T_burnin,
        avg_hazard=avg_hazard
    )


def compute_geometric_envelope(times: np.ndarray, survival: np.ndarray, 
                                n_risk: np.ndarray, 
                                first_hit: int,
                                min_risk: int = 5) -> Tuple[Optional[float], Optional[int]]:
    """
    Compute tightest valid geometric envelope rate.
    
    FIXED: Burn-in is now based on first hit time, not max generations.
    T_burnin = first_hit (start of active phase)
    """
    if len(times) == 0:
        return None, None
    
    # Use first hit time as burn-in point
    T = first_hit
    
    # Find the index for T in times array
    start_idx = np.searchsorted(times, T)
    if start_idx >= len(times):
        start_idx = 0  # Fallback to first event
    
    # Check minimum risk
    if n_risk[start_idx] < min_risk:
        # Find first point with sufficient risk
        valid_idx = np.where(n_risk >= min_risk)[0]
        if len(valid_idx) == 0:
            return None, None
        start_idx = valid_idx[0]
        T = int(times[start_idx])
    
    # Get S(T) - survival just before T
    if start_idx == 0:
        S_T = 1.0
    else:
        S_T = survival[start_idx - 1]
    
    if S_T <= 0:
        return None, T
    
    # Find maximum root: max over n >= T of (S(n)/S(T))^{1/(n-T+1)}
    max_root = 0.0
    for i in range(start_idx, len(times)):
        if survival[i] > 0 and survival[i] < S_T:
            n = times[i]
            exponent = n - T + 1
            if exponent > 0:
                ratio = survival[i] / S_T
                root = ratio ** (1.0 / exponent)
                max_root = max(max_root, root)
        elif survival[i] == 0:
            # Handle S=0 case: need to find largest valid root before this
            pass
    
    # If survival reaches 0, we need special handling
    if survival[-1] == 0:
        # Find the last non-zero survival point
        last_nonzero = np.where(survival > 0)[0]
        if len(last_nonzero) > 0:
            last_idx = last_nonzero[-1]
            n = times[last_idx]
            exponent = n - T + 1
            if exponent > 0 and survival[last_idx] < S_T:
                ratio = survival[last_idx] / S_T
                root = ratio ** (1.0 / exponent)
                max_root = max(max_root, root)
    
    if max_root == 0 or max_root >= 1.0:
        # Can't find valid envelope - try simpler approach
        # Use the empirical hazard as a guide
        if len(times) >= 2:
            # Average hazard from T onwards
            mask = times >= T
            if np.sum(mask) > 0:
                hazards = n_events[mask] / n_risk[mask]
                avg_h = np.mean(hazards)
                if 0 < avg_h < 1:
                    return avg_h, T
        return None, T
    
    a_valid = 1.0 - max_root
    return a_valid, T


def classify_convergence(km: KMResult) -> str:
    """Classify convergence behavior."""
    if km.n_hits == 0:
        return "no_hits"
    if km.n_hits < 5:
        return "few_hits"
    if km.a_valid is None or km.avg_hazard is None:
        return "moderate"
    if km.avg_hazard <= 0:
        return "moderate"
    
    ratio = km.a_valid / km.avg_hazard
    
    if ratio < 0.20:
        return "clustered"
    elif ratio > 0.60:
        return "geometric"
    else:
        return "moderate"


# =============================================================================
# Output
# =============================================================================

def print_km_table(km: KMResult, func_id: int):
    """Print detailed KM table."""
    print(f"\n{'='*70}")
    print(f"Function F{func_id}")
    print(f"{'='*70}")
    print(f"Runs: {km.n_runs}, Hits: {km.n_hits}, Hit rate: {100*km.hit_rate:.1f}%")
    
    if km.n_hits > 0:
        print(f"Hitting times: mean={km.mean_time:.1f}, median={km.median_time:.1f}, "
              f"range=[{km.min_time}, {km.max_time}]")
        print(f"Average empirical hazard: {km.avg_hazard:.6f}")
    
    if km.a_valid is not None:
        print(f"\nGeometric envelope:")
        print(f"  T (burn-in) = {km.T_burnin}")
        print(f"  a_valid = {km.a_valid:.6f}")
        if km.avg_hazard and km.avg_hazard > 0:
            print(f"  ratio a_valid/avg_hazard = {km.a_valid/km.avg_hazard:.2%}")
        print(f"  Classification: {classify_convergence(km)}")
        print(f"  Tail bound: P(τ > n) ≤ (1 - {km.a_valid:.6f})^(n - {km.T_burnin} + 1)")
    
    # Print survival table
    print(f"\nKaplan-Meier table (first 15 event times):")
    print(f"{'t':>8} {'Y_t':>8} {'d_t':>6} {'h_t':>10} {'S(t)':>10}")
    print("-" * 50)
    
    for i in range(min(15, len(km.times))):
        print(f"{km.times[i]:8.0f} {km.n_risk[i]:8.0f} {km.n_events[i]:6.0f} "
              f"{km.hazard[i]:10.6f} {km.survival[i]:10.6f}")
    
    if len(km.times) > 15:
        print(f"... ({len(km.times) - 15} more rows)")


def generate_latex_table(results: Dict[int, KMResult]) -> str:
    """Generate LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Kaplan--Meier survival analysis for L-SHADE on CEC2017 (D=10, $\varepsilon=0.01$).}",
        r"\small",
        r"\begin{tabular}{@{}rrrrrrrrrl@{}}",
        r"\toprule",
        r"$f$ & Runs & Hits & Hit\% & $\bar{\tau}$ & $\tilde{\tau}$ & $T$ & $a_{\mathrm{valid}}$ & $\bar{h}$ & Type \\",
        r"\midrule"
    ]
    
    for func_id in sorted(results.keys()):
        km = results[func_id]
        
        hit_pct = f"{100*km.hit_rate:.1f}"
        mean_t = f"{km.mean_time:.0f}" if km.mean_time else "--"
        med_t = f"{km.median_time:.0f}" if km.median_time else "--"
        T_str = f"{km.T_burnin}" if km.T_burnin else "--"
        a_str = f"{km.a_valid:.4f}" if km.a_valid else "--"
        h_str = f"{km.avg_hazard:.4f}" if km.avg_hazard else "--"
        ctype = classify_convergence(km)
        
        lines.append(
            f"F{func_id} & {km.n_runs} & {km.n_hits} & {hit_pct} & "
            f"{mean_t} & {med_t} & {T_str} & {a_str} & {h_str} & {ctype} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:km-cec2017}",
        r"\end{table}"
    ])
    
    return "\n".join(lines)


def generate_summary(results: Dict[int, KMResult]) -> str:
    """Generate summary."""
    lines = [
        "=" * 60,
        "SUMMARY STATISTICS",
        "=" * 60,
    ]
    
    # Classification
    types = {fid: classify_convergence(km) for fid, km in results.items()}
    type_counts = {}
    for t in types.values():
        type_counts[t] = type_counts.get(t, 0) + 1
    
    lines.append("\nConvergence classification:")
    for t in ['clustered', 'moderate', 'geometric', 'few_hits', 'no_hits']:
        if t in type_counts:
            funcs = [f"F{fid}" for fid, tt in types.items() if tt == t]
            lines.append(f"  {t:12s}: {type_counts[t]:2d} functions ({', '.join(funcs)})")
    
    # a_valid statistics
    a_vals = [(fid, km.a_valid) for fid, km in results.items() if km.a_valid is not None]
    if a_vals:
        vals = [a for _, a in a_vals]
        lines.extend([
            "",
            f"Geometric envelope a_valid ({len(a_vals)} functions):",
            f"  Range: [{min(vals):.6f}, {max(vals):.6f}]",
            f"  Mean:  {np.mean(vals):.6f}",
        ])
    
    # Hit statistics
    hit_funcs = [(fid, km) for fid, km in results.items() if km.n_hits > 0]
    lines.extend([
        "",
        f"Functions with hits: {len(hit_funcs)} / {len(results)}",
        f"Functions with 100% hit rate: {sum(1 for _, km in hit_funcs if km.hit_rate == 1.0)}",
    ])
    
    return "\n".join(lines)


def plot_survival_curves(results: Dict[int, KMResult], output_path: Path):
    """Generate survival plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    selected = [(fid, km) for fid, km in sorted(results.items()) if km.n_hits >= 5]
    
    if not selected:
        print("No functions with >= 5 hits for plotting")
        return
    
    n_plots = min(9, len(selected))
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (func_id, km) in enumerate(selected[:n_plots]):
        ax = axes[idx]
        
        # Extend to t=0
        times_ext = np.concatenate([[0], km.times])
        surv_ext = np.concatenate([[1.0], km.survival])
        
        ax.step(times_ext, surv_ext, where='post', linewidth=2, 
                color='blue', label='KM estimate')
        
        # Geometric envelope
        if km.a_valid is not None and km.T_burnin is not None:
            T = km.T_burnin
            t_max = int(km.times[-1])
            
            # Find S(T)
            idx_T = np.searchsorted(km.times, T)
            S_T = km.survival[idx_T-1] if idx_T > 0 else 1.0
            
            t_env = np.arange(T, t_max + 1, max(1, (t_max-T)//100))
            S_env = S_T * (1 - km.a_valid) ** (t_env - T + 1)
            
            ax.plot(t_env, S_env, 'r--', linewidth=1.5,
                   label=f'Geom (a={km.a_valid:.4f})')
            ax.axvline(T, color='gray', linestyle=':', alpha=0.5, label=f'T={T}')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('P(τ > t)')
        ax.set_title(f'F{func_id}: {km.n_hits}/{km.n_runs} hits, {classify_convergence(km)}')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--output-dir', type=str, default='results/km_analysis')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pkl_files = list(data_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"ERROR: No pkl files in {data_dir}")
        return
    
    pkl_path = pkl_files[0]
    print(f"Loading: {pkl_path}")
    print(f"Epsilon: {args.epsilon}")
    
    # Load
    hit_data = load_cec2017_data(pkl_path, epsilon=args.epsilon)
    print(f"Loaded {len(hit_data)} functions")
    
    # Get max gen
    with open(pkl_path, 'rb') as f:
        raw_data = pickle.load(f)
    max_gen = max(run['nit'] for runs in raw_data.values() for run in runs)
    print(f"Max generations: {max_gen}")
    
    # Analyze
    results = {}
    for func_id, hit_times in sorted(hit_data.items()):
        km = kaplan_meier(hit_times, max_gen)
        km.func_id = func_id
        results[func_id] = km
        
        n_hits = km.n_hits
        status = f"{n_hits} hits ({100*km.hit_rate:.0f}%)"
        if km.a_valid is not None:
            status += f", a={km.a_valid:.4f}"
        print(f"  F{func_id}: {status}")
    
    # Verbose output
    if args.verbose:
        for func_id, km in sorted(results.items()):
            if km.n_hits >= 5:
                print_km_table(km, func_id)
    
    # Summary
    print("\n" + generate_summary(results))
    
    # LaTeX
    latex = generate_latex_table(results)
    latex_path = output_dir / "km_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"\nSaved: {latex_path}")
    
    print("\n" + "="*60 + "\nLATEX TABLE\n" + "="*60)
    print(latex)
    
    # Save
    with open(output_dir / "km_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(generate_summary(results))
    
    # Plot
    if args.plot:
        plot_survival_curves(results, output_dir / "survival_curves.png")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
    
