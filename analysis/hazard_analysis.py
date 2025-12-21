"""
Hazard analysis for L-SHADE on CEC2017.
Computes first-hitting times, survival curves, and empirical hazard rates.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CEC2017 optima: f_i has optimum at 100*i
CEC2017_OPTIMA = {f'cec2017_f{i}': 100.0 * i for i in range(1, 31)}


def load_data(pkl_path):
    """Load PKL file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def compute_hitting_times(curves, f_star, epsilon):
    """
    Compute first-hitting times to epsilon-sublevel set.
    
    Args:
        curves: list of convergence curves (best fitness per generation)
        f_star: known optimum value
        epsilon: tolerance (hit when curve[t] <= f_star + epsilon)
    
    Returns:
        hitting_times: array of first-hitting times (np.inf if not hit)
    """
    hitting_times = []
    target = f_star + epsilon
    
    for curve in curves:
        hits = np.where(curve <= target)[0]
        if len(hits) > 0:
            hitting_times.append(hits[0])
        else:
            hitting_times.append(np.inf)
    
    return np.array(hitting_times)


def empirical_survival(hitting_times, n_max=None):
    """
    Compute empirical survival function P(tau > n).
    
    Returns:
        n_values: array [0, 1, 2, ..., n_max]
        survival: P(tau > n) for each n
    """
    finite_times = hitting_times[np.isfinite(hitting_times)]
    
    if n_max is None:
        n_max = int(np.max(finite_times)) if len(finite_times) > 0 else 1000
    
    n_values = np.arange(n_max + 1)
    n_runs = len(hitting_times)
    
    survival = np.array([np.sum(hitting_times > n) / n_runs for n in n_values])
    
    return n_values, survival


def empirical_hazard(hitting_times, n_max=None):
    """
    Compute empirical hazard rate p_t = P(tau = t | tau >= t).
    
    Returns:
        t_values: array [1, 2, ..., n_max]
        hazard: estimated p_t for each t
    """
    finite_times = hitting_times[np.isfinite(hitting_times)]
    
    if n_max is None:
        n_max = int(np.max(finite_times)) if len(finite_times) > 0 else 1000
    
    t_values = np.arange(1, n_max + 1)
    hazard = []
    
    for t in t_values:
        at_risk = np.sum(hitting_times >= t)  # runs that haven't hit yet
        hits_at_t = np.sum(hitting_times == t)  # runs that hit exactly at t
        
        if at_risk > 0:
            hazard.append(hits_at_t / at_risk)
        else:
            hazard.append(np.nan)
    
    return t_values, np.array(hazard)


def analyze_function(func_data, func_name, epsilons=[1e-2, 1e-4, 1e-6, 1e-8]):
    """
    Full hazard analysis for one function.
    """
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run['curve'] for run in func_data]
    n_runs = len(curves)
    
    print(f"\n{'='*60}")
    print(f"{func_name} (f* = {f_star}, {n_runs} runs)")
    print(f"{'='*60}")
    
    results = {}
    
    for eps in epsilons:
        tau = compute_hitting_times(curves, f_star, eps)
        n_hit = np.sum(np.isfinite(tau))
        
        print(f"\nε = {eps:.0e}:")
        print(f"  Hits: {n_hit}/{n_runs} ({100*n_hit/n_runs:.1f}%)")
        
        if n_hit > 0:
            finite_tau = tau[np.isfinite(tau)]
            print(f"  τ: min={np.min(finite_tau):.0f}, "
                  f"median={np.median(finite_tau):.0f}, "
                  f"max={np.max(finite_tau):.0f}")
        
        results[eps] = {
            'hitting_times': tau,
            'n_hits': n_hit,
            'hit_rate': n_hit / n_runs
        }
    
    return results


def plot_survival_curves(func_data, func_name, epsilons=[1e-2, 1e-4, 1e-6, 1e-8], 
                         save_path=None):
    """
    Plot empirical survival curves for different epsilon values.
    """
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run['curve'] for run in func_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for eps in epsilons:
        tau = compute_hitting_times(curves, f_star, eps)
        n_vals, surv = empirical_survival(tau)
        
        # Only plot if there are any hits
        if np.any(np.isfinite(tau)):
            ax.semilogy(n_vals, surv + 1e-10, label=f'ε = {eps:.0e}')
    
    ax.set_xlabel('Generation n')
    ax.set_ylabel('P(τ > n)')
    ax.set_title(f'Survival Curves: {func_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


def plot_hazard_rates(func_data, func_name, epsilon=1e-4, window=50, 
                      save_path=None):
    """
    Plot empirical hazard rates (smoothed).
    """
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run['curve'] for run in func_data]
    tau = compute_hitting_times(curves, f_star, epsilon)
    
    if not np.any(np.isfinite(tau)):
        print(f"No hits for {func_name} at ε={epsilon}")
        return None
    
    t_vals, hazard = empirical_hazard(tau)
    
    # Smooth with moving average
    hazard_smooth = np.convolve(hazard, np.ones(window)/window, mode='valid')
    t_smooth = t_vals[:len(hazard_smooth)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t_vals, hazard, alpha=0.3, label='Raw')
    ax.plot(t_smooth, hazard_smooth, label=f'Smoothed (window={window})')
    
    ax.set_xlabel('Generation t')
    ax.set_ylabel('p_t = P(τ = t | τ ≥ t)')
    ax.set_title(f'Empirical Hazard Rate: {func_name} (ε = {epsilon:.0e})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    return fig


if __name__ == '__main__':
    import sys
    
    # Default path
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else \
        '../lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl'
    
    print(f"Loading: {pkl_path}")
    data = load_data(pkl_path)
    
    # Analyze a few functions
    for func_name in ['cec2017_f1', 'cec2017_f5', 'cec2017_f10', 'cec2017_f15']:
        if func_name in data:
            results = analyze_function(data[func_name], func_name)
