"""
Generate survival and hazard plots for paper.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CEC2017_OPTIMA = {f'cec2017_f{i}': 100.0 * i for i in range(1, 31)}


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def compute_hitting_times(curves, f_star, epsilon):
    hitting_times = []
    target = f_star + epsilon
    for curve in curves:
        hits = np.where(curve <= target)[0]
        hitting_times.append(hits[0] if len(hits) > 0 else np.inf)
    return np.array(hitting_times)


def empirical_survival(hitting_times):
    finite_times = hitting_times[np.isfinite(hitting_times)]
    if len(finite_times) == 0:
        return np.array([0]), np.array([1.0])
    
    n_max = int(np.max(finite_times)) + 10
    n_values = np.arange(n_max + 1)
    n_runs = len(hitting_times)
    survival = np.array([np.sum(hitting_times > n) / n_runs for n in n_values])
    return n_values, survival


def empirical_hazard(hitting_times):
    finite_times = hitting_times[np.isfinite(hitting_times)]
    if len(finite_times) == 0:
        return np.array([1]), np.array([0.0])
    
    n_max = int(np.max(finite_times)) + 10
    t_values = np.arange(1, n_max + 1)
    hazard = []
    
    for t in t_values:
        at_risk = np.sum(hitting_times >= t)
        hits_at_t = np.sum(hitting_times == t)
        hazard.append(hits_at_t / at_risk if at_risk > 0 else np.nan)
    
    return t_values, np.array(hazard)


def fit_exponential(n_vals, survival):
    """Fit P(tau > n) = (1-p)^n, return p."""
    # Use log-linear regression on positive survival values
    mask = survival > 0
    if np.sum(mask) < 2:
        return None
    
    log_surv = np.log(survival[mask])
    n_fit = n_vals[mask]
    
    # Linear fit: log(S) = n * log(1-p)
    slope, _ = np.polyfit(n_fit, log_surv, 1)
    p = 1 - np.exp(slope)
    return p


def main():
    import sys
    
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/svety/lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl'
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading: {pkl_path}")
    data = load_data(pkl_path)
    
    # ===== F1: Full analysis =====
    func_name = 'cec2017_f1'
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run['curve'] for run in data[func_name]]
    
    print(f"\n=== {func_name} Analysis ===")
    
    # Plot 1: Survival curves for different epsilon
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epsilons = [1e-2, 1e-4, 1e-6, 1e-8]
    colors = ['blue', 'green', 'orange', 'red']
    
    for eps, color in zip(epsilons, colors):
        tau = compute_hitting_times(curves, f_star, eps)
        n_vals, surv = empirical_survival(tau)
        
        # Left: linear scale
        axes[0].plot(n_vals, surv, color=color, label=f'ε = {eps:.0e}')
        
        # Right: log scale
        axes[1].semilogy(n_vals, surv + 1e-10, color=color, label=f'ε = {eps:.0e}')
        
        # Fit exponential
        p_fit = fit_exponential(n_vals, surv)
        if p_fit and p_fit > 0:
            print(f"  ε = {eps:.0e}: fitted p = {p_fit:.4f}, E[τ] ≈ {1/p_fit:.1f}")
            # Plot theoretical curve
            surv_theory = (1 - p_fit) ** n_vals
            axes[1].semilogy(n_vals, surv_theory, '--', color=color, alpha=0.5)
    
    axes[0].set_xlabel('Generation n')
    axes[0].set_ylabel('P(τ > n)')
    axes[0].set_title(f'{func_name}: Survival Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Generation n')
    axes[1].set_ylabel('P(τ > n) [log scale]')
    axes[1].set_title(f'{func_name}: Survival Curves (log scale)\nDashed = exponential fit')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{func_name}_survival.png', dpi=150)
    print(f"  Saved: {output_dir / f'{func_name}_survival.png'}")
    plt.close()
    
    # Plot 2: Hazard rate for epsilon = 1e-8
    eps = 1e-8
    tau = compute_hitting_times(curves, f_star, eps)
    t_vals, hazard = empirical_hazard(tau)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Raw hazard
    ax.plot(t_vals, hazard, 'b.', alpha=0.5, markersize=3, label='Empirical p_t')
    
    # Smoothed hazard (moving average)
    window = 20
    hazard_clean = np.nan_to_num(hazard, nan=0.0)
    hazard_smooth = np.convolve(hazard_clean, np.ones(window)/window, mode='valid')
    t_smooth = t_vals[window//2 : window//2 + len(hazard_smooth)]
    ax.plot(t_smooth, hazard_smooth, 'r-', linewidth=2, label=f'Smoothed (window={window})')
    
    # Mean hazard line
    mean_hazard = np.nanmean(hazard[hazard > 0])
    ax.axhline(mean_hazard, color='green', linestyle='--', label=f'Mean p = {mean_hazard:.3f}')
    
    ax.set_xlabel('Generation t')
    ax.set_ylabel('p_t = P(τ = t | τ ≥ t)')
    ax.set_title(f'{func_name}: Empirical Hazard Rate (ε = {eps:.0e})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{func_name}_hazard.png', dpi=150)
    print(f"  Saved: {output_dir / f'{func_name}_hazard.png'}")
    plt.close()
    
    # ===== Summary table for all functions =====
    print("\n=== Summary: Hit Rates at ε = 1e-2 ===")
    print(f"{'Function':<15} {'Hits':>6} {'Rate':>8} {'Min τ':>8} {'Med τ':>8} {'Max τ':>8}")
    print("-" * 60)
    
    for i in range(1, 31):
        func_name = f'cec2017_f{i}'
        f_star = CEC2017_OPTIMA[func_name]
        curves = [run['curve'] for run in data[func_name]]
        
        tau = compute_hitting_times(curves, f_star, epsilon=1e-2)
        n_hits = np.sum(np.isfinite(tau))
        rate = n_hits / len(tau)
        
        if n_hits > 0:
            finite_tau = tau[np.isfinite(tau)]
            print(f"{func_name:<15} {n_hits:>6} {rate:>8.1%} {np.min(finite_tau):>8.0f} "
                  f"{np.median(finite_tau):>8.0f} {np.max(finite_tau):>8.0f}")
        else:
            print(f"{func_name:<15} {n_hits:>6} {rate:>8.1%} {'--':>8} {'--':>8} {'--':>8}")


if __name__ == '__main__':
    main()
