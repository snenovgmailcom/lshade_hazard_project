"""
Lemma 1 & 2 verification across all functions with sufficient data.

Lemma 1: P(τ > n) = E[∏ 1_{E_t^c}]  (identity)
Lemma 2: P(τ > n) ≤ E[∏ (1-p_t)]    (upper bound)

Key insight: For empirical data, these are equal (Kaplan-Meier).
The interesting test is comparing ACROSS functions and checking
if the theoretical structure (two-phase, etc.) is consistent.
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


def empirical_survival(hitting_times, n_max=None):
    finite_times = hitting_times[np.isfinite(hitting_times)]
    if len(finite_times) == 0:
        return np.array([0]), np.array([1.0])
    if n_max is None:
        n_max = int(np.max(finite_times)) + 10
    n_values = np.arange(n_max + 1)
    n_runs = len(hitting_times)
    survival = np.array([np.sum(hitting_times > n) / n_runs for n in n_values])
    return n_values, survival


def empirical_hazard(hitting_times, n_max=None):
    finite_times = hitting_times[np.isfinite(hitting_times)]
    if len(finite_times) == 0:
        return np.array([1]), np.array([0.0])
    if n_max is None:
        n_max = int(np.max(finite_times)) + 10
    t_values = np.arange(1, n_max + 1)
    hazard = []
    at_risk_list = []
    for t in t_values:
        at_risk = np.sum(hitting_times >= t)
        hits_at_t = np.sum(hitting_times == t)
        hazard.append(hits_at_t / at_risk if at_risk > 0 else 0.0)
        at_risk_list.append(at_risk)
    return t_values, np.array(hazard), np.array(at_risk_list)


def analyze_function(tau, func_name):
    """Detailed analysis for one function."""
    finite_tau = tau[np.isfinite(tau)]
    n_runs = len(tau)
    n_hits = len(finite_tau)
    
    if n_hits < 5:
        return None
    
    # Basic stats
    T = int(np.min(finite_tau))  # First hit generation
    tau_max = int(np.max(finite_tau))
    
    n_max = tau_max + 20
    n_vals, survival = empirical_survival(tau, n_max)
    t_vals, hazard, at_risk = empirical_hazard(tau, n_max)
    
    # Product of (1-p_t) - this equals survival by Kaplan-Meier
    product = np.cumprod(1 - hazard)
    product = np.insert(product, 0, 1.0)
    
    # Key generations to report
    key_gens = [0, T-1, T, T+5, T+10, T+20, tau_max]
    key_gens = [g for g in key_gens if 0 <= g < len(survival)]
    
    return {
        'func_name': func_name,
        'n_runs': n_runs,
        'n_hits': n_hits,
        'hit_rate': n_hits / n_runs,
        'T_first': T,
        'tau_max': tau_max,
        'tau_mean': np.mean(finite_tau),
        'tau_std': np.std(finite_tau),
        'n_vals': n_vals,
        'survival': survival,
        'hazard': hazard,
        'at_risk': at_risk,
        'product': product,
        'key_gens': key_gens
    }


def main():
    pkl_path = '/home/svety/lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl'
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    data = load_data(pkl_path)
    epsilon = 1e-2  # Use 1e-2 to get more hits
    
    print("="*80)
    print("LEMMA 1 & 2 VERIFICATION ACROSS CEC2017 FUNCTIONS")
    print("="*80)
    print(f"\nEpsilon = {epsilon}")
    print("\nLemma 1: P(τ > n) = E[∏ 1_{E_t^c}] = ∏ P(E_t^c | H_{t-1})")
    print("Lemma 2: P(τ > n) ≤ E[∏ (1-p_t)]")
    print("\nNote: For empirical data, Lemma 2 becomes equality (Kaplan-Meier)")
    
    # Collect results for all functions with sufficient data
    results = []
    
    for i in range(1, 31):
        func_name = f'cec2017_f{i}'
        f_star = CEC2017_OPTIMA[func_name]
        curves = [run['curve'] for run in data[func_name]]
        
        tau = compute_hitting_times(curves, f_star, epsilon)
        result = analyze_function(tau, func_name)
        
        if result:
            results.append(result)
    
    # Print summary table
    print("\n" + "-"*80)
    print(f"{'Function':<12} {'Hits':>5} {'Rate':>7} {'T_first':>8} {'τ_max':>8} "
          f"{'E[τ]':>8} {'σ[τ]':>8}")
    print("-"*80)
    
    for r in results:
        print(f"{r['func_name']:<12} {r['n_hits']:>5} {r['hit_rate']:>7.1%} "
              f"{r['T_first']:>8} {r['tau_max']:>8} "
              f"{r['tau_mean']:>8.1f} {r['tau_std']:>8.1f}")
    
    # Detailed output for each function
    print("\n" + "="*80)
    print("DETAILED SURVIVAL TABLES")
    print("="*80)
    
    for r in results:
        print(f"\n--- {r['func_name']} (hits: {r['n_hits']}/{r['n_runs']}) ---")
        print(f"{'Gen n':>8} {'P(τ>n)':>10} {'∏(1-p_t)':>10} {'p_n':>10} {'At risk':>10}")
        print("-"*52)
        
        for n in r['key_gens']:
            surv = r['survival'][n]
            prod = r['product'][n] if n < len(r['product']) else 0
            p_n = r['hazard'][n-1] if 0 < n <= len(r['hazard']) else 0
            at_risk = r['at_risk'][n-1] if 0 < n <= len(r['at_risk']) else r['n_runs']
            
            print(f"{n:>8} {surv:>10.4f} {prod:>10.4f} {p_n:>10.4f} {at_risk:>10}")
    
    # Create multi-panel figure
    n_funcs = len(results)
    n_cols = 3
    n_rows = (n_funcs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, r in enumerate(results):
        ax = axes[idx]
        
        # Plot survival curve
        ax.plot(r['n_vals'], r['survival'], 'b-', linewidth=2, label='P(τ > n)')
        
        # Mark key points
        ax.axvline(r['T_first'], color='red', linestyle='--', alpha=0.5, 
                   label=f"T={r['T_first']}")
        
        ax.set_xlabel('Generation n')
        ax.set_ylabel('P(τ > n)')
        ax.set_title(f"{r['func_name']}: {r['n_hits']}/{r['n_runs']} hits\n"
                     f"E[τ]={r['tau_mean']:.0f}, σ={r['tau_std']:.0f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lemma1_all_functions.png', dpi=150)
    print(f"\nSaved: results/lemma1_all_functions.png")
    plt.close()
    
    # Create hazard rate comparison figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, r in enumerate(results):
        ax = axes[idx]
        
        t_vals = np.arange(1, len(r['hazard']) + 1)
        
        # Plot hazard rate
        ax.plot(t_vals, r['hazard'], 'b.', markersize=2, alpha=0.5)
        
        # Smoothed hazard (moving average)
        window = min(20, len(r['hazard']) // 5)
        if window > 1:
            hazard_smooth = np.convolve(r['hazard'], np.ones(window)/window, mode='valid')
            t_smooth = t_vals[window//2:window//2+len(hazard_smooth)]
            ax.plot(t_smooth, hazard_smooth, 'r-', linewidth=2, label='Smoothed')
        
        # Mark T_first
        ax.axvline(r['T_first'], color='green', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Generation t')
        ax.set_ylabel('p_t = P(τ=t|τ≥t)')
        ax.set_title(f"{r['func_name']}: Hazard Rate")
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hazard_all_functions.png', dpi=150)
    print(f"Saved: results/hazard_all_functions.png")
    plt.close()


if __name__ == '__main__':
    main()
