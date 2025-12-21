"""
Two-phase hazard model analysis.

Model: p_t = 0 for t < T (warm-up), p_t ~ p for t >= T (active phase)

This matches the theoretical framework if we shift time: tau' = tau - T
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


def fit_two_phase_model(hitting_times):
    """
    Fit two-phase model:
    - Phase 1: t < T, no hits (p_t = 0)
    - Phase 2: t >= T, constant hazard p
    
    After shifting: tau' = tau - T should be approximately Geometric(p)
    """
    finite_tau = hitting_times[np.isfinite(hitting_times)]
    
    if len(finite_tau) < 5:
        return None
    
    # Estimate T as minimum hitting time (or slightly before)
    T = np.min(finite_tau) - 1
    
    # Shifted hitting times
    tau_shifted = finite_tau - T
    
    # For geometric distribution, E[tau'] = 1/p, so p = 1/E[tau']
    mean_shifted = np.mean(tau_shifted)
    p_estimated = 1.0 / mean_shifted if mean_shifted > 0 else 0
    
    # Also estimate from median: median = -1/log(1-p) * log(2)
    # => p = 1 - exp(-log(2)/median) = 1 - 0.5^(1/median)
    median_shifted = np.median(tau_shifted)
    p_from_median = 1 - 0.5 ** (1.0 / median_shifted) if median_shifted > 0 else 0
    
    # Variance check: for Geometric, Var = (1-p)/p^2
    var_shifted = np.var(tau_shifted)
    theoretical_var = (1 - p_estimated) / (p_estimated ** 2) if p_estimated > 0 else np.inf
    
    return {
        'T': T,
        'p_from_mean': p_estimated,
        'p_from_median': p_from_median,
        'mean_shifted': mean_shifted,
        'median_shifted': median_shifted,
        'var_shifted': var_shifted,
        'theoretical_var': theoretical_var,
        'var_ratio': var_shifted / theoretical_var if theoretical_var > 0 else np.inf
    }


def plot_shifted_survival(hitting_times, func_name, output_dir):
    """
    Plot survival curve of shifted times tau' = tau - T.
    Compare with geometric (exponential) model.
    """
    finite_tau = hitting_times[np.isfinite(hitting_times)]
    
    if len(finite_tau) < 5:
        return None
    
    T = np.min(finite_tau) - 1
    tau_shifted = finite_tau - T
    
    # Empirical survival of shifted times
    n_max = int(np.max(tau_shifted)) + 5
    n_vals = np.arange(n_max + 1)
    n_runs = len(finite_tau)
    survival = np.array([np.sum(tau_shifted > n) / n_runs for n in n_vals])
    
    # Fit geometric model
    p_est = 1.0 / np.mean(tau_shifted)
    survival_geometric = (1 - p_est) ** n_vals
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].plot(n_vals, survival, 'b-', linewidth=2, label="Empirical P(τ' > n)")
    axes[0].plot(n_vals, survival_geometric, 'r--', linewidth=2, 
                 label=f'Geometric (p={p_est:.3f})')
    axes[0].set_xlabel("Shifted generation n' = n - T")
    axes[0].set_ylabel("P(τ' > n')")
    axes[0].set_title(f"{func_name}: Shifted Survival (T={T:.0f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log scale
    axes[1].semilogy(n_vals, survival + 1e-10, 'b-', linewidth=2, label="Empirical")
    axes[1].semilogy(n_vals, survival_geometric + 1e-10, 'r--', linewidth=2, 
                     label=f'Geometric (p={p_est:.3f})')
    axes[1].set_xlabel("Shifted generation n' = n - T")
    axes[1].set_ylabel("P(τ' > n') [log scale]")
    axes[1].set_title(f"Log scale - Linear = good geometric fit")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{func_name}_shifted_survival.png', dpi=150)
    plt.close()
    
    return p_est


def main():
    import sys
    
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/svety/lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl'
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading: {pkl_path}")
    data = load_data(pkl_path)
    
    print("\n" + "="*80)
    print("TWO-PHASE HAZARD MODEL ANALYSIS")
    print("Model: p_t = 0 for t < T, then p_t ~ p (constant) for t >= T")
    print("="*80)
    
    print(f"\n{'Function':<12} {'Hits':>5} {'T':>8} {'p(mean)':>8} {'p(med)':>8} "
          f"{'E[τ-T]':>8} {'Var ratio':>10}")
    print("-" * 75)
    
    # Analyze functions with enough hits
    good_functions = []
    
    for i in range(1, 31):
        func_name = f'cec2017_f{i}'
        f_star = CEC2017_OPTIMA[func_name]
        curves = [run['curve'] for run in data[func_name]]
        
        tau = compute_hitting_times(curves, f_star, epsilon=1e-2)
        n_hits = np.sum(np.isfinite(tau))
        
        if n_hits >= 10:
            result = fit_two_phase_model(tau)
            if result:
                print(f"{func_name:<12} {n_hits:>5} {result['T']:>8.0f} "
                      f"{result['p_from_mean']:>8.4f} {result['p_from_median']:>8.4f} "
                      f"{result['mean_shifted']:>8.1f} {result['var_ratio']:>10.2f}")
                
                good_functions.append((func_name, tau, result))
        else:
            print(f"{func_name:<12} {n_hits:>5} {'--':>8} {'--':>8} {'--':>8} "
                  f"{'--':>8} {'--':>10}")
    
    # Plot shifted survival for best functions
    print("\n" + "="*80)
    print("GENERATING SHIFTED SURVIVAL PLOTS")
    print("="*80)
    
    for func_name, tau, result in good_functions:
        p_est = plot_shifted_survival(tau, func_name, output_dir)
        if p_est:
            print(f"Saved: results/{func_name}_shifted_survival.png")
    
    # Theoretical interpretation
    print("\n" + "="*80)
    print("THEORETICAL INTERPRETATION")
    print("="*80)
    
    print("""
The two-phase model suggests:

1. WARM-UP PHASE (t < T):
   - Algorithm converges toward optimum region
   - p_t ≈ 0 (hitting target is essentially impossible)
   - This phase depends on function landscape
   
2. ACTIVE PHASE (t >= T):  
   - Population is "close enough" for hits to occur
   - p_t ≈ p (approximately constant hazard)
   - Theorem 2 applies to shifted time τ' = τ - T
   
3. IMPLICATIONS FOR BOUNDS:
   - P(τ > n) = 1 for n < T
   - P(τ > n) ≤ (1-p)^(n-T) for n >= T
   - E[τ] = T + 1/p
   
4. VARIANCE RATIO interpretation:
   - Var_ratio ≈ 1: Good geometric fit (constant hazard)
   - Var_ratio < 1: Under-dispersed (hits more regular than geometric)
   - Var_ratio > 1: Over-dispersed (hits more clustered)
""")
    
    # Find best geometric fits (var_ratio close to 1)
    print("\nBest geometric fits (Var ratio ~ 1):")
    for func_name, tau, result in good_functions:
        if 0.5 < result['var_ratio'] < 2.0:
            print(f"  {func_name}: Var ratio = {result['var_ratio']:.2f}, "
                  f"p = {result['p_from_mean']:.4f}, T = {result['T']:.0f}")


if __name__ == '__main__':
    main()
