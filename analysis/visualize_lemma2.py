"""
Visualize Lemma 2 verification: P(τ > n) ≤ E[∏(1-p_t)]
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


def empirical_survival(hitting_times, n_max):
    n_values = np.arange(n_max + 1)
    n_runs = len(hitting_times)
    survival = np.array([np.sum(hitting_times > n) / n_runs for n in n_values])
    return n_values, survival


def empirical_hazard(hitting_times, n_max):
    t_values = np.arange(1, n_max + 1)
    hazard = []
    for t in t_values:
        at_risk = np.sum(hitting_times >= t)
        hits_at_t = np.sum(hitting_times == t)
        hazard.append(hits_at_t / at_risk if at_risk > 0 else 0.0)
    return t_values, np.array(hazard)


def main():
    pkl_path = '/home/svety/lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl'
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    data = load_data(pkl_path)
    
    # Use F1 with epsilon=1e-8 (all 51 runs hit)
    func_name = 'cec2017_f1'
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run['curve'] for run in data[func_name]]
    epsilon = 1e-8
    
    tau = compute_hitting_times(curves, f_star, epsilon)
    n_max = int(np.max(tau)) + 20
    
    # Compute empirical survival P(τ > n)
    n_vals, survival = empirical_survival(tau, n_max)
    
    # Compute empirical hazard p_t
    t_vals, hazard = empirical_hazard(tau, n_max)
    
    # Compute product bound ∏(1-p_t)
    product_bound = np.cumprod(1 - hazard)
    product_bound = np.insert(product_bound, 0, 1.0)  # P(τ > 0) = 1
    
    # Check inequality at each point
    diff = product_bound[:len(survival)] - survival
    violations = np.sum(diff < -1e-10)
    
    print(f"Function: {func_name}, epsilon = {epsilon}")
    print(f"Runs: {len(tau)}, all hit: {np.all(np.isfinite(tau))}")
    print(f"\nLemma 2 check: P(τ > n) ≤ ∏(1-p_t)")
    print(f"Violations: {violations}")
    print(f"\nSample values:")
    print(f"{'n':>6} {'P(τ>n)':>12} {'∏(1-p_t)':>12} {'Gap':>12}")
    print("-" * 48)
    
    for n in [0, 100, 200, 300, 320, 340, 350, 360]:
        if n < len(survival):
            print(f"{n:>6} {survival[n]:>12.6f} {product_bound[n]:>12.6f} "
                  f"{product_bound[n] - survival[n]:>12.6f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Linear scale
    axes[0].plot(n_vals, survival, 'b-', linewidth=2, label='P(τ > n) [empirical]')
    axes[0].plot(n_vals[:len(product_bound)], product_bound, 'r--', linewidth=2, 
                 label='∏(1-p_t) [upper bound]')
    axes[0].fill_between(n_vals[:len(product_bound)], survival[:len(product_bound)], 
                         product_bound, alpha=0.3, color='green', label='Gap (bound - actual)')
    axes[0].set_xlabel('Generation n')
    axes[0].set_ylabel('Probability')
    axes[0].set_title(f'Lemma 2 Verification: {func_name}\nP(τ > n) ≤ ∏(1-p_t)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: Log scale (better view of tail)
    axes[1].semilogy(n_vals, survival + 1e-10, 'b-', linewidth=2, label='P(τ > n)')
    axes[1].semilogy(n_vals[:len(product_bound)], product_bound + 1e-10, 'r--', 
                     linewidth=2, label='∏(1-p_t)')
    axes[1].set_xlabel('Generation n')
    axes[1].set_ylabel('Probability (log scale)')
    axes[1].set_title('Log scale view')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lemma2_verification.png', dpi=150)
    print(f"\nSaved: results/lemma2_verification.png")
    plt.close()


if __name__ == '__main__':
    main()
