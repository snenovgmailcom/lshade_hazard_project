"""
Correct empirical analysis of hazard bounds.

What we observe: hitting times τ^(1), ..., τ^(51) per function
What we can compute: h_t = P(E_t | H_{t-1}) via Kaplan-Meier
What we CANNOT compute: p_t = P(E_t | F_{t-1}) (requires population state)

What we CAN verify:
1. Corollary 1: If p_t >= a_t (deterministic) on H_{t-1}, then P(τ>n) <= ∏(1-a_t)
2. Structure of h_t: constant, decaying, two-phase, etc.
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


def kaplan_meier(hitting_times, n_max=None):
    """
    Kaplan-Meier estimator for survival function.
    
    Returns:
        n_vals: generation indices
        S_hat: estimated P(τ > n)
        h_hat: estimated h_t = P(E_t | H_{t-1})
        at_risk: number at risk at each t
    """
    finite_tau = hitting_times[np.isfinite(hitting_times)]
    n_runs = len(hitting_times)
    
    if len(finite_tau) == 0:
        return None
    
    if n_max is None:
        n_max = int(np.max(finite_tau)) + 10
    
    t_vals = np.arange(1, n_max + 1)
    h_hat = []
    at_risk = []
    
    for t in t_vals:
        n_t = np.sum(hitting_times >= t)  # at risk
        d_t = np.sum(hitting_times == t)  # events (hits)
        at_risk.append(n_t)
        h_hat.append(d_t / n_t if n_t > 0 else 0.0)
    
    h_hat = np.array(h_hat)
    at_risk = np.array(at_risk)
    
    # Survival: S(n) = ∏_{t=1}^{n} (1 - h_t)
    S_hat = np.cumprod(1 - h_hat)
    S_hat = np.insert(S_hat, 0, 1.0)  # S(0) = 1
    n_vals = np.arange(n_max + 1)
    
    return {
        'n_vals': n_vals,
        'S_hat': S_hat,
        't_vals': t_vals,
        'h_hat': h_hat,
        'at_risk': at_risk
    }


def verify_corollary1(hitting_times, a_t_func, n_max=None):
    """
    Verify Corollary 1: P(τ > n) <= ∏(1 - a_t)
    
    Args:
        hitting_times: observed hitting times
        a_t_func: function t -> a_t (deterministic lower bound on p_t)
    
    Returns:
        True if bound holds for all n
    """
    km = kaplan_meier(hitting_times, n_max)
    if km is None:
        return None
    
    n_vals = km['n_vals']
    S_hat = km['S_hat']
    
    # Compute theoretical bound
    bound = np.ones(len(n_vals))
    for n in range(1, len(n_vals)):
        a_vals = [a_t_func(t) for t in range(1, n+1)]
        bound[n] = np.prod([1 - a for a in a_vals])
    
    # Check inequality
    violations = np.where(S_hat > bound + 1e-10)[0]
    
    return {
        'verified': len(violations) == 0,
        'violations': violations,
        'S_hat': S_hat,
        'bound': bound,
        'n_vals': n_vals
    }


def main():
    pkl_path = '/home/svety/lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl'
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    data = load_data(pkl_path)
    epsilon = 1e-2
    
    print("="*80)
    print("CORRECT EMPIRICAL ANALYSIS")
    print("="*80)
    
    print("""
CLARIFICATION:
- We observe: hitting times τ for each run
- We estimate: h_t = P(E_t | H_{t-1}) via Kaplan-Meier
- We CANNOT observe: p_t = P(E_t | F_{t-1}) (requires population state)

WHAT WE TEST:
- Corollary 1: If p_t >= a_t (deterministic) on H_{t-1}, then P(τ>n) <= ∏(1-a_t)
- We choose a_t based on observed h_t pattern and verify the bound holds
""")
    
    # Analyze F1 in detail
    func_name = 'cec2017_f1'
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run['curve'] for run in data[func_name]]
    tau = compute_hitting_times(curves, f_star, epsilon)
    
    km = kaplan_meier(tau)
    
    print(f"\n{'='*60}")
    print(f"FUNCTION: {func_name}")
    print(f"{'='*60}")
    
    finite_tau = tau[np.isfinite(tau)]
    T = int(np.min(finite_tau))
    
    print(f"Runs: {len(tau)}, Hits: {len(finite_tau)}")
    print(f"First hit (T): {T}")
    print(f"τ range: [{np.min(finite_tau)}, {np.max(finite_tau)}]")
    
    # Analyze h_t structure
    h_active = km['h_hat'][T-1:]  # Hazard after warm-up
    h_positive = h_active[h_active > 0]
    
    print(f"\nMarginal hazard h_t = P(E_t | H_{{t-1}}):")
    print(f"  h_t = 0 for t < {T}")
    print(f"  h_t > 0 for t >= {T}")
    print(f"  Mean h_t (when > 0): {np.mean(h_positive):.4f}")
    print(f"  Min h_t (when > 0): {np.min(h_positive):.4f}")
    print(f"  Max h_t (when > 0): {np.max(h_positive):.4f}")
    
    # Test Corollary 1 with conservative bound
    print(f"\n--- Corollary 1 Verification ---")
    
    # Conservative bound: a_t = 0 for t < T, a_t = a for t >= T
    a_conservative = np.min(h_positive) * 0.5  # 50% of minimum observed
    
    def a_t_conservative(t):
        return a_conservative if t >= T else 0.0
    
    result = verify_corollary1(tau, a_t_conservative)
    
    print(f"\nUsing a_t = {a_conservative:.4f} for t >= {T}, else 0:")
    print(f"Corollary 1 verified: {result['verified']}")
    if not result['verified']:
        print(f"Violations at n = {result['violations']}")
    
    # Also test with a_t = min(h_t)
    a_min = np.min(h_positive)
    
    def a_t_min(t):
        return a_min if t >= T else 0.0
    
    result2 = verify_corollary1(tau, a_t_min)
    
    print(f"\nUsing a_t = {a_min:.4f} (min observed h_t) for t >= {T}:")
    print(f"Corollary 1 verified: {result2['verified']}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Survival curve
    axes[0,0].plot(km['n_vals'], km['S_hat'], 'b-', linewidth=2, 
                   label=r'$\hat{S}(n) = \prod(1-\hat{h}_t)$ (Kaplan-Meier)')
    axes[0,0].plot(result['n_vals'], result['bound'], 'r--', linewidth=2,
                   label=f'Bound: $\\prod(1-a_t)$, $a_t={a_conservative:.3f}$')
    axes[0,0].axvline(T, color='gray', linestyle=':', alpha=0.7, label=f'T={T}')
    axes[0,0].set_xlabel('Generation n')
    axes[0,0].set_ylabel('P(τ > n)')
    axes[0,0].set_title(f'{func_name}: Survival and Corollary 1 Bound')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Top right: Log scale
    axes[0,1].semilogy(km['n_vals'], km['S_hat'] + 1e-10, 'b-', linewidth=2)
    axes[0,1].semilogy(result['n_vals'], result['bound'] + 1e-10, 'r--', linewidth=2)
    axes[0,1].axvline(T, color='gray', linestyle=':', alpha=0.7)
    axes[0,1].set_xlabel('Generation n')
    axes[0,1].set_ylabel('P(τ > n) [log scale]')
    axes[0,1].set_title('Log scale view')
    axes[0,1].grid(True, alpha=0.3)
    
    # Bottom left: Marginal hazard h_t
    axes[1,0].plot(km['t_vals'], km['h_hat'], 'b.', markersize=3, alpha=0.5)
    axes[1,0].axhline(a_conservative, color='red', linestyle='--', 
                      label=f'$a_t = {a_conservative:.4f}$')
    axes[1,0].axhline(a_min, color='orange', linestyle=':', 
                      label=f'min $h_t = {a_min:.4f}$')
    axes[1,0].axvline(T, color='gray', linestyle=':', alpha=0.7)
    axes[1,0].set_xlabel('Generation t')
    axes[1,0].set_ylabel(r'$\hat{h}_t = P(E_t | H_{t-1})$')
    axes[1,0].set_title('Marginal Hazard (Kaplan-Meier)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Bottom right: At risk
    axes[1,1].plot(km['t_vals'], km['at_risk'], 'g-', linewidth=2)
    axes[1,1].axvline(T, color='gray', linestyle=':', alpha=0.7)
    axes[1,1].set_xlabel('Generation t')
    axes[1,1].set_ylabel('Number at risk')
    axes[1,1].set_title('Runs Still Surviving at t')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correct_analysis_f1.png', dpi=150)
    print(f"\nSaved: results/correct_analysis_f1.png")
    plt.close()
    
    # Summary for all functions
    print(f"\n{'='*80}")
    print("SUMMARY: Corollary 1 Verification (all functions with >= 10 hits)")
    print(f"{'='*80}")
    print(f"\n{'Function':<12} {'Hits':>5} {'T':>6} {'min h_t':>10} {'a_t':>10} {'Verified':>10}")
    print("-"*60)
    
    for i in range(1, 31):
        func_name = f'cec2017_f{i}'
        f_star = CEC2017_OPTIMA[func_name]
        curves = [run['curve'] for run in data[func_name]]
        tau = compute_hitting_times(curves, f_star, epsilon)
        
        finite_tau = tau[np.isfinite(tau)]
        if len(finite_tau) < 10:
            print(f"{func_name:<12} {len(finite_tau):>5} {'--':>6} {'--':>10} {'--':>10} {'--':>10}")
            continue
        
        km = kaplan_meier(tau)
        T = int(np.min(finite_tau))
        
        h_active = km['h_hat'][T-1:]
        h_positive = h_active[h_active > 0]
        
        if len(h_positive) == 0:
            continue
            
        a_test = np.min(h_positive) * 0.5
        
        def a_t_test(t, T=T, a=a_test):
            return a if t >= T else 0.0
        
        result = verify_corollary1(tau, a_t_test)
        
        print(f"{func_name:<12} {len(finite_tau):>5} {T:>6} {np.min(h_positive):>10.4f} "
              f"{a_test:>10.4f} {'✓' if result['verified'] else '✗':>10}")


if __name__ == '__main__':
    main()
