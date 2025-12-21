"""
Correct Corollary 1 verification.

Issue: a_t must be a LOWER bound on p_t, not based on observed h_t.

Approach: Use a_t = 0 at generations with no observed hits,
          and a_t = min(h_t > 0) only at generations with hits.
          
Or: Use very conservative a_t that accounts for estimation uncertainty.
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
    finite_tau = hitting_times[np.isfinite(hitting_times)]
    n_runs = len(hitting_times)
    
    if len(finite_tau) == 0:
        return None
    
    if n_max is None:
        n_max = int(np.max(finite_tau)) + 10
    
    t_vals = np.arange(1, n_max + 1)
    h_hat = []
    at_risk = []
    d_t_list = []
    
    for t in t_vals:
        n_t = np.sum(hitting_times >= t)
        d_t = np.sum(hitting_times == t)
        at_risk.append(n_t)
        d_t_list.append(d_t)
        h_hat.append(d_t / n_t if n_t > 0 else 0.0)
    
    h_hat = np.array(h_hat)
    at_risk = np.array(at_risk)
    d_t_list = np.array(d_t_list)
    
    S_hat = np.cumprod(1 - h_hat)
    S_hat = np.insert(S_hat, 0, 1.0)
    n_vals = np.arange(n_max + 1)
    
    return {
        'n_vals': n_vals,
        'S_hat': S_hat,
        't_vals': t_vals,
        'h_hat': h_hat,
        'd_t': d_t_list,
        'at_risk': at_risk
    }


def main():
    pkl_path = '/home/svety/lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl'
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    data = load_data(pkl_path)
    epsilon = 1e-2
    
    func_name = 'cec2017_f1'
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run['curve'] for run in data[func_name]]
    tau = compute_hitting_times(curves, f_star, epsilon)
    
    km = kaplan_meier(tau)
    finite_tau = tau[np.isfinite(tau)]
    T = int(np.min(finite_tau))
    
    print("="*70)
    print(f"UNDERSTANDING THE VIOLATION: {func_name}")
    print("="*70)
    
    # Show h_t around first hit
    print(f"\nHazard h_t around first hit (T = {T}):")
    print(f"{'t':>6} {'d_t':>6} {'n_t':>6} {'h_t':>10} {'S(t)':>10}")
    print("-"*50)
    
    for i, t in enumerate(range(T-2, min(T+15, len(km['t_vals'])))):
        if t < 1:
            continue
        idx = t - 1
        d_t = km['d_t'][idx]
        n_t = km['at_risk'][idx]
        h_t = km['h_hat'][idx]
        S_t = km['S_hat'][t]
        marker = " <-- first hit" if t == T else ""
        marker = " <-- h_t = 0!" if h_t == 0 and t >= T else marker
        print(f"{t:>6} {d_t:>6} {n_t:>6} {h_t:>10.4f} {S_t:>10.4f}{marker}")
    
    # Count generations with h_t = 0 after T
    h_after_T = km['h_hat'][T-1:]
    n_zero = np.sum(h_after_T == 0)
    n_positive = np.sum(h_after_T > 0)
    
    print(f"\nAfter T={T}:")
    print(f"  Generations with h_t = 0: {n_zero}")
    print(f"  Generations with h_t > 0: {n_positive}")
    
    print(f"\n" + "="*70)
    print("WHY COROLLARY 1 'FAILS' WITH NAIVE a_t")
    print("="*70)
    print("""
The bound ∏(1-a_t) assumes a_t > 0 for ALL t >= T.
But empirically, h_t = 0 at many generations (no hits there).

At those generations:
  - S_hat stays flat: S_hat(t) = S_hat(t-1) × 1 = S_hat(t-1)  
  - Bound decreases: B(t) = B(t-1) × (1-a_t) < B(t-1)

Eventually B(t) < S_hat(t) → "violation"

This is NOT a failure of the theorem! It means:
  - a_t is too large (not a valid lower bound on p_t)
  - OR the bound is simply not tight
""")
    
    # Correct approach: only apply a_t where h_t > 0
    print(f"\n" + "="*70)
    print("CORRECT APPROACH: Bound only where h_t > 0")
    print("="*70)
    
    # Method 1: Use a_t = h_t (this gives equality by K-M)
    # Method 2: Use a_t = c * h_t for some c < 1
    # Method 3: Use constant a after T, but only count generations with hits
    
    # Let's compute: if we assume constant hazard p in active phase,
    # what bound do we get?
    
    # From two-phase model: E[τ - T] = 1/p, so p ≈ 1/mean(τ - T)
    tau_shifted = finite_tau - T + 1
    p_est = 1.0 / np.mean(tau_shifted)
    
    # Theoretical bound with this p
    n_vals = km['n_vals']
    bound_constant = np.ones(len(n_vals))
    for n in range(1, len(n_vals)):
        if n < T:
            bound_constant[n] = 1.0
        else:
            bound_constant[n] = (1 - p_est) ** (n - T + 1)
    
    # Check
    violations_constant = np.sum(km['S_hat'] > bound_constant + 1e-10)
    
    print(f"\nUsing constant a_t = p = {p_est:.4f} (from E[τ-T] = 1/p):")
    print(f"Bound: P(τ > n) ≤ (1-p)^(n-T+1) for n >= T")
    print(f"Violations: {violations_constant}")
    
    # More conservative: use p/2
    p_conservative = p_est / 2
    bound_conservative = np.ones(len(n_vals))
    for n in range(1, len(n_vals)):
        if n < T:
            bound_conservative[n] = 1.0
        else:
            bound_conservative[n] = (1 - p_conservative) ** (n - T + 1)
    
    violations_conservative = np.sum(km['S_hat'] > bound_conservative + 1e-10)
    
    print(f"\nUsing conservative a_t = p/2 = {p_conservative:.4f}:")
    print(f"Violations: {violations_conservative}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: All bounds
    axes[0].plot(n_vals, km['S_hat'], 'b-', linewidth=2, label='Empirical S(n)')
    axes[0].plot(n_vals, bound_constant, 'r--', linewidth=2, 
                 label=f'$(1-p)^{{n-T+1}}$, p={p_est:.3f}')
    axes[0].plot(n_vals, bound_conservative, 'g:', linewidth=2,
                 label=f'$(1-p/2)^{{n-T+1}}$')
    axes[0].axvline(T, color='gray', linestyle='--', alpha=0.5, label=f'T={T}')
    axes[0].set_xlabel('Generation n')
    axes[0].set_ylabel('P(τ > n)')
    axes[0].set_title(f'{func_name}: Survival and Bounds')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([T-10, np.max(finite_tau)+20])
    
    # Right: Log scale
    axes[1].semilogy(n_vals, km['S_hat'] + 1e-10, 'b-', linewidth=2, label='Empirical')
    axes[1].semilogy(n_vals, bound_constant + 1e-10, 'r--', linewidth=2, label='Constant p')
    axes[1].semilogy(n_vals, bound_conservative + 1e-10, 'g:', linewidth=2, label='Conservative p/2')
    axes[1].axvline(T, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Generation n')
    axes[1].set_ylabel('P(τ > n) [log]')
    axes[1].set_title('Log scale')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([T-10, np.max(finite_tau)+20])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'corollary1_correct.png', dpi=150)
    print(f"\nSaved: results/corollary1_correct.png")
    plt.close()
    
    # Summary for all functions
    print(f"\n" + "="*70)
    print("SUMMARY: All functions (using constant p from two-phase model)")
    print("="*70)
    
    print(f"\n{'Function':<12} {'Hits':>5} {'T':>6} {'p_est':>8} {'Verified':>10} {'p/2':>10}")
    print("-"*60)
    
    for i in range(1, 31):
        func_name = f'cec2017_f{i}'
        f_star = CEC2017_OPTIMA[func_name]
        curves = [run['curve'] for run in data[func_name]]
        tau = compute_hitting_times(curves, f_star, epsilon)
        
        finite_tau = tau[np.isfinite(tau)]
        if len(finite_tau) < 10:
            print(f"{func_name:<12} {len(finite_tau):>5} {'--':>6} {'--':>8} {'--':>10} {'--':>10}")
            continue
        
        km = kaplan_meier(tau)
        T = int(np.min(finite_tau))
        
        tau_shifted = finite_tau - T + 1
        p_est = 1.0 / np.mean(tau_shifted)
        
        # Check constant bound
        n_vals = km['n_vals']
        bound = np.array([1.0 if n < T else (1-p_est)**(n-T+1) for n in n_vals])
        v1 = np.sum(km['S_hat'] > bound + 1e-10)
        
        # Check conservative bound
        bound2 = np.array([1.0 if n < T else (1-p_est/2)**(n-T+1) for n in n_vals])
        v2 = np.sum(km['S_hat'] > bound2 + 1e-10)
        
        print(f"{func_name:<12} {len(finite_tau):>5} {T:>6} {p_est:>8.4f} "
              f"{'✓' if v1 == 0 else f'✗({v1})':>10} {'✓' if v2 == 0 else f'✗({v2})':>10}")


if __name__ == '__main__':
    main()
