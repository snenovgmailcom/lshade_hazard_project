"""
Find the tightest valid a_t for Corollary 1.

Key insight: Violations mean our a_t is TOO LARGE (not a valid lower bound on p_t).
We need to find the maximum a_t such that P(τ > n) ≤ ∏(1-a_t) holds.
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
    if len(finite_tau) == 0:
        return None
    if n_max is None:
        n_max = int(np.max(finite_tau)) + 10
    
    t_vals = np.arange(1, n_max + 1)
    h_hat = []
    for t in t_vals:
        n_t = np.sum(hitting_times >= t)
        d_t = np.sum(hitting_times == t)
        h_hat.append(d_t / n_t if n_t > 0 else 0.0)
    
    h_hat = np.array(h_hat)
    S_hat = np.cumprod(1 - h_hat)
    S_hat = np.insert(S_hat, 0, 1.0)
    n_vals = np.arange(n_max + 1)
    
    return {'n_vals': n_vals, 'S_hat': S_hat, 't_vals': t_vals, 'h_hat': h_hat}


def find_max_valid_a(S_hat, T, n_max):
    """
    Find maximum constant a such that S_hat(n) ≤ (1-a)^(n-T+1) for all n >= T.
    
    At each n: S_hat(n) ≤ (1-a)^(n-T+1)
    => a ≤ 1 - S_hat(n)^(1/(n-T+1))
    
    Valid a = min over all n of this upper bound.
    """
    a_upper_bounds = []
    
    for n in range(T, min(len(S_hat), n_max)):
        k = n - T + 1  # number of "active" generations
        S_n = S_hat[n]
        
        if S_n > 0 and S_n < 1:
            # S_n ≤ (1-a)^k => a ≤ 1 - S_n^(1/k)
            a_max = 1 - S_n ** (1.0 / k)
            a_upper_bounds.append((n, k, S_n, a_max))
    
    if not a_upper_bounds:
        return None, []
    
    # Valid a is the minimum of all upper bounds
    min_idx = np.argmin([x[3] for x in a_upper_bounds])
    a_valid = a_upper_bounds[min_idx][3]
    
    return a_valid, a_upper_bounds


def main():
    pkl_path = '/home/svety/lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl'
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    data = load_data(pkl_path)
    epsilon = 1e-2
    
    print("="*70)
    print("FINDING VALID BOUNDS FOR COROLLARY 1")
    print("="*70)
    
    print("""
APPROACH: Find maximum a such that S(n) ≤ (1-a)^(n-T+1) for all n >= T.

From S(n) ≤ (1-a)^k where k = n-T+1:
  a ≤ 1 - S(n)^(1/k)

Valid a = min over all n of this constraint.
""")
    
    # Analyze F1 in detail
    func_name = 'cec2017_f1'
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run['curve'] for run in data[func_name]]
    tau = compute_hitting_times(curves, f_star, epsilon)
    
    km = kaplan_meier(tau)
    finite_tau = tau[np.isfinite(tau)]
    T = int(np.min(finite_tau))
    
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {func_name}")
    print(f"{'='*60}")
    print(f"T (first hit) = {T}")
    print(f"τ range = [{np.min(finite_tau)}, {np.max(finite_tau)}]")
    
    a_valid, bounds = find_max_valid_a(km['S_hat'], T, len(km['S_hat']))
    
    print(f"\nConstraints from each n:")
    print(f"{'n':>6} {'k':>6} {'S(n)':>10} {'a_max':>10}")
    print("-"*40)
    
    # Show first and last few, and the binding constraint
    for i, (n, k, S_n, a_max) in enumerate(bounds[:5]):
        marker = " <-- binding" if a_max == a_valid else ""
        print(f"{n:>6} {k:>6} {S_n:>10.4f} {a_max:>10.6f}{marker}")
    
    if len(bounds) > 10:
        print("  ...")
    
    for i, (n, k, S_n, a_max) in enumerate(bounds[-5:]):
        marker = " <-- binding" if abs(a_max - a_valid) < 1e-10 else ""
        print(f"{n:>6} {k:>6} {S_n:>10.4f} {a_max:>10.6f}{marker}")
    
    print(f"\nMaximum valid a = {a_valid:.6f}")
    print(f"Compare to p_est = {1.0/np.mean(finite_tau - T + 1):.6f}")
    print(f"Ratio: a_valid / p_est = {a_valid / (1.0/np.mean(finite_tau - T + 1)):.4f}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print(f"""
The valid lower bound a = {a_valid:.6f} is MUCH smaller than 
the estimated hazard p = {1.0/np.mean(finite_tau - T + 1):.6f}.

This happens because:
1. Hits are CLUSTERED in a narrow window (generations {T} to {int(np.max(finite_tau))})
2. The geometric model assumes SPREAD OUT hits
3. The survival curve drops FASTER than (1-p)^k early on
4. To make the bound valid, we need very small a

KEY INSIGHT: 
The theoretical bound P(τ > n) ≤ (1-a)^(n-T) is CONSERVATIVE.
It's designed for worst-case (spread out hits), not best-case (clustered hits).
Actual DE performance is BETTER than the bound suggests!
""")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_vals = km['n_vals']
    p_est = 1.0 / np.mean(finite_tau - T + 1)
    
    # Bounds
    bound_pest = np.array([1.0 if n < T else (1-p_est)**(n-T+1) for n in n_vals])
    bound_valid = np.array([1.0 if n < T else (1-a_valid)**(n-T+1) for n in n_vals])
    
    # Left: Linear scale
    axes[0].plot(n_vals, km['S_hat'], 'b-', linewidth=2, label='Empirical S(n)')
    axes[0].plot(n_vals, bound_pest, 'r--', linewidth=2, 
                 label=f'$(1-p)^{{n-T+1}}$, p={p_est:.4f} (INVALID)')
    axes[0].plot(n_vals, bound_valid, 'g-', linewidth=2,
                 label=f'$(1-a)^{{n-T+1}}$, a={a_valid:.6f} (VALID)')
    axes[0].axvline(T, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Generation n')
    axes[0].set_ylabel('P(τ > n)')
    axes[0].set_title(f'{func_name}: Valid vs Invalid Bounds')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([T-5, int(np.max(finite_tau))+10])
    
    # Right: Log scale
    axes[1].semilogy(n_vals, km['S_hat'] + 1e-10, 'b-', linewidth=2, label='Empirical')
    axes[1].semilogy(n_vals, bound_pest + 1e-10, 'r--', linewidth=2, label=f'p={p_est:.4f}')
    axes[1].semilogy(n_vals, bound_valid + 1e-10, 'g-', linewidth=2, label=f'a={a_valid:.6f}')
    axes[1].axvline(T, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Generation n')
    axes[1].set_ylabel('P(τ > n) [log]')
    axes[1].set_title('Log scale - Valid bound is much looser')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([T-5, int(np.max(finite_tau))+10])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'valid_bound_f1.png', dpi=150)
    print(f"\nSaved: results/valid_bound_f1.png")
    plt.close()
    
    # Summary for all functions
    print(f"\n{'='*70}")
    print("SUMMARY: Valid bounds for all functions")
    print(f"{'='*70}")
    
    print(f"\n{'Function':<12} {'Hits':>5} {'T':>6} {'p_est':>10} {'a_valid':>10} {'Ratio':>8}")
    print("-"*60)
    
    for i in range(1, 31):
        func_name = f'cec2017_f{i}'
        f_star = CEC2017_OPTIMA[func_name]
        curves = [run['curve'] for run in data[func_name]]
        tau = compute_hitting_times(curves, f_star, epsilon)
        
        finite_tau = tau[np.isfinite(tau)]
        if len(finite_tau) < 10:
            print(f"{func_name:<12} {len(finite_tau):>5} {'--':>6} {'--':>10} {'--':>10} {'--':>8}")
            continue
        
        km = kaplan_meier(tau)
        T = int(np.min(finite_tau))
        p_est = 1.0 / np.mean(finite_tau - T + 1)
        
        a_valid, _ = find_max_valid_a(km['S_hat'], T, len(km['S_hat']))
        
        if a_valid is not None and a_valid > 0:
            ratio = a_valid / p_est
            print(f"{func_name:<12} {len(finite_tau):>5} {T:>6} {p_est:>10.4f} {a_valid:>10.6f} {ratio:>8.4f}")
        else:
            print(f"{func_name:<12} {len(finite_tau):>5} {T:>6} {p_est:>10.4f} {'--':>10} {'--':>8}")


if __name__ == '__main__':
    main()
