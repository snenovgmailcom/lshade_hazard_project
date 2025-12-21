"""
Compare empirical results with theoretical bounds from the paper.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

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
    for t in t_values:
        at_risk = np.sum(hitting_times >= t)
        hits_at_t = np.sum(hitting_times == t)
        hazard.append(hits_at_t / at_risk if at_risk > 0 else np.nan)
    return t_values, np.array(hazard)


def verify_product_formula(hitting_times):
    """Verify: P(tau > n) vs product formula."""
    t_vals, hazard = empirical_hazard(hitting_times)
    n_vals, survival = empirical_survival(hitting_times)
    
    hazard_clean = np.nan_to_num(hazard, nan=0.0)
    product_bound = np.cumprod(1 - hazard_clean)
    product_bound = np.insert(product_bound, 0, 1.0)
    
    return n_vals, survival, product_bound[:len(n_vals)]


def fit_tail_models(n_vals, survival):
    """Fit exponential, stretched-exponential, and power-law models."""
    results = {}
    
    mask = survival > 0
    if np.sum(mask) < 5:
        return results
    
    n_fit = n_vals[mask]
    surv_fit = survival[mask]
    log_surv = np.log(surv_fit)
    
    # 1. Exponential
    try:
        slope, intercept = np.polyfit(n_fit, log_surv, 1)
        p_exp = 1 - np.exp(slope)
        surv_pred = np.exp(slope * n_fit + intercept)
        ss_res = np.sum((surv_fit - surv_pred)**2)
        ss_tot = np.sum((surv_fit - np.mean(surv_fit))**2)
        r2_exp = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results['exponential'] = {'p': p_exp, 'R2': r2_exp, 'E_tau': 1/p_exp if p_exp > 0 else np.inf}
    except:
        pass
    
    # 2. Stretched exponential
    try:
        mask2 = (surv_fit < 1) & (surv_fit > 0) & (n_fit > 0)
        if np.sum(mask2) >= 5:
            log_log_surv = np.log(-np.log(surv_fit[mask2]))
            log_n = np.log(n_fit[mask2])
            beta, log_c = np.polyfit(log_n, log_log_surv, 1)
            c = np.exp(log_c)
            surv_pred = np.exp(-c * n_fit[mask2]**beta)
            ss_res = np.sum((surv_fit[mask2] - surv_pred)**2)
            ss_tot = np.sum((surv_fit[mask2] - np.mean(surv_fit[mask2]))**2)
            r2_str = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            results['stretched_exp'] = {'c': c, 'beta': beta, 'R2': r2_str}
    except:
        pass
    
    return results


def analyze_hazard_behavior(hitting_times):
    """Analyze if hazard rate is constant, increasing, or decreasing."""
    t_vals, hazard = empirical_hazard(hitting_times)
    
    mask = ~np.isnan(hazard) & (hazard > 0)
    if np.sum(mask) < 10:
        return None
    
    t_clean = t_vals[mask]
    h_clean = hazard[mask]
    
    mean_h = np.mean(h_clean)
    std_h = np.std(h_clean)
    cv = std_h / mean_h if mean_h > 0 else np.inf
    
    # Polynomial decay fit
    try:
        log_t = np.log(t_clean)
        log_h = np.log(h_clean)
        slope, intercept, r_value, _, _ = stats.linregress(log_t, log_h)
        alpha = -slope
        c = np.exp(intercept)
        r2_decay = r_value**2
    except:
        alpha, c, r2_decay = 0, 0, 0
    
    # Detect threshold behavior
    first_hit_gen = t_clean[0]
    
    return {
        'mean_hazard': mean_h,
        'std_hazard': std_h,
        'cv': cv,
        'decay_alpha': alpha,
        'decay_c': c,
        'decay_R2': r2_decay,
        'first_hit_gen': first_hit_gen,
        'is_approximately_constant': cv < 0.5,
        'is_decaying': alpha > 0.1 and r2_decay > 0.3
    }


def main():
    import sys
    
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/svety/lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl'
    
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading: {pkl_path}")
    data = load_data(pkl_path)
    
    # ===== Detailed analysis for F1 =====
    func_name = 'cec2017_f1'
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run['curve'] for run in data[func_name]]
    epsilon = 1e-8
    
    tau = compute_hitting_times(curves, f_star, epsilon)
    
    print(f"\n{'='*70}")
    print(f"THEORETICAL COMPARISON: {func_name}, eps = {epsilon}")
    print(f"{'='*70}")
    
    # 1. Verify product formula
    print("\n--- Product Formula Verification (Lemma 1 & 2) ---")
    n_vals, survival, product_bound = verify_product_formula(tau)
    
    violations = np.sum(survival > product_bound + 1e-10)
    print(f"P(tau>n) <= E[prod(1-p_t)]: {'VERIFIED' if violations == 0 else f'VIOLATED at {violations} points'}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(n_vals, survival + 1e-10, 'b-', linewidth=2, label='Empirical P(tau > n)')
    ax.semilogy(n_vals, product_bound + 1e-10, 'r--', linewidth=2, label='Product bound prod(1-p_t)')
    ax.set_xlabel('Generation n')
    ax.set_ylabel('Probability (log scale)')
    ax.set_title(f'{func_name}: Survival vs Product Bound (eps = {epsilon:.0e})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{func_name}_product_comparison.png', dpi=150)
    print(f"Saved: {output_dir / f'{func_name}_product_comparison.png'}")
    plt.close()
    
    # 2. Fit tail models
    print("\n--- Tail Model Fitting ---")
    fits = fit_tail_models(n_vals, survival)
    
    for model, params in fits.items():
        print(f"\n{model}:")
        for k, v in params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    # 3. Analyze hazard behavior
    print("\n--- Hazard Rate Analysis ---")
    hazard_analysis = analyze_hazard_behavior(tau)
    
    if hazard_analysis:
        print(f"First hit at generation: {hazard_analysis['first_hit_gen']}")
        print(f"Mean hazard p (when p>0): {hazard_analysis['mean_hazard']:.4f}")
        print(f"Coefficient of variation: {hazard_analysis['cv']:.4f}")
        
        alpha = hazard_analysis['decay_alpha']
        c = hazard_analysis['decay_c']
        r2 = hazard_analysis['decay_R2']
        print(f"Decay fit: p_t ~ {c:.4f} * t^(-{alpha:.4f}) (R2 = {r2:.4f})")
    
    # 4. Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Hazard Analysis for All Functions (eps = 1e-2)")
    print(f"{'='*70}")
    print(f"{'Function':<12} {'Hits':>5} {'1st Gen':>8} {'Mean p':>8} {'CV(p)':>8}")
    print("-" * 50)
    
    for i in range(1, 31):
        func_name = f'cec2017_f{i}'
        f_star = CEC2017_OPTIMA[func_name]
        curves = [run['curve'] for run in data[func_name]]
        
        tau = compute_hitting_times(curves, f_star, epsilon=1e-2)
        n_hits = np.sum(np.isfinite(tau))
        
        if n_hits >= 5:
            hazard_analysis = analyze_hazard_behavior(tau)
            if hazard_analysis:
                first_gen = hazard_analysis['first_hit_gen']
                mean_p = hazard_analysis['mean_hazard']
                cv = hazard_analysis['cv']
                print(f"{func_name:<12} {n_hits:>5} {first_gen:>8.0f} {mean_p:>8.4f} {cv:>8.2f}")
            else:
                print(f"{func_name:<12} {n_hits:>5} {'--':>8} {'--':>8} {'--':>8}")
        else:
            print(f"{func_name:<12} {n_hits:>5} {'--':>8} {'--':>8} {'--':>8}")


if __name__ == '__main__':
    main()
