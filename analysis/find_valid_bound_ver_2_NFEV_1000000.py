"""
Analyze extended budget experiments (1M evaluations) and compare with baseline (100k).
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

CEC2017_OPTIMA = {f"cec2017_f{i}": 100.0 * i for i in range(1, 31)}

# Paths
BASELINE_PKL = "experiments/r_lshade_D10_nfev_100000/raw_results_lshade.pkl"
EXTENDED_PKL = "experiments/r_lshade_D10_nfev_1000000/raw_results_lshade.pkl"
OUTPUT_DIR = Path("results")

EPSILON = 1e-2


def load_data(pkl_path: str) -> Dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@dataclass
class EventData:
    y: np.ndarray       # observed time (hit or censor)
    delta: np.ndarray   # 1 if hit, 0 if censored
    c: np.ndarray       # censor time (max available)


def compute_event_data(curves: List[np.ndarray], f_star: float, epsilon: float) -> EventData:
    target = f_star + epsilon
    y, delta, c = [], [], []
    
    for curve in curves:
        c_i = len(curve) - 1
        hits = np.where(curve <= target)[0]
        
        if len(hits) > 0:
            y.append(int(hits[0]))
            delta.append(1)
        else:
            y.append(c_i)
            delta.append(0)
        c.append(c_i)
    
    return EventData(
        y=np.array(y, dtype=int),
        delta=np.array(delta, dtype=int),
        c=np.array(c, dtype=int)
    )


def kaplan_meier(y: np.ndarray, delta: np.ndarray, n_max: Optional[int] = None):
    if n_max is None:
        n_max = int(np.max(y))
    
    t_vals = np.arange(0, n_max + 1)
    h_hat = np.zeros(len(t_vals), dtype=float)
    n_at_risk = np.zeros(len(t_vals), dtype=int)
    d_events = np.zeros(len(t_vals), dtype=int)
    
    for idx, t in enumerate(t_vals):
        n_t = int(np.sum(y >= t))
        d_t = int(np.sum((delta == 1) & (y == t)))
        n_at_risk[idx] = n_t
        d_events[idx] = d_t
        h_hat[idx] = d_t / n_t if n_t > 0 else 0.0
    
    S_hat = np.cumprod(1.0 - h_hat)
    
    return {
        "t_vals": t_vals,
        "S_hat": S_hat,
        "h_hat": h_hat,
        "n_at_risk": n_at_risk,
        "d_events": d_events,
        "n_max": n_max
    }


def find_T(y: np.ndarray, delta: np.ndarray) -> Optional[int]:
    hits = y[delta == 1]
    return int(np.min(hits)) if len(hits) > 0 else None


def constant_hazard_mle(y: np.ndarray, delta: np.ndarray, T: int) -> Optional[float]:
    mask = (y >= T)
    if not np.any(mask):
        return None
    
    exposure = (y[mask] - T + 1).astype(float)
    events = delta[mask].astype(float)
    
    total_exposure = float(np.sum(exposure))
    total_events = float(np.sum(events))
    
    return total_events / total_exposure if total_exposure > 0 else None


def find_max_valid_a(S_hat: np.ndarray, T: int) -> Tuple[Optional[float], int]:
    if T < 1 or T >= len(S_hat):
        return None, 0
    
    S0 = S_hat[T - 1]
    if S0 <= 0:
        return None, 0
    
    a_candidates = []
    binding_n = 0
    
    for n in range(T, len(S_hat)):
        k = n - T + 1
        S_cond = S_hat[n] / S0
        
        if 0 < S_cond < 1:
            a_max = 1.0 - S_cond ** (1.0 / k)
            a_candidates.append((a_max, n))
    
    if not a_candidates:
        return None, 0
    
    a_valid, binding_n = min(a_candidates, key=lambda x: x[0])
    return float(a_valid), binding_n


def analyze_function(curves: List[np.ndarray], f_star: float, epsilon: float) -> Dict:
    ev = compute_event_data(curves, f_star, epsilon)
    
    hits = int(np.sum(ev.delta == 1))
    cens = int(np.sum(ev.delta == 0))
    n_runs = len(ev.y)
    
    result = {
        "n_runs": n_runs,
        "hits": hits,
        "censored": cens,
        "hit_rate": hits / n_runs if n_runs > 0 else 0,
        "T": None,
        "p_cens": None,
        "a_valid": None,
        "ratio": None,
        "tau_min": None,
        "tau_median": None,
        "tau_max": None,
        "tau_mean": None,
    }
    
    if hits == 0:
        return result
    
    finite_tau = ev.y[ev.delta == 1]
    result["tau_min"] = int(np.min(finite_tau))
    result["tau_median"] = float(np.median(finite_tau))
    result["tau_max"] = int(np.max(finite_tau))
    result["tau_mean"] = float(np.mean(finite_tau))
    
    T = find_T(ev.y, ev.delta)
    result["T"] = T
    
    if T is None or hits < 5:
        return result
    
    km = kaplan_meier(ev.y, ev.delta)
    
    p_cens = constant_hazard_mle(ev.y, ev.delta, T)
    result["p_cens"] = p_cens
    
    a_valid, _ = find_max_valid_a(km["S_hat"], T)
    result["a_valid"] = a_valid
    
    if p_cens and a_valid and p_cens > 0:
        result["ratio"] = a_valid / p_cens
    
    return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 90)
    print("EXTENDED BUDGET ANALYSIS: D=10, NFEV=1,000,000 vs 100,000")
    print("=" * 90)
    
    # Load data
    print(f"\nLoading baseline (100k): {BASELINE_PKL}")
    try:
        data_100k = load_data(BASELINE_PKL)
        has_100k = True
    except FileNotFoundError:
        print("  Baseline not found, skipping comparison")
        has_100k = False
        data_100k = {}
    
    print(f"Loading extended (1M): {EXTENDED_PKL}")
    data_1M = load_data(EXTENDED_PKL)
    
    # Get list of functions in extended data
    funcs_1M = sorted(data_1M.keys(), key=lambda x: int(x.split('_f')[1]))
    print(f"\nFunctions in extended data: {len(funcs_1M)}")
    print(f"  {', '.join(funcs_1M)}")
    
    # Analyze extended data
    print("\n" + "=" * 90)
    print("EXTENDED BUDGET RESULTS (NFEV = 1,000,000)")
    print("=" * 90)
    
    print(f"\n{'Function':<12} {'Hits':>5} {'Rate':>7} {'T':>8} {'τ_min':>8} {'τ_med':>8} "
          f"{'τ_max':>8} {'p_cens':>10} {'a_valid':>10} {'a/p':>8}")
    print("-" * 100)
    
    results_1M = {}
    for func_name in funcs_1M:
        f_star = CEC2017_OPTIMA[func_name]
        curves = [run["curve"] for run in data_1M[func_name]]
        
        r = analyze_function(curves, f_star, EPSILON)
        results_1M[func_name] = r
        
        hits_str = f"{r['hits']}/{r['n_runs']}"
        rate_str = f"{r['hit_rate']:.1%}"
        T_str = f"{r['T']}" if r['T'] else "--"
        tau_min_str = f"{r['tau_min']}" if r['tau_min'] else "--"
        tau_med_str = f"{r['tau_median']:.0f}" if r['tau_median'] else "--"
        tau_max_str = f"{r['tau_max']}" if r['tau_max'] else "--"
        p_str = f"{r['p_cens']:.6f}" if r['p_cens'] else "--"
        a_str = f"{r['a_valid']:.6f}" if r['a_valid'] else "--"
        ratio_str = f"{r['ratio']:.4f}" if r['ratio'] else "--"
        
        print(f"{func_name:<12} {hits_str:>5} {rate_str:>7} {T_str:>8} {tau_min_str:>8} "
              f"{tau_med_str:>8} {tau_max_str:>8} {p_str:>10} {a_str:>10} {ratio_str:>8}")
    
    # Compare with baseline if available
    if has_100k:
        print("\n" + "=" * 90)
        print("COMPARISON: 100k vs 1M evaluations")
        print("=" * 90)
        
        # Only compare functions present in both
        common_funcs = set(data_100k.keys()) & set(data_1M.keys())
        
        print(f"\n{'Function':<12} {'100k Hits':>10} {'100k Rate':>10} {'1M Hits':>10} {'1M Rate':>10} {'Improvement':>12}")
        print("-" * 70)
        
        for i in range(1, 31):
            func_name = f"cec2017_f{i}"
            
            # 100k results
            if func_name in data_100k:
                f_star = CEC2017_OPTIMA[func_name]
                curves_100k = [run["curve"] for run in data_100k[func_name]]
                r_100k = analyze_function(curves_100k, f_star, EPSILON)
                hits_100k = f"{r_100k['hits']}/{r_100k['n_runs']}"
                rate_100k = r_100k['hit_rate']
                rate_100k_str = f"{rate_100k:.1%}"
            else:
                hits_100k = "--"
                rate_100k = None
                rate_100k_str = "--"
            
            # 1M results
            if func_name in results_1M:
                r_1M = results_1M[func_name]
                hits_1M = f"{r_1M['hits']}/{r_1M['n_runs']}"
                rate_1M = r_1M['hit_rate']
                rate_1M_str = f"{rate_1M:.1%}"
            else:
                hits_1M = "--"
                rate_1M = None
                rate_1M_str = "--"
            
            # Improvement
            if rate_100k is not None and rate_1M is not None and rate_100k > 0:
                improvement = f"{rate_1M / rate_100k:.1f}x"
            elif rate_100k == 0 and rate_1M and rate_1M > 0:
                improvement = "∞ (0→>0)"
            else:
                improvement = "--"
            
            print(f"{func_name:<12} {hits_100k:>10} {rate_100k_str:>10} {hits_1M:>10} {rate_1M_str:>10} {improvement:>12}")
    
    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS (1M budget)")
    print("=" * 90)
    
    total_funcs = len(results_1M)
    funcs_100pct = sum(1 for r in results_1M.values() if r['hit_rate'] == 1.0)
    funcs_some_hits = sum(1 for r in results_1M.values() if r['hits'] > 0)
    funcs_no_hits = sum(1 for r in results_1M.values() if r['hits'] == 0)
    
    print(f"\nTotal functions analyzed: {total_funcs}")
    print(f"Functions with 100% hit rate: {funcs_100pct}")
    print(f"Functions with some hits (>0%): {funcs_some_hits}")
    print(f"Functions with no hits: {funcs_no_hits}")
    
    # Classify by a/p ratio
    print("\n" + "-" * 50)
    print("Classification by a/p ratio (hazard clustering):")
    print("-" * 50)
    
    clustered = []      # a/p < 0.3
    moderate = []       # 0.3 <= a/p < 0.7
    geometric = []      # a/p >= 0.7
    insufficient = []   # can't compute
    
    for fname, r in results_1M.items():
        if r['ratio'] is None:
            insufficient.append(fname)
        elif r['ratio'] < 0.3:
            clustered.append((fname, r['ratio']))
        elif r['ratio'] < 0.7:
            moderate.append((fname, r['ratio']))
        else:
            geometric.append((fname, r['ratio']))
    
    print(f"\nClustered hits (a/p < 0.3): {len(clustered)}")
    for f, ratio in sorted(clustered, key=lambda x: x[1]):
        print(f"  {f}: a/p = {ratio:.4f}")
    
    print(f"\nModerate (0.3 ≤ a/p < 0.7): {len(moderate)}")
    for f, ratio in sorted(moderate, key=lambda x: x[1]):
        print(f"  {f}: a/p = {ratio:.4f}")
    
    print(f"\nGeometric-like (a/p ≥ 0.7): {len(geometric)}")
    for f, ratio in sorted(geometric, key=lambda x: x[1]):
        print(f"  {f}: a/p = {ratio:.4f}")
    
    print(f"\nInsufficient data: {len(insufficient)}")
    if insufficient:
        print(f"  {', '.join(insufficient)}")
    
    # Generate plots for functions with good data
    print("\n" + "=" * 90)
    print("GENERATING PLOTS")
    print("=" * 90)
    
    # Plot survival curves for selected functions
    funcs_to_plot = [f for f in results_1M if results_1M[f]['hits'] >= 10]
    
    if funcs_to_plot:
        n_funcs = len(funcs_to_plot)
        n_cols = 4
        n_rows = (n_funcs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        axes = axes.flatten() if n_funcs > 1 else [axes]
        
        for idx, func_name in enumerate(funcs_to_plot):
            ax = axes[idx]
            
            f_star = CEC2017_OPTIMA[func_name]
            curves = [run["curve"] for run in data_1M[func_name]]
            ev = compute_event_data(curves, f_star, EPSILON)
            km = kaplan_meier(ev.y, ev.delta)
            
            r = results_1M[func_name]
            
            ax.plot(km["t_vals"], km["S_hat"], 'b-', lw=1.5)
            
            if r['T']:
                ax.axvline(r['T'], color='red', ls='--', alpha=0.5, label=f"T={r['T']}")
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('P(τ > n)')
            ax.set_title(f"{func_name}\nHits: {r['hits']}/{r['n_runs']}, "
                        f"a/p: {r['ratio']:.3f}" if r['ratio'] else f"{func_name}\nHits: {r['hits']}/{r['n_runs']}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.05, 1.05])
        
        # Hide unused subplots
        for idx in range(len(funcs_to_plot), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        out_path = OUTPUT_DIR / "survival_curves_1M.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        plt.close()
    
    # Save results to CSV
    import csv
    csv_path = OUTPUT_DIR / "hazard_analysis_1M.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['function', 'hits', 'n_runs', 'hit_rate', 'T', 
                        'tau_min', 'tau_median', 'tau_max', 'tau_mean',
                        'p_cens', 'a_valid', 'a_over_p'])
        
        for func_name in sorted(results_1M.keys(), key=lambda x: int(x.split('_f')[1])):
            r = results_1M[func_name]
            writer.writerow([
                func_name, r['hits'], r['n_runs'], r['hit_rate'], r['T'],
                r['tau_min'], r['tau_median'], r['tau_max'], r['tau_mean'],
                r['p_cens'], r['a_valid'], r['ratio']
            ])
    
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
