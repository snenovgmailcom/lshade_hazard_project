#!/usr/bin/env python3
"""
validate_morse_theorem.py

Validates eq:a-tilde-morse using the EXACT theorem formula:

    ã_t = (c_pair / H) × (g⁻ × Δ_F) × (q⁻ × η_r)

where g⁻ = min_{F ∈ [F⁻, F⁺]} g_k^F(F) is the density lower bound,
NOT the interval probability p_F = P(F ∈ [F⁻, F⁺]).

Since p_F ≥ g⁻ × Δ_F, the theorem bound is more conservative.

Usage:
    python validate_morse_theorem.py \
        --pkl experiments/D10/raw_results_lshade.pkl \
        --func f1 --eps 10 --window 20 \
        --output morse_theorem_f1.json
"""

import argparse
import json
import pickle
from typing import Dict, List, Optional
import numpy as np
from scipy.stats import cauchy, norm, binom


def f_star_from_func(func: str) -> float:
    func = func.strip().lower()
    if func.startswith("cec2017_"):
        func = func.replace("cec2017_", "")
    if func.startswith("f") and func[1:].isdigit():
        return float(100 * int(func[1:]))
    raise ValueError(f"Cannot parse f* from func={func}")


def compute_hitting_time(curve: np.ndarray, threshold: float) -> float:
    hit_idx = np.where(curve <= threshold)[0]
    return float(hit_idx[0]) if len(hit_idx) > 0 else np.inf


def eta_r(d: int, c_cr: float, r: Optional[int] = None) -> float:
    """η_r(d, c_cr) = P(Bin(d-1, c_cr) ≥ d - r - 1)"""
    if r is None:
        r = (d - 1) // 2
    k_min = d - r - 1
    return float(1 - binom.cdf(k_min - 1, d - 1, c_cr))


# =============================================================================
# CORRECTED: Use g⁻ × Δ_F (density bound), not p_F (probability)
# =============================================================================

def truncated_cauchy_density(F: float, loc: float, scale: float,
                              lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Density of Cauchy(loc, scale) truncated to [lo, hi] at point F.
    g(F) = cauchy.pdf(F) / Z  where Z = P(lo < X < hi)
    """
    if F <= lo or F >= hi:
        return 0.0
    Z = cauchy.cdf(hi, loc, scale) - cauchy.cdf(lo, loc, scale)
    if Z <= 0:
        return 0.0
    return cauchy.pdf(F, loc, scale) / Z


def g_minus_at_slot(M_F_k: float, F_minus: float, F_plus: float,
                    scale: float = 0.1, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    g_k⁻ = min_{F ∈ [F⁻, F⁺]} g_k^F(F)
    
    For unimodal Cauchy, minimum is at the boundary furthest from mode.
    """
    g_at_Fminus = truncated_cauchy_density(F_minus, M_F_k, scale, lo, hi)
    g_at_Fplus = truncated_cauchy_density(F_plus, M_F_k, scale, lo, hi)
    return min(g_at_Fminus, g_at_Fplus)


def q_minus_at_slot(M_CR_k: float, c_cr: float, sigma_CR: float = 0.1,
                    lo: float = 0.0, hi: float = 1.0) -> float:
    """
    q_k⁻ = P(CR ≥ c_cr | K = k) for truncated Normal(M_CR[k], σ_CR) on [lo, hi]
    """
    if np.isnan(M_CR_k):
        M_CR_k = 0.5  # fallback
    
    Z = norm.cdf(hi, M_CR_k, sigma_CR) - norm.cdf(lo, M_CR_k, sigma_CR)
    if Z <= 0:
        return 0.5
    if c_cr <= lo:
        return 1.0
    if c_cr >= hi:
        return 0.0
    return (norm.cdf(hi, M_CR_k, sigma_CR) - norm.cdf(c_cr, M_CR_k, sigma_CR)) / Z


def find_best_slot_theorem(M_F: np.ndarray, M_CR: np.ndarray,
                           F_minus: float, F_plus: float, c_cr: float,
                           sigma_F: float = 0.1, sigma_CR: float = 0.1) -> Dict:
    """
    Find best slot k* using theorem quantities:
    k* = argmax_k (g_k⁻ × q_k⁻)
    
    Returns g⁻, q⁻, and their product for the best slot.
    """
    H = len(M_F)
    Delta_F = F_plus - F_minus
    
    best_product = -1
    k_star = None
    g_minus_star = 0.0
    q_minus_star = 0.0
    
    for k in range(H):
        g_k = g_minus_at_slot(M_F[k], F_minus, F_plus, sigma_F)
        q_k = q_minus_at_slot(M_CR[k], c_cr, sigma_CR)
        
        # Product for slot selection: (g⁻ × Δ_F) × q⁻
        product = (g_k * Delta_F) * q_k
        
        if product > best_product:
            best_product = product
            k_star = k
            g_minus_star = g_k
            q_minus_star = q_k
    
    return {
        "k_star": k_star,
        "g_minus": g_minus_star,
        "q_minus": q_minus_star,
        "g_minus_Delta_F": g_minus_star * Delta_F,
        "product": best_product
    }


def compute_c_pair(n_clust: int, N: int, A: int, p_best: float = 0.11) -> float:
    """Conservative c_pair proxy."""
    s1 = max(int(np.ceil(p_best * N)), 1)
    s2 = N + A
    if s1 <= 0 or s2 <= 1 or n_clust < 2:
        return 0.0
    n_good_pairs = n_clust * (n_clust - 1)
    n_total_pairs = s1 * (s2 - 1)
    return min(n_good_pairs / n_total_pairs, 1.0) if n_total_pairs > 0 else 0.0


# =============================================================================
# Main analysis
# =============================================================================

def analyze_run(run: Dict, f_star: float, eps: float, 
                d: int, H: int, c_cr: float, r_conc: float,
                F_minus: float, F_plus: float,
                sigma_F: float, sigma_CR: float, p_best: float,
                window: int) -> Dict:
    """Analyze one run using exact theorem formula."""
    
    curve = np.asarray(run["curve"], dtype=float)
    hist = run.get("history", {})
    
    tau_eps4 = compute_hitting_time(curve, f_star + eps / 4)
    tau_eps = compute_hitting_time(curve, f_star + eps)
    
    if not np.isfinite(tau_eps4):
        return {"valid": False, "reason": "never hit A_{eps/4}"}
    
    tau_eps4 = int(tau_eps4)
    
    memory_f = hist.get("memory_f", [])
    memory_cr = hist.get("memory_cr", [])
    positions = hist.get("positions", [])
    fitness = hist.get("fitness", [])
    archive_size = hist.get("archive_size", [])
    trial_fitness_best = hist.get("trial_fitness_best", [])
    
    if not trial_fitness_best:
        return {"valid": False, "reason": "trial_fitness_best not logged"}
    
    Delta_F = F_plus - F_minus
    r = (d - 1) // 2
    eta = eta_r(d, c_cr, r)
    
    generations = []
    t_start = tau_eps4
    t_end = min(tau_eps4 + window, len(curve) - 1, len(trial_fitness_best) - 1)
    
    for t in range(t_start, t_end + 1):
        if t >= len(memory_f) or t >= len(positions) or t >= len(trial_fitness_best):
            continue
        
        M_F = np.array(memory_f[t], dtype=float)
        M_CR = np.array(memory_cr[t], dtype=float)
        
        if len(M_F) != H:
            continue
        
        pos_t = positions[t]
        fit_t = fitness[t]
        
        if pos_t is None or fit_t is None:
            continue
        
        pos_t = np.asarray(pos_t, dtype=float)
        fit_t = np.asarray(fit_t, dtype=float)
        N_t = len(pos_t)
        
        if N_t == 0:
            continue
        
        A_t = int(archive_size[t]) if t < len(archive_size) else 0
        f_best_t = float(curve[t])
        
        # (C1): f_best ≤ f* + ε/4
        C1 = f_best_t <= f_star + eps / 4
        
        # Find best slot using theorem quantities
        slot = find_best_slot_theorem(M_F, M_CR, F_minus, F_plus, c_cr, 
                                       sigma_F, sigma_CR)
        
        # (C2): g⁻ > 0 (density bound exists)
        C2 = slot["g_minus"] > 0.001
        # (C3): q⁻ > threshold
        C3 = slot["q_minus"] > 0.1
        
        all_cond = C1 and C2 and C3
        
        # Concentration for c_pair
        best_idx = np.argmin(fit_t)
        x_best = pos_t[best_idx]
        distances = np.linalg.norm(pos_t - x_best, axis=1)
        n_clust = int(np.sum(distances <= r_conc))
        beta1 = n_clust / N_t
        
        c_pair = compute_c_pair(n_clust, N_t, A_t, p_best)
        
        # THEOREM BOUND: ã_t = (c_pair/H) × (g⁻ × Δ_F) × (q⁻ × η_r)
        g_minus = slot["g_minus"]
        q_minus = slot["q_minus"]
        
        a_tilde = (c_pair / H) * (g_minus * Delta_F) * (q_minus * eta)
        
        # LHS: trial_fitness_best[t] ≤ f* + ε ?
        f_trial_b = trial_fitness_best[t]
        if f_trial_b is None or (isinstance(f_trial_b, float) and np.isnan(f_trial_b)):
            u_in_A_eps = None
        else:
            u_in_A_eps = float(f_trial_b) <= f_star + eps
        
        generations.append({
            "t": t,
            "t_rel": t - tau_eps4,
            "f_best": f_best_t,
            "f_trial_b": float(f_trial_b) if f_trial_b is not None and np.isfinite(f_trial_b) else None,
            "u_in_A_eps": u_in_A_eps,
            "N_t": N_t,
            "C1": C1,
            "C2": C2,
            "C3": C3,
            "all_cond": all_cond,
            "k_star": slot["k_star"],
            "g_minus": g_minus,
            "q_minus": q_minus,
            "g_minus_Delta_F": g_minus * Delta_F,
            "beta1": beta1,
            "c_pair": c_pair,
            "eta_r": eta,
            "a_tilde": a_tilde,
        })
    
    return {
        "valid": True,
        "tau_eps": int(tau_eps) if np.isfinite(tau_eps) else None,
        "tau_eps4": tau_eps4,
        "generations": generations
    }


def aggregate_validation(all_runs: List[Dict], Delta_F: float) -> Dict:
    """Aggregate across runs for conditional validation."""
    
    cond_samples = []
    
    for run_data in all_runs:
        if not run_data.get("valid"):
            continue
        for gen in run_data["generations"]:
            if gen["all_cond"] and gen["u_in_A_eps"] is not None:
                cond_samples.append(gen)
    
    if not cond_samples:
        return {
            "n_cond_samples": 0,
            "p_empirical": None,
            "a_tilde_median": None,
            "bound_satisfied": None,
            "note": "No conditioned samples with LHS data"
        }
    
    n_cond = len(cond_samples)
    n_success = sum(1 for s in cond_samples if s["u_in_A_eps"])
    p_empirical = n_success / n_cond
    
    a_tildes = [s["a_tilde"] for s in cond_samples if s["a_tilde"] > 0]
    a_tilde_median = float(np.median(a_tildes)) if a_tildes else 0.0
    a_tilde_mean = float(np.mean(a_tildes)) if a_tildes else 0.0
    a_tilde_min = float(np.min(a_tildes)) if a_tildes else 0.0
    a_tilde_max = float(np.max(a_tildes)) if a_tildes else 0.0
    
    # Component statistics
    c_pairs = [s["c_pair"] for s in cond_samples]
    g_minuses = [s["g_minus"] for s in cond_samples]
    g_Delta_Fs = [s["g_minus_Delta_F"] for s in cond_samples]
    q_minuses = [s["q_minus"] for s in cond_samples]
    beta1s = [s["beta1"] for s in cond_samples]
    
    bound_satisfied = p_empirical >= a_tilde_median
    
    return {
        "n_cond_samples": n_cond,
        "n_success": n_success,
        "p_empirical": p_empirical,
        "a_tilde": {
            "median": a_tilde_median,
            "mean": a_tilde_mean,
            "min": a_tilde_min,
            "max": a_tilde_max
        },
        "ratio": p_empirical / a_tilde_median if a_tilde_median > 0 else None,
        "bound_satisfied": bound_satisfied,
        "components": {
            "c_pair": {
                "median": float(np.median(c_pairs)), 
                "min": float(np.min(c_pairs)), 
                "max": float(np.max(c_pairs))
            },
            "g_minus": {
                "median": float(np.median(g_minuses)), 
                "min": float(np.min(g_minuses)), 
                "max": float(np.max(g_minuses))
            },
            "g_minus_Delta_F": {
                "median": float(np.median(g_Delta_Fs)), 
                "min": float(np.min(g_Delta_Fs)), 
                "max": float(np.max(g_Delta_Fs))
            },
            "q_minus": {
                "median": float(np.median(q_minuses)), 
                "min": float(np.min(q_minuses)), 
                "max": float(np.max(q_minuses))
            },
            "beta1": {
                "median": float(np.median(beta1s)), 
                "min": float(np.min(beta1s)), 
                "max": float(np.max(beta1s))
            },
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Validate exact Morse theorem")
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--func", default="f1")
    parser.add_argument("--eps", type=float, default=10.0)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--H", type=int, default=6)
    parser.add_argument("--c_cr", type=float, default=0.5)
    parser.add_argument("--r_conc", type=float, default=1.0)
    parser.add_argument("--F_minus", type=float, default=0.1)
    parser.add_argument("--F_plus", type=float, default=0.9)
    parser.add_argument("--sigma_F", type=float, default=0.1)
    parser.add_argument("--sigma_CR", type=float, default=0.1)
    parser.add_argument("--p_best", type=float, default=0.11)
    parser.add_argument("--output", default="morse_theorem.json")
    args = parser.parse_args()
    
    print(f"Loading {args.pkl}...")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    
    func_key = args.func if args.func.startswith("cec2017_") else f"cec2017_{args.func}"
    runs = data[func_key]
    f_star = f_star_from_func(args.func)
    
    Delta_F = args.F_plus - args.F_minus
    
    print(f"\nValidating EXACT eq:a-tilde-morse for {func_key}")
    print(f"  ã_t = (c_pair/H) × (g⁻ × Δ_F) × (q⁻ × η_r)")
    print(f"  ε = {args.eps}, ε/4 = {args.eps/4}")
    print(f"  [F⁻, F⁺] = [{args.F_minus}, {args.F_plus}], Δ_F = {Delta_F}")
    print(f"  Window: [τ_{{ε/4}}, τ_{{ε/4}} + {args.window}]")
    print(f"  r_conc = {args.r_conc}")
    
    all_runs = []
    tau_eps4_list = []
    tau_eps_list = []
    
    for i, run in enumerate(runs):
        run_data = analyze_run(
            run, f_star, args.eps, args.dim, args.H,
            args.c_cr, args.r_conc, args.F_minus, args.F_plus,
            args.sigma_F, args.sigma_CR, args.p_best, args.window
        )
        all_runs.append(run_data)
        
        if run_data.get("valid"):
            tau_eps4_list.append(run_data["tau_eps4"])
            if run_data["tau_eps"] is not None:
                tau_eps_list.append(run_data["tau_eps"])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(runs)} runs...")
    
    n_valid = sum(1 for r in all_runs if r.get("valid"))
    print(f"\nValid runs: {n_valid}/{len(runs)}")
    
    if tau_eps_list:
        print(f"τ_ε (hit A_ε): median={np.median(tau_eps_list):.0f}")
    if tau_eps4_list:
        print(f"τ_{{ε/4}} (hit A_{{ε/4}}): median={np.median(tau_eps4_list):.0f}")
    
    # Aggregate
    val = aggregate_validation(all_runs, Delta_F)
    
    # Get η_r for display
    r = (args.dim - 1) // 2
    eta = eta_r(args.dim, args.c_cr, r)
    
    print(f"\n{'='*70}")
    print("THEOREM VALIDATION: ã_t = (c_pair/H) × (g⁻ Δ_F) × (q⁻ η_r)")
    print(f"{'='*70}")
    print(f"Conditioned (run, t) pairs: {val['n_cond_samples']}")
    
    if val['n_cond_samples'] > 0:
        print(f"\nLHS: P(u_{{t,b}} ∈ A_ε | C1∧C2∧C3)")
        print(f"  Successes: {val['n_success']}/{val['n_cond_samples']}")
        print(f"  P_empirical = {val['p_empirical']:.4f}")
        
        print(f"\nRHS: ã_t (exact theorem formula)")
        print(f"  ã_t median = {val['a_tilde']['median']:.6f}")
        print(f"  ã_t mean   = {val['a_tilde']['mean']:.6f}")
        print(f"  ã_t range  = [{val['a_tilde']['min']:.6f}, {val['a_tilde']['max']:.6f}]")
        
        if val['ratio'] is not None:
            print(f"\nRatio P_emp / ã_t = {val['ratio']:.2f}")
        
        status = "✓ BOUND SATISFIED" if val['bound_satisfied'] else "✗ BOUND VIOLATED"
        print(f"\nStatus: {status}")
        
        print(f"\nComponent statistics (theorem quantities):")
        print(f"  c_pair:      median={val['components']['c_pair']['median']:.4f}, "
              f"range=[{val['components']['c_pair']['min']:.4f}, {val['components']['c_pair']['max']:.4f}]")
        print(f"  g⁻:          median={val['components']['g_minus']['median']:.4f}, "
              f"range=[{val['components']['g_minus']['min']:.4f}, {val['components']['g_minus']['max']:.4f}]")
        print(f"  g⁻ × Δ_F:    median={val['components']['g_minus_Delta_F']['median']:.4f}, "
              f"range=[{val['components']['g_minus_Delta_F']['min']:.4f}, {val['components']['g_minus_Delta_F']['max']:.4f}]")
        print(f"  q⁻:          median={val['components']['q_minus']['median']:.4f}, "
              f"range=[{val['components']['q_minus']['min']:.4f}, {val['components']['q_minus']['max']:.4f}]")
        print(f"  η_r:         {eta:.4f} (d={args.dim}, c_cr={args.c_cr})")
        print(f"  β₁:          median={val['components']['beta1']['median']:.4f}, "
              f"range=[{val['components']['beta1']['min']:.4f}, {val['components']['beta1']['max']:.4f}]")
        
        # Show bound decomposition
        print(f"\nBound decomposition (medians):")
        c = val['components']['c_pair']['median']
        g = val['components']['g_minus_Delta_F']['median']
        q = val['components']['q_minus']['median']
        print(f"  (c_pair/H) = {c}/{args.H} = {c/args.H:.6f}")
        print(f"  (g⁻ × Δ_F) = {g:.6f}")
        print(f"  (q⁻ × η_r) = {q:.4f} × {eta:.4f} = {q*eta:.6f}")
        print(f"  Product    = {(c/args.H) * g * (q*eta):.6f}")
    else:
        print(f"  {val.get('note', 'No data')}")
    
    # Save
    results = {
        "meta": {
            "func": args.func,
            "f_star": f_star,
            "eps": args.eps,
            "eps_quarter": args.eps / 4,
            "F_minus": args.F_minus,
            "F_plus": args.F_plus,
            "Delta_F": Delta_F,
            "window": args.window,
            "dim": args.dim,
            "H": args.H,
            "c_cr": args.c_cr,
            "eta_r": eta,
            "r_conc": args.r_conc,
            "n_runs": len(runs),
            "n_valid": n_valid,
            "formula": "a_tilde = (c_pair/H) × (g_minus × Delta_F) × (q_minus × eta_r)"
        },
        "hitting_times": {
            "tau_eps_median": float(np.median(tau_eps_list)) if tau_eps_list else None,
            "tau_eps4_median": float(np.median(tau_eps4_list)) if tau_eps4_list else None,
        },
        "validation": val
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
