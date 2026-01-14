#!/usr/bin/env python3
"""
validate_morse.py

Validates eq:a-tilde-morse using the EXACT theorem formula.

Three modes:
  1. Default (best slot from G_t): 
     ã_t = (1/H) × c_pair × (g⁻_{k*} × Δ_F) × (q⁻_{k*} × η_r)
     where k* = argmax_{k ∈ G_t}[(g_k⁻ × Δ_F)(q_k⁻)]

  2. Set of good slots (--set_of_good_slots): 
     ã_t = (|G_t|/H) × c_pair × (min g⁻) × (min q⁻) × Δ_F × η_r
     where mins are over G_t

  3. Sum mode (--sum_mode, tightest):
     ã_t = (c_pair/H) × Σ_k [(g_k⁻ × Δ_F)(q_k⁻ × η_r)]

Conditions (faithful to theorem):
  (C1) x_b ∈ A_{ε/4}
  (C2) Concentration: n_clust ≥ 2 (can form donor pairs)
  (C3) Memory: |G_t| ≥ 1 (∃k with g_k⁻ ≥ g_thresh and q_k⁻ ≥ q_thresh)

Usage:
    # Default (best slot from G_t)
    python validate_morse.py --pkl experiments/D10/f1/f1.pkl --func f1 --eps 10
    
    # Generalized (|G_t|/H with min-min)
    python validate_morse.py --pkl ... --set_of_good_slots
    
    # Sum mode (tightest)
    python validate_morse.py --pkl ... --sum_mode
"""

import argparse
import json
import pickle
from typing import Dict, List, Optional, Tuple
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
# Density and probability computations
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
    
    FIXED: Returns 0.0 (conservative) when normalization fails, not 0.5.
    """
    if np.isnan(M_CR_k):
        M_CR_k = 0.5  # fallback location
    
    Z = norm.cdf(hi, M_CR_k, sigma_CR) - norm.cdf(lo, M_CR_k, sigma_CR)
    if Z <= 0:
        return 0.0  # Conservative fallback
    if c_cr <= lo:
        return 1.0
    if c_cr >= hi:
        return 0.0
    return (norm.cdf(hi, M_CR_k, sigma_CR) - norm.cdf(c_cr, M_CR_k, sigma_CR)) / Z


# =============================================================================
# Slot analysis
# =============================================================================

def analyze_all_slots(M_F: np.ndarray, M_CR: np.ndarray,
                      F_minus: float, F_plus: float, c_cr: float,
                      sigma_F: float = 0.1, sigma_CR: float = 0.1,
                      g_threshold: float = 0.01, q_threshold: float = 0.1) -> Dict:
    """
    Analyze all memory slots and compute:
    - G_t: set of good slots (pass thresholds)
    - Per-slot (g_k⁻, q_k⁻) values
    - Best slot within G_t
    - Sum of products over all slots
    """
    H = len(M_F)
    Delta_F = F_plus - F_minus
    
    slots = []
    G_t = []  # Indices of good slots
    
    for k in range(H):
        g_k = g_minus_at_slot(M_F[k], F_minus, F_plus, sigma_F)
        q_k = q_minus_at_slot(M_CR[k], c_cr, sigma_CR)
        product = (g_k * Delta_F) * q_k
        
        slots.append({
            "k": k,
            "g_k": g_k,
            "q_k": q_k,
            "product": product,
            "is_good": g_k >= g_threshold and q_k >= q_threshold
        })
        
        if slots[-1]["is_good"]:
            G_t.append(k)
    
    # Best slot within G_t (theorem-faithful)
    k_star = None
    g_star = 0.0
    q_star = 0.0
    best_product = -1
    
    for k in G_t:
        if slots[k]["product"] > best_product:
            best_product = slots[k]["product"]
            k_star = k
            g_star = slots[k]["g_k"]
            q_star = slots[k]["q_k"]
    
    # Min over G_t (for |G_t|/H mode)
    if G_t:
        g_min_Gt = min(slots[k]["g_k"] for k in G_t)
        q_min_Gt = min(slots[k]["q_k"] for k in G_t)
    else:
        g_min_Gt = 0.0
        q_min_Gt = 0.0
    
    # Sum of products over ALL slots (for sum mode)
    sum_products = sum(s["product"] for s in slots)
    
    return {
        "H": H,
        "slots": slots,
        "G_t": G_t,
        "G_t_size": len(G_t),
        # Best slot in G_t
        "k_star": k_star,
        "g_star": g_star,
        "q_star": q_star,
        # Min over G_t
        "g_min_Gt": g_min_Gt,
        "q_min_Gt": q_min_Gt,
        # Sum over all
        "sum_products": sum_products,
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
                window: int, mode: str = "best_slot",
                g_threshold: float = 0.01, q_threshold: float = 0.1) -> Dict:
    """
    Analyze one run.
    
    mode: "best_slot" | "good_slots" | "sum"
    """
    
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
        
        # (C1): f_best ≤ f* + ε/4  (x_b ∈ A_{ε/4})
        C1 = f_best_t <= f_star + eps / 4
        
        # Concentration
        best_idx = np.argmin(fit_t)
        x_best = pos_t[best_idx]
        distances = np.linalg.norm(pos_t - x_best, axis=1)
        n_clust = int(np.sum(distances <= r_conc))
        beta1 = n_clust / N_t
        c_pair = compute_c_pair(n_clust, N_t, A_t, p_best)
        
        # (C2): Concentration - need at least 2 to form donor pairs
        # FIXED: n_clust >= 2, not beta1 > 0
        C2 = n_clust >= 2
        
        # Analyze all slots
        slot_analysis = analyze_all_slots(M_F, M_CR, F_minus, F_plus, c_cr,
                                          sigma_F, sigma_CR, g_threshold, q_threshold)
        
        # (C3): ∃ memory slot k with g_k⁻ ≥ g_thresh and q_k⁻ ≥ q_thresh
        C3 = slot_analysis["G_t_size"] >= 1
        
        all_cond = C1 and C2 and C3
        
        # Compute bound based on mode
        if mode == "best_slot":
            # Use best slot from G_t (theorem-faithful)
            g_minus = slot_analysis["g_star"]
            q_minus = slot_analysis["q_star"]
            slot_factor = 1.0 / H
            a_tilde = slot_factor * c_pair * (g_minus * Delta_F) * (q_minus * eta)
            
        elif mode == "good_slots":
            # |G_t|/H with min over G_t
            g_minus = slot_analysis["g_min_Gt"]
            q_minus = slot_analysis["q_min_Gt"]
            slot_factor = slot_analysis["G_t_size"] / H
            a_tilde = slot_factor * c_pair * (g_minus * Delta_F) * (q_minus * eta)
            
        elif mode == "sum":
            # Sum of products (tightest)
            g_minus = None  # Not applicable
            q_minus = None
            slot_factor = 1.0 / H
            a_tilde = (c_pair / H) * slot_analysis["sum_products"] * eta
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # LHS: trial_fitness_best[t] ≤ f* + ε ?
        f_trial_b = trial_fitness_best[t]
        if f_trial_b is None or (isinstance(f_trial_b, float) and np.isnan(f_trial_b)):
            u_in_A_eps = None
        else:
            u_in_A_eps = float(f_trial_b) <= f_star + eps
        
        gen_data = {
            "t": t,
            "t_rel": t - tau_eps4,
            "f_best": f_best_t,
            "f_trial_b": float(f_trial_b) if f_trial_b is not None and np.isfinite(f_trial_b) else None,
            "u_in_A_eps": u_in_A_eps,
            "N_t": N_t,
            "C1": C1,  # x_b ∈ A_{ε/4}
            "C2": C2,  # n_clust ≥ 2
            "C3": C3,  # |G_t| ≥ 1
            "all_cond": all_cond,
            "mode": mode,
            "G_t_size": slot_analysis["G_t_size"],
            "k_star": slot_analysis["k_star"],
            "slot_factor": slot_factor,
            "g_minus": g_minus,
            "q_minus": q_minus,
            "g_min_Gt": slot_analysis["g_min_Gt"],
            "q_min_Gt": slot_analysis["q_min_Gt"],
            "sum_products": slot_analysis["sum_products"],
            "n_clust": n_clust,
            "beta1": beta1,
            "c_pair": c_pair,
            "eta_r": eta,
            "a_tilde": a_tilde,
        }
        
        generations.append(gen_data)
    
    return {
        "valid": True,
        "tau_eps": int(tau_eps) if np.isfinite(tau_eps) else None,
        "tau_eps4": tau_eps4,
        "generations": generations
    }


def aggregate_validation(all_runs: List[Dict], mode: str) -> Dict:
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
            "a_tilde_mean": None,
            "bound_satisfied": None,
            "note": "No conditioned samples with LHS data"
        }
    
    n_cond = len(cond_samples)
    n_success = sum(1 for s in cond_samples if s["u_in_A_eps"])
    p_empirical = n_success / n_cond
    
    # All a_tilde values (including zeros)
    a_tildes = [s["a_tilde"] for s in cond_samples]
    a_tilde_mean = float(np.mean(a_tildes))
    a_tilde_median = float(np.median(a_tildes))
    a_tilde_min = float(np.min(a_tildes))
    a_tilde_max = float(np.max(a_tildes))
    
    # FIXED: Primary comparison is mean vs mean (theorem implies E[indicator] ≥ E[bound])
    bound_satisfied = p_empirical >= a_tilde_mean
    
    # Component statistics
    c_pairs = [s["c_pair"] for s in cond_samples]
    beta1s = [s["beta1"] for s in cond_samples]
    n_clusts = [s["n_clust"] for s in cond_samples]
    N_ts = [s["N_t"] for s in cond_samples]
    G_t_sizes = [s["G_t_size"] for s in cond_samples]
    slot_factors = [s["slot_factor"] for s in cond_samples]
    
    # Mode-specific stats
    if mode == "sum":
        sum_products = [s["sum_products"] for s in cond_samples]
        g_minuses = [0.0]  # placeholder
        q_minuses = [0.0]
    else:
        g_minuses = [s["g_minus"] for s in cond_samples if s["g_minus"] is not None]
        q_minuses = [s["q_minus"] for s in cond_samples if s["q_minus"] is not None]
        sum_products = [s["sum_products"] for s in cond_samples]
    
    g_min_Gts = [s["g_min_Gt"] for s in cond_samples]
    q_min_Gts = [s["q_min_Gt"] for s in cond_samples]
    
    return {
        "n_cond_samples": n_cond,
        "n_success": n_success,
        "p_empirical": p_empirical,
        "a_tilde": {
            "mean": a_tilde_mean,
            "median": a_tilde_median,
            "min": a_tilde_min,
            "max": a_tilde_max
        },
        "ratio": p_empirical / a_tilde_mean if a_tilde_mean > 0 else None,
        "bound_satisfied": bound_satisfied,
        "components": {
            "c_pair": {
                "mean": float(np.mean(c_pairs)),
                "median": float(np.median(c_pairs)), 
                "min": float(np.min(c_pairs)), 
                "max": float(np.max(c_pairs))
            },
            "n_clust": {
                "mean": float(np.mean(n_clusts)),
                "median": float(np.median(n_clusts)),
                "min": int(np.min(n_clusts)),
                "max": int(np.max(n_clusts))
            },
            "beta1": {
                "mean": float(np.mean(beta1s)),
                "median": float(np.median(beta1s)), 
                "min": float(np.min(beta1s)), 
                "max": float(np.max(beta1s))
            },
            "N_t": {
                "median": float(np.median(N_ts)),
                "min": int(np.min(N_ts)),
                "max": int(np.max(N_ts))
            },
            "G_t_size": {
                "mean": float(np.mean(G_t_sizes)),
                "median": float(np.median(G_t_sizes)),
                "min": int(np.min(G_t_sizes)),
                "max": int(np.max(G_t_sizes))
            },
            "slot_factor": {
                "mean": float(np.mean(slot_factors)),
                "median": float(np.median(slot_factors)),
                "min": float(np.min(slot_factors)),
                "max": float(np.max(slot_factors))
            },
            "g_minus": {
                "mean": float(np.mean(g_minuses)) if g_minuses else 0.0,
                "median": float(np.median(g_minuses)) if g_minuses else 0.0,
                "min": float(np.min(g_minuses)) if g_minuses else 0.0,
                "max": float(np.max(g_minuses)) if g_minuses else 0.0
            },
            "q_minus": {
                "mean": float(np.mean(q_minuses)) if q_minuses else 0.0,
                "median": float(np.median(q_minuses)) if q_minuses else 0.0,
                "min": float(np.min(q_minuses)) if q_minuses else 0.0,
                "max": float(np.max(q_minuses)) if q_minuses else 0.0
            },
            "g_min_Gt": {
                "mean": float(np.mean(g_min_Gts)),
                "median": float(np.median(g_min_Gts)),
                "min": float(np.min(g_min_Gts)),
                "max": float(np.max(g_min_Gts))
            },
            "q_min_Gt": {
                "mean": float(np.mean(q_min_Gts)),
                "median": float(np.median(q_min_Gts)),
                "min": float(np.min(q_min_Gts)),
                "max": float(np.max(q_min_Gts))
            },
            "sum_products": {
                "mean": float(np.mean(sum_products)),
                "median": float(np.median(sum_products)),
                "min": float(np.min(sum_products)),
                "max": float(np.max(sum_products))
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate Morse theorem bound",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Default:           (1/H) × c_pair × (g⁻_{k*} Δ_F) × (q⁻_{k*} η_r)
                     where k* = argmax over G_t
  --set_of_good_slots: (|G_t|/H) × c_pair × (min g⁻) × (min q⁻) × Δ_F × η_r
  --sum_mode:        (c_pair/H) × Σ_k [(g_k⁻ Δ_F)(q_k⁻ η_r)]  [tightest]

Examples:
  python validate_morse.py --pkl experiments/D10/f1/f1.pkl --func f1 --eps 10
  python validate_morse.py --pkl ... --set_of_good_slots
  python validate_morse.py --pkl ... --sum_mode
        """
    )
    parser.add_argument("--pkl", required=True, help="Path to PKL file")
    parser.add_argument("--func", default="f1", help="Function name (e.g., f1, f5)")
    parser.add_argument("--eps", type=float, default=10.0, help="Tolerance ε")
    parser.add_argument("--window", type=int, default=20, help="Window size after τ_{ε/4}")
    parser.add_argument("--dim", type=int, default=10, help="Dimension")
    parser.add_argument("--H", type=int, default=6, help="Memory size")
    parser.add_argument("--c_cr", type=float, default=0.5, help="Crossover threshold")
    parser.add_argument("--r_conc", type=float, default=1.0, help="Concentration radius")
    parser.add_argument("--F_minus", type=float, default=0.1, help="F lower bound")
    parser.add_argument("--F_plus", type=float, default=0.9, help="F upper bound")
    parser.add_argument("--sigma_F", type=float, default=0.1, help="Cauchy scale for F")
    parser.add_argument("--sigma_CR", type=float, default=0.1, help="Normal std for CR")
    parser.add_argument("--p_best", type=float, default=0.11, help="p-best rate")
    parser.add_argument("--output", default="morse_validation.json", help="Output JSON file")
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--set_of_good_slots", action="store_true",
                            help="Use |G_t|/H × min-min bound")
    mode_group.add_argument("--sum_mode", action="store_true",
                            help="Use sum-of-products bound (tightest)")
    
    # Thresholds for G_t
    parser.add_argument("--g_threshold", type=float, default=0.01,
                        help="Threshold for g_k⁻ to include slot in G_t")
    parser.add_argument("--q_threshold", type=float, default=0.1,
                        help="Threshold for q_k⁻ to include slot in G_t")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.sum_mode:
        mode = "sum"
    elif args.set_of_good_slots:
        mode = "good_slots"
    else:
        mode = "best_slot"
    
    print(f"Loading {args.pkl}...")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    
    func_key = args.func if args.func.startswith("cec2017_") else f"cec2017_{args.func}"
    runs = data[func_key]
    f_star = f_star_from_func(args.func)
    
    Delta_F = args.F_plus - args.F_minus
    
    # Mode description
    if mode == "best_slot":
        method_str = "best_slot from G_t (1/H factor)"
        formula = "ã_t = (1/H) × c_pair × (g⁻_{k*} × Δ_F) × (q⁻_{k*} × η_r)"
    elif mode == "good_slots":
        method_str = f"|G_t|/H × min-min (g≥{args.g_threshold}, q≥{args.q_threshold})"
        formula = "ã_t = (|G_t|/H) × c_pair × (min g⁻) × Δ_F × (min q⁻) × η_r"
    else:  # sum
        method_str = "sum-of-products (tightest)"
        formula = "ã_t = (c_pair/H) × Σ_k [(g_k⁻ Δ_F)(q_k⁻ η_r)]"
    
    print(f"\nValidating Morse bound for {func_key}")
    print(f"  Mode: {method_str}")
    print(f"  Formula: {formula}")
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
            args.sigma_F, args.sigma_CR, args.p_best, args.window,
            mode=mode,
            g_threshold=args.g_threshold,
            q_threshold=args.q_threshold
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
    val = aggregate_validation(all_runs, mode)
    
    # Get η_r for display
    r = (args.dim - 1) // 2
    eta = eta_r(args.dim, args.c_cr, r)
    
    print(f"\n{'='*70}")
    print(f"THEOREM VALIDATION")
    print(f"  Mode: {method_str}")
    print(f"{'='*70}")
    
    print(f"\nConditions (theorem-faithful):")
    print(f"  (C1) x_b ∈ A_{{ε/4}}: f_best ≤ f* + ε/4")
    print(f"  (C2) Concentration: n_clust ≥ 2 (can form donor pairs)")
    print(f"  (C3) Memory slot: |G_t| ≥ 1 (∃k with g_k⁻ ≥ {args.g_threshold}, q_k⁻ ≥ {args.q_threshold})")
    
    print(f"\nConditioned (run, t) pairs: {val['n_cond_samples']}")
    
    if val['n_cond_samples'] > 0:
        print(f"\nLHS: E[𝟙{{u_{{t,b}} ∈ A_ε}} | C1∧C2∧C3]")
        print(f"  Successes: {val['n_success']}/{val['n_cond_samples']}")
        print(f"  P_empirical = {val['p_empirical']:.4f}")
        
        print(f"\nRHS: E[ã_t | C1∧C2∧C3]")
        print(f"  ã_t mean   = {val['a_tilde']['mean']:.6f}  ← primary comparison")
        print(f"  ã_t median = {val['a_tilde']['median']:.6f}")
        print(f"  ã_t range  = [{val['a_tilde']['min']:.6f}, {val['a_tilde']['max']:.6f}]")
        
        if val['ratio'] is not None:
            print(f"\nRatio P_emp / E[ã_t] = {val['ratio']:.2f}")
        
        status = "✓ BOUND SATISFIED" if val['bound_satisfied'] else "✗ BOUND VIOLATED"
        print(f"\nStatus: {status}")
        print(f"  (Checking: P_empirical ≥ E[ã_t])")
        
        print(f"\nComponent statistics:")
        print(f"  N_t:         median={val['components']['N_t']['median']:.0f}, "
              f"range=[{val['components']['N_t']['min']}, {val['components']['N_t']['max']}]")
        print(f"  n_clust:     mean={val['components']['n_clust']['mean']:.1f}, "
              f"median={val['components']['n_clust']['median']:.0f}, "
              f"range=[{val['components']['n_clust']['min']}, {val['components']['n_clust']['max']}]")
        print(f"  β₁:          mean={val['components']['beta1']['mean']:.4f}, "
              f"range=[{val['components']['beta1']['min']:.4f}, {val['components']['beta1']['max']:.4f}]")
        print(f"  c_pair:      mean={val['components']['c_pair']['mean']:.4f}, "
              f"range=[{val['components']['c_pair']['min']:.4f}, {val['components']['c_pair']['max']:.4f}]")
        print(f"  |G_t|:       mean={val['components']['G_t_size']['mean']:.1f}, "
              f"range=[{val['components']['G_t_size']['min']}, {val['components']['G_t_size']['max']}]")
        
        if mode == "best_slot":
            print(f"  g⁻_{{k*}}:    mean={val['components']['g_minus']['mean']:.4f}, "
                  f"range=[{val['components']['g_minus']['min']:.4f}, {val['components']['g_minus']['max']:.4f}]")
            print(f"  q⁻_{{k*}}:    mean={val['components']['q_minus']['mean']:.4f}, "
                  f"range=[{val['components']['q_minus']['min']:.4f}, {val['components']['q_minus']['max']:.4f}]")
        elif mode == "good_slots":
            print(f"  min g⁻:      mean={val['components']['g_min_Gt']['mean']:.4f}, "
                  f"range=[{val['components']['g_min_Gt']['min']:.4f}, {val['components']['g_min_Gt']['max']:.4f}]")
            print(f"  min q⁻:      mean={val['components']['q_min_Gt']['mean']:.4f}, "
                  f"range=[{val['components']['q_min_Gt']['min']:.4f}, {val['components']['q_min_Gt']['max']:.4f}]")
        else:  # sum
            print(f"  Σ_k[prod]:   mean={val['components']['sum_products']['mean']:.4f}, "
                  f"range=[{val['components']['sum_products']['min']:.4f}, {val['components']['sum_products']['max']:.4f}]")
        
        print(f"  η_r:         {eta:.4f} (d={args.dim}, c_cr={args.c_cr})")
        
        # Bound decomposition
        print(f"\nBound decomposition (means):")
        c = val['components']['c_pair']['mean']
        
        if mode == "best_slot":
            sf = 1.0 / args.H
            g = val['components']['g_minus']['mean']
            q = val['components']['q_minus']['mean']
            print(f"  (1/H)        = 1/{args.H} = {sf:.6f}")
            print(f"  c_pair       = {c:.6f}")
            print(f"  (g⁻ × Δ_F)   = {g:.4f} × {Delta_F} = {g*Delta_F:.6f}")
            print(f"  (q⁻ × η_r)   = {q:.4f} × {eta:.4f} = {q*eta:.6f}")
            print(f"  Product      = {sf * c * (g*Delta_F) * (q*eta):.6f}")
            
        elif mode == "good_slots":
            sf = val['components']['slot_factor']['mean']
            g = val['components']['g_min_Gt']['mean']
            q = val['components']['q_min_Gt']['mean']
            G_size = val['components']['G_t_size']['mean']
            print(f"  (|G_t|/H)    = {G_size:.1f}/{args.H} = {sf:.6f}")
            print(f"  c_pair       = {c:.6f}")
            print(f"  (min g⁻×Δ_F) = {g:.4f} × {Delta_F} = {g*Delta_F:.6f}")
            print(f"  (min q⁻×η_r) = {q:.4f} × {eta:.4f} = {q*eta:.6f}")
            print(f"  Product      = {sf * c * (g*Delta_F) * (q*eta):.6f}")
            
        else:  # sum
            sf = 1.0 / args.H
            sum_p = val['components']['sum_products']['mean']
            print(f"  (1/H)        = 1/{args.H} = {sf:.6f}")
            print(f"  c_pair       = {c:.6f}")
            print(f"  Σ_k[g_k⁻Δ_F × q_k⁻] = {sum_p:.6f}")
            print(f"  η_r          = {eta:.4f}")
            print(f"  Product      = {sf * c * sum_p * eta:.6f}")
    else:
        print(f"  {val.get('note', 'No data')}")
    
    # Save results
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
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
            "mode": mode,
            "g_threshold": args.g_threshold,
            "q_threshold": args.q_threshold,
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
