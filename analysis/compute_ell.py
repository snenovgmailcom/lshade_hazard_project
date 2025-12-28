#!/usr/bin/env python3
"""
Compute ℓ = λ(I_t) for L-SHADE on CEC2017.

For sphere-like functions, I_t is computed analytically.
For general functions, I_t is estimated by sampling.

I_t(i,b,r1,r2) = {F ∈ [F_L, F_U] : f(x_i + F*d) ≤ f_min + ε}

where d = (x_b - x_i) + (x_r1 - x_r2)
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class EllResult:
    """Result of computing ℓ for one tuple."""
    ell: float              # Length of successful F-interval
    F_interval: Tuple[float, float]  # (F_low, F_high) or (nan, nan) if empty
    is_positive: bool       # Whether ℓ > 0


def compute_ell_sphere(
    x_i: np.ndarray,
    x_b: np.ndarray,
    x_r1: np.ndarray,
    x_r2: np.ndarray,
    x_star: np.ndarray,
    epsilon: float,
    F_L: float = 0.1,
    F_U: float = 1.0,
) -> EllResult:
    """
    Compute ℓ analytically for sphere function f(x) = ||x - x*||^2.
    
    The condition f(x_i + F*d) ≤ ε becomes:
    ||y + F*d||^2 ≤ ε, where y = x_i - x*
    
    This is a quadratic in F: a*F^2 + b*F + c ≤ 0
    """
    d = (x_b - x_i) + (x_r1 - x_r2)
    y = x_i - x_star
    
    a = np.dot(d, d)
    b = 2 * np.dot(y, d)
    c = np.dot(y, y) - epsilon
    
    if a < 1e-12:
        # d ≈ 0, no movement possible
        if c <= 0:
            # Already in target
            return EllResult(ell=F_U - F_L, F_interval=(F_L, F_U), is_positive=True)
        else:
            return EllResult(ell=0.0, F_interval=(np.nan, np.nan), is_positive=False)
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        # No real roots, quadratic always positive (since a > 0)
        return EllResult(ell=0.0, F_interval=(np.nan, np.nan), is_positive=False)
    
    sqrt_disc = np.sqrt(discriminant)
    F_minus = (-b - sqrt_disc) / (2*a)
    F_plus = (-b + sqrt_disc) / (2*a)
    
    # Intersection with [F_L, F_U]
    F_low = max(F_L, F_minus)
    F_high = min(F_U, F_plus)
    
    if F_low >= F_high:
        return EllResult(ell=0.0, F_interval=(np.nan, np.nan), is_positive=False)
    
    ell = F_high - F_low
    return EllResult(ell=ell, F_interval=(F_low, F_high), is_positive=True)


def compute_ell_sampling(
    x_i: np.ndarray,
    x_b: np.ndarray,
    x_r1: np.ndarray,
    x_r2: np.ndarray,
    func,
    f_star: float,
    epsilon: float,
    F_L: float = 0.1,
    F_U: float = 1.0,
    n_samples: int = 1000,
) -> EllResult:
    """
    Estimate ℓ by sampling F values.
    """
    d = (x_b - x_i) + (x_r1 - x_r2)
    
    F_vals = np.linspace(F_L, F_U, n_samples)
    target = f_star + epsilon
    
    success = np.zeros(n_samples, dtype=bool)
    for idx, F in enumerate(F_vals):
        v = x_i + F * d
        success[idx] = (func(v) <= target)
    
    # Estimate ℓ as fraction of successful F values × interval length
    ell = np.sum(success) / n_samples * (F_U - F_L)
    
    if ell > 0:
        # Find approximate interval
        successful_F = F_vals[success]
        F_low = successful_F.min()
        F_high = successful_F.max()
        return EllResult(ell=ell, F_interval=(F_low, F_high), is_positive=True)
    else:
        return EllResult(ell=0.0, F_interval=(np.nan, np.nan), is_positive=False)


def analyze_generation_ell(
    log_entry: dict,
    f_star: float,
    epsilon: float,
    x_star: Optional[np.ndarray] = None,
    func=None,
    F_L: float = 0.1,
    F_U: float = 1.0,
    n_sample_tuples: int = 1000,
    use_analytic: bool = True,
    rng=None,
) -> dict:
    """
    Analyze one generation for ℓ statistics.
    
    If use_analytic=True and x_star is provided, uses sphere formula.
    Otherwise, uses sampling with func.
    
    Samples n_sample_tuples random (i, b, r1, r2) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    pop = log_entry['population']
    fit = log_entry['fitness']
    archive = log_entry['archive']
    pbest_indices = log_entry['pbest_indices']
    
    NP = len(pop)
    n_archive = len(archive)
    total_pool = NP + n_archive
    
    # Build donor pool
    donor_pool = list(range(total_pool))
    
    def get_vector(idx):
        if idx < NP:
            return pop[idx]
        else:
            return archive[idx - NP]
    
    ell_values = []
    n_positive = 0
    n_checked = 0
    
    for _ in range(n_sample_tuples):
        # Sample random tuple (i, b, r1, r2)
        i = rng.integers(0, NP)
        b = rng.choice(pbest_indices)
        
        # r1, r2 distinct and not equal to i
        available = [j for j in donor_pool if j != i]
        if len(available) < 2:
            continue
        
        r1, r2 = rng.choice(available, size=2, replace=False)
        
        x_i = pop[i]
        x_b = pop[b]
        x_r1 = get_vector(r1)
        x_r2 = get_vector(r2)
        
        if use_analytic and x_star is not None:
            result = compute_ell_sphere(x_i, x_b, x_r1, x_r2, x_star, epsilon, F_L, F_U)
        elif func is not None:
            result = compute_ell_sampling(x_i, x_b, x_r1, x_r2, func, f_star, epsilon, F_L, F_U)
        else:
            raise ValueError("Must provide either x_star (analytic) or func (sampling)")
        
        n_checked += 1
        if result.is_positive:
            n_positive += 1
            ell_values.append(result.ell)
    
    return {
        't': log_entry['t'],
        'n_checked': n_checked,
        'n_positive': n_positive,
        'gamma_hat': n_positive / n_checked if n_checked > 0 else 0.0,
        'ell_values': ell_values,
        'ell_mean': np.mean(ell_values) if ell_values else 0.0,
        'ell_max': np.max(ell_values) if ell_values else 0.0,
        'ell_min': np.min(ell_values) if ell_values else 0.0,
        'best_fitness': log_entry['best_fitness'],
        'NP': log_entry['NP'],
    }


def analyze_all_generations(
    generation_logs: List[dict],
    f_star: float,
    epsilon: float,
    x_star: Optional[np.ndarray] = None,
    func=None,
    F_L: float = 0.1,
    F_U: float = 1.0,
    n_sample_tuples: int = 1000,
    use_analytic: bool = True,
    seed: int = 42,
) -> List[dict]:
    """
    Analyze all logged generations.
    """
    rng = np.random.default_rng(seed)
    
    results = []
    for log_entry in generation_logs:
        res = analyze_generation_ell(
            log_entry=log_entry,
            f_star=f_star,
            epsilon=epsilon,
            x_star=x_star,
            func=func,
            F_L=F_L,
            F_U=F_U,
            n_sample_tuples=n_sample_tuples,
            use_analytic=use_analytic,
            rng=rng,
        )
        results.append(res)
    
    return results


if __name__ == "__main__":
    # Test with simple sphere function
    print("Testing compute_ell_sphere...")
    
    x_star = np.zeros(10)
    epsilon = 1e-2
    
    # Case 1: Close to optimum, should have positive ℓ
    x_i = np.ones(10) * 0.05
    x_b = np.zeros(10)
    x_r1 = np.ones(10) * 0.01
    x_r2 = np.ones(10) * 0.02
    
    result = compute_ell_sphere(x_i, x_b, x_r1, x_r2, x_star, epsilon)
    print(f"Case 1 (close): ℓ = {result.ell:.6f}, interval = {result.F_interval}")
    
    # Case 2: Far from optimum, should have ℓ = 0
    x_i = np.ones(10) * 10.0
    result = compute_ell_sphere(x_i, x_b, x_r1, x_r2, x_star, epsilon)
    print(f"Case 2 (far): ℓ = {result.ell:.6f}, positive = {result.is_positive}")
    
    print("Done.")
