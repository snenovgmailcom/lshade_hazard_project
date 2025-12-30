#!/usr/bin/env python3
"""
Gap analysis: comparing theoretical bounds to empirical hazard rates.

This module computes the gap factor between:
- Theoretical hazard floor: a_t * gamma_t
- Empirical hazard: p_hat from KM analysis

It also provides plug-in tail certificates using the theoretical bound
with empirically estimated gamma_t values.
"""

import numpy as np
from typing import Dict, List, Any, Optional


def compute_gap_analysis(
    km_stats: Dict[str, Any],
    gamma_stats: Dict[str, Any],
    theoretical: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compute gap between theoretical and empirical hazard.
    
    Args:
        km_stats: Output from compute_km_statistics()
        gamma_stats: Output from estimate_gamma()
        theoretical: Output from compute_theoretical_a_t()
    
    Returns:
        Dictionary with gap analysis results
    """
    a_t = theoretical['a_t']
    gamma_mean = gamma_stats.get('gamma_mean')
    p_hat = km_stats.get('p_hat')
    
    if gamma_mean is None or p_hat is None or a_t == 0:
        return {
            'a_t': a_t,
            'gamma_mean': gamma_mean,
            'predicted_hazard': None,
            'empirical_hazard': p_hat,
            'gap_factor': None,
            'error': 'Missing data for gap computation'
        }
    
    predicted_hazard = a_t * gamma_mean
    gap_factor = p_hat / predicted_hazard if predicted_hazard > 0 else np.inf
    
    return {
        'a_t': a_t,
        'gamma_mean': gamma_mean,
        'predicted_hazard': predicted_hazard,
        'empirical_hazard': p_hat,
        'gap_factor': gap_factor,
        'log10_gap': np.log10(gap_factor) if gap_factor > 0 and np.isfinite(gap_factor) else None,
    }


def analyze_gap_sources(
    theoretical: Dict[str, float],
    d: int,
    n_runs: int = 51
) -> Dict[str, Any]:
    """
    Analyze sources of the gap between theory and practice.
    
    The theoretical bound uses only ONE witness configuration.
    This function estimates how much tighter the bound would be
    if we summed over all configurations.
    
    Args:
        theoretical: Output from compute_theoretical_a_t()
        d: Dimension
        n_runs: Number of runs
    
    Returns:
        Dictionary with gap source analysis
    """
    H = theoretical['H']
    m_t = theoretical['m_t']
    s1 = theoretical['s1']
    s2 = theoretical['s2']
    N_init = theoretical['N_init']
    
    # Number of possible witness configurations
    # For each individual i: choose b from m_t, r1 from s1, r2 from s2-1, k from H
    n_configs_per_individual = m_t * s1 * (s2 - 1) * H
    n_total_configs = N_init * n_configs_per_individual
    
    # If we sum over all configs (union bound), we gain factor of N_init
    # But this is an upper bound; actual improvement depends on correlations
    
    # Combinatorial factor in a_t
    combinatorial_factor = (1/H) * (1/m_t) * (1/(s1 * (s2 - 1)))
    
    return {
        'n_configs_per_individual': n_configs_per_individual,
        'n_total_configs': n_total_configs,
        'combinatorial_factor': combinatorial_factor,
        'potential_improvement_factor': N_init,  # Upper bound on improvement from summing
        'explanation': {
            'single_config': 'Bound uses worst-case over all (i, b, r1, r2, k)',
            'density_bound': f'Uses g^- = {theoretical["g_minus"]} uniformly over [F^-, F^+]',
            'L1_unknown': 'Condition (L1) not verified - lambda(I_t) >= Delta_F assumed',
            'crossover_margin': 'Uses epsilon/2 margin for crossover stability',
        }
    }


def format_gap_table_row(
    fname: str,
    d: int,
    eps: float,
    km_stats: Dict,
    gamma_stats: Dict,
    gap_analysis: Dict
) -> Dict[str, Any]:
    """
    Format a single row for the gap analysis table.
    
    Args:
        fname: Function name
        d: Dimension  
        eps: Epsilon tolerance
        km_stats: KM statistics
        gamma_stats: Gamma estimation results
        gap_analysis: Gap analysis results
    
    Returns:
        Dictionary suitable for DataFrame row
    """
    return {
        'function': fname,
        'd': d,
        'eps': eps,
        'hits': f"{km_stats['n_hits']}/{km_stats['n_runs']}",
        'p_hat': km_stats.get('p_hat'),
        'gamma_mean': gamma_stats.get('gamma_mean'),
        'a_t': gap_analysis.get('a_t'),
        'a_t_gamma': gap_analysis.get('predicted_hazard'),
        'gap_factor': gap_analysis.get('gap_factor'),
        'log10_gap': gap_analysis.get('log10_gap'),
        'clustering': km_stats.get('clustering'),
    }


# =============================================================================
# Plug-in Tail Certificates
# =============================================================================

def compute_plugin_survival_bound(
    a_t: float,
    gamma_t: np.ndarray,
    P_E0c: float = 1.0
) -> np.ndarray:
    """
    Compute plug-in witness-based survival upper bound.
    
    From equation (S-witness):
        S_L(n) = P(E_0^c) * exp(-sum_{t=1}^{n} a_t * gamma_t)
    
    Args:
        a_t: Theoretical hazard floor (constant per generation)
        gamma_t: Array of estimated gamma values (one per generation)
        P_E0c: Probability of favorable initialization (default 1.0)
    
    Returns:
        Array of S_L(n) values for n = 1, ..., len(gamma_t)
    """
    # Replace NaN with 0 (conservative: assume no witness if unknown)
    gamma_clean = np.where(np.isnan(gamma_t), 0.0, gamma_t)
    
    # Cumulative sum of a_t * gamma_t
    cumsum = np.cumsum(a_t * gamma_clean)
    
    # S_L(n) = P(E_0^c) * exp(-cumsum)
    S_L = P_E0c * np.exp(-cumsum)
    
    return S_L


def compute_plugin_survival_lcb(
    a_t: float,
    gamma_lower: np.ndarray,
    P_E0c: float = 1.0
) -> np.ndarray:
    """
    Compute conservative plug-in survival bound using lower confidence bound on gamma.
    
    From equation (S-witness-LCB):
        S_L_LCB(n) = P(E_0^c) * exp(-sum_{t=1}^{n} a_t * underline{gamma}_t)
    
    Uses Clopper-Pearson lower bounds for gamma_t to provide a conservative
    (higher) survival estimate.
    
    Args:
        a_t: Theoretical hazard floor (constant per generation)
        gamma_lower: Array of lower confidence bounds for gamma
        P_E0c: Probability of favorable initialization (default 1.0)
    
    Returns:
        Array of S_L_LCB(n) values for n = 1, ..., len(gamma_lower)
    """
    # Replace NaN with 0 (conservative: assume no witness if unknown)
    gamma_clean = np.where(np.isnan(gamma_lower), 0.0, gamma_lower)
    
    # Cumulative sum of a_t * gamma_lower
    cumsum = np.cumsum(a_t * gamma_clean)
    
    # S_L_LCB(n) = P(E_0^c) * exp(-cumsum)
    S_L_LCB = P_E0c * np.exp(-cumsum)
    
    return S_L_LCB


def compute_tail_certificates(
    a_t: float,
    gamma_stats: Dict[str, Any],
    P_E0c: float = 1.0
) -> Dict[str, Any]:
    """
    Compute all plug-in tail certificates.
    
    Args:
        a_t: Theoretical hazard floor
        gamma_stats: Output from estimate_gamma_with_ci()
        P_E0c: Probability of favorable initialization
    
    Returns:
        Dictionary with:
        - generations: generation indices
        - S_L: witness-based survival bound (using gamma_t)
        - S_L_LCB: conservative bound (using gamma_lower)
        - final_S_L: S_L at last generation
        - final_S_L_LCB: S_L_LCB at last generation
    """
    gamma_t = gamma_stats.get('gamma_t')
    gamma_lower = gamma_stats.get('gamma_lower')
    generations = gamma_stats.get('generations')
    
    if gamma_t is None or generations is None:
        return {
            'generations': None,
            'S_L': None,
            'S_L_LCB': None,
            'final_S_L': None,
            'final_S_L_LCB': None,
            'error': 'Missing gamma estimates'
        }
    
    # Compute witness-based bound
    S_L = compute_plugin_survival_bound(a_t, gamma_t, P_E0c)
    
    # Compute conservative bound if lower bounds available
    if gamma_lower is not None:
        S_L_LCB = compute_plugin_survival_lcb(a_t, gamma_lower, P_E0c)
    else:
        S_L_LCB = None
    
    return {
        'generations': generations,
        'S_L': S_L,
        'S_L_LCB': S_L_LCB,
        'final_S_L': float(S_L[-1]) if len(S_L) > 0 else None,
        'final_S_L_LCB': float(S_L_LCB[-1]) if S_L_LCB is not None and len(S_L_LCB) > 0 else None,
        'a_t': a_t,
        'P_E0c': P_E0c,
    }


def compare_bounds_to_km(
    km_times: np.ndarray,
    km_survival: np.ndarray,
    certificates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare plug-in certificates to empirical KM survival curve.
    
    Quantifies the conservatism of the theoretical bound.
    
    Args:
        km_times: Event times from KM analysis
        km_survival: KM survival estimates at event times
        certificates: Output from compute_tail_certificates()
    
    Returns:
        Dictionary with comparison metrics
    """
    generations = certificates.get('generations')
    S_L = certificates.get('S_L')
    S_L_LCB = certificates.get('S_L_LCB')
    
    if generations is None or S_L is None or km_times is None:
        return {
            'valid_bound': None,
            'conservatism_factor': None,
            'error': 'Missing data for comparison'
        }
    
    # Check if theoretical bound is valid (S_L >= S_KM at all points)
    # Interpolate KM to generation grid
    km_at_gen = np.ones(len(generations))
    for i, g in enumerate(generations):
        # Find S_KM at generation g
        idx = np.searchsorted(km_times, g)
        if idx == 0:
            km_at_gen[i] = 1.0  # Before first event
        elif idx >= len(km_times):
            km_at_gen[i] = km_survival[-1]
        else:
            km_at_gen[i] = km_survival[idx - 1]
    
    # Check validity: S_L should upper-bound survival (be >= empirical)
    # But our bound is on P(tau > n), so S_L is an upper bound
    # The theoretical guarantee is: P(tau > n) <= S_L(n)
    # Empirical S_KM estimates P(tau > n)
    # Valid if S_L >= S_KM at all points
    valid_bound = np.all(S_L >= km_at_gen - 1e-10)  # Small tolerance for numerics
    
    # Conservatism: ratio of theoretical bound to empirical survival
    # Higher ratio = more conservative
    mask = km_at_gen > 0
    if mask.sum() > 0:
        ratios = S_L[mask] / km_at_gen[mask]
        mean_conservatism = float(np.mean(ratios))
        max_conservatism = float(np.max(ratios))
    else:
        mean_conservatism = None
        max_conservatism = None
    
    return {
        'valid_bound': bool(valid_bound),
        'mean_conservatism': mean_conservatism,
        'max_conservatism': max_conservatism,
        'km_at_generations': km_at_gen,
    }


def compute_required_generations(
    a_t: float,
    gamma_mean: float,
    target_survival: float = 0.01,
    P_E0c: float = 1.0
) -> Optional[int]:
    """
    Compute generations required to guarantee survival <= target.
    
    Solves: P(E_0^c) * exp(-n * a_t * gamma) <= target_survival
    
    Args:
        a_t: Theoretical hazard floor
        gamma_mean: Mean witness frequency
        target_survival: Target upper bound on survival probability
        P_E0c: Probability of favorable initialization
    
    Returns:
        Minimum n such that S_L(n) <= target_survival, or None if impossible
    """
    if a_t <= 0 or gamma_mean <= 0:
        return None
    
    if P_E0c <= target_survival:
        return 0  # Already satisfied
    
    # P_E0c * exp(-n * a_t * gamma) <= target
    # exp(-n * a_t * gamma) <= target / P_E0c
    # -n * a_t * gamma <= log(target / P_E0c)
    # n >= -log(target / P_E0c) / (a_t * gamma)
    
    n = -np.log(target_survival / P_E0c) / (a_t * gamma_mean)
    
    return int(np.ceil(n))


def format_certificate_summary(
    fname: str,
    d: int,
    eps: float,
    km_stats: Dict,
    gamma_stats: Dict,
    certificates: Dict,
    comparison: Dict
) -> Dict[str, Any]:
    """
    Format summary row for plug-in certificate table.
    
    Args:
        fname: Function name
        d: Dimension
        eps: Epsilon tolerance
        km_stats: KM statistics
        gamma_stats: Gamma estimation results
        certificates: Plug-in certificate results
        comparison: Comparison metrics
    
    Returns:
        Dictionary suitable for DataFrame row
    """
    return {
        'function': fname,
        'd': d,
        'eps': eps,
        'n_hits': km_stats.get('n_hits'),
        'tau_max': km_stats.get('tau_max'),
        'gamma_mean': gamma_stats.get('gamma_mean'),
        'gamma_lower_mean': gamma_stats.get('gamma_lower_mean'),
        'a_t': certificates.get('a_t'),
        'final_S_L': certificates.get('final_S_L'),
        'final_S_L_LCB': certificates.get('final_S_L_LCB'),
        'valid_bound': comparison.get('valid_bound'),
        'mean_conservatism': comparison.get('mean_conservatism'),
    }
