#!/usr/bin/env python3
"""
Gap analysis: comparing theoretical bounds to empirical hazard rates.

This module computes the gap factor between:
- Theoretical hazard floor: a_t * gamma_t
- Empirical hazard: p_hat from KM analysis
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
