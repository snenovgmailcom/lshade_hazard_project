#!/usr/bin/env python3
"""
Witness frequency (gamma_t) estimation for L-SHADE.

This module estimates the partial witness frequency by checking
conditions (L2) and (L3) from the logged memory states.

Condition (L1) - the success-F window - is problem-dependent and
cannot be verified from algorithm state alone.
"""

import numpy as np
from scipy.stats import cauchy, norm, beta, binom
from typing import Dict, List, Optional, Tuple, Any


def check_L2(
    memory_f: np.ndarray,
    F_minus: float = 0.1,
    F_plus: float = 0.9,
    g_minus: float = 0.1,
    sigma_f: float = 0.1
) -> bool:
    """
    Check if any memory slot satisfies the F-density lower bound (L2).
    
    Condition: inf_{F in [F^-, F^+]} g^F_{t,i,k}(F) >= g^-
    
    The F-distribution is truncated Cauchy on (0, 1] with:
    - location = M_F[k]
    - scale = sigma_f (default 0.1)
    
    Args:
        memory_f: Array of M_F[k] values for k = 1, ..., H
        F_minus, F_plus: Interval for F
        g_minus: Density lower bound threshold
        sigma_f: Cauchy scale parameter
    
    Returns:
        True if at least one slot satisfies the condition
    """
    for mu in memory_f:
        if np.isnan(mu):
            continue
        
        # Find worst-case F in [F_minus, F_plus] (point furthest from mode)
        if mu <= F_minus:
            F_worst = F_plus
        elif mu >= F_plus:
            F_worst = F_minus
        else:
            # Mode is inside interval; worst case is at boundary furthest from mode
            F_worst = F_minus if (mu - F_minus) > (F_plus - mu) else F_plus
        
        # Truncated Cauchy density
        density = cauchy.pdf(F_worst, loc=mu, scale=sigma_f)
        norm_const = (
            cauchy.cdf(1.0, loc=mu, scale=sigma_f) - 
            cauchy.cdf(0.0, loc=mu, scale=sigma_f)
        )
        
        if norm_const > 0:
            density_normalized = density / norm_const
            if density_normalized >= g_minus:
                return True
    
    return False


def check_L3(
    memory_cr: np.ndarray,
    c_cr: float = 0.5,
    q_minus: float = 0.25,
    sigma_cr: float = 0.1
) -> bool:
    """
    Check if any memory slot satisfies the CR tail bound (L3).
    
    Condition: P(CR >= c_cr | M_CR[k]) >= q^-
    
    The CR-distribution is clipped Normal on [0, 1] with:
    - mean = M_CR[k] 
    - std = sigma_cr (default 0.1)
    
    Args:
        memory_cr: Array of M_CR[k] values (NaN for terminal value)
        c_cr: CR threshold
        q_minus: Tail probability lower bound
        sigma_cr: Normal std parameter
    
    Returns:
        True if at least one slot satisfies the condition
    """
    for mu in memory_cr:
        if np.isnan(mu):
            # Terminal value: CR = 0, so P(CR >= c_cr) = 0
            continue
        
        # P(CR >= c_cr) for Normal(mu, sigma_cr)
        # Note: This is approximate; exact would account for clipping to [0,1]
        prob = 1.0 - norm.cdf(c_cr, loc=mu, scale=sigma_cr)
        
        if prob >= q_minus:
            return True
    
    return False


def compute_witness_indicators(
    history: Dict[str, List],
    F_minus: float = 0.1,
    F_plus: float = 0.9,
    g_minus: float = 0.1,
    c_cr: float = 0.5,
    q_minus: float = 0.25,
    sigma_f: float = 0.1,
    sigma_cr: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute partial witness indicators for each generation.
    
    Args:
        history: Dictionary with 'memory_f' and 'memory_cr' lists
        Other args: Threshold parameters
    
    Returns:
        Tuple of:
        - indicators: 0/1 array (L2 AND L3)
        - l2_indicators: 0/1 array (L2 only)
        - l3_indicators: 0/1 array (L3 only)
    """
    n_gen = len(history['memory_f'])
    
    indicators = np.zeros(n_gen)
    l2_indicators = np.zeros(n_gen)
    l3_indicators = np.zeros(n_gen)
    
    for t in range(n_gen):
        mem_f = np.asarray(history['memory_f'][t])
        mem_cr = np.asarray(history['memory_cr'][t])
        
        l2_ok = check_L2(mem_f, F_minus, F_plus, g_minus, sigma_f)
        l3_ok = check_L3(mem_cr, c_cr, q_minus, sigma_cr)
        
        l2_indicators[t] = 1 if l2_ok else 0
        l3_indicators[t] = 1 if l3_ok else 0
        indicators[t] = 1 if (l2_ok and l3_ok) else 0
    
    return indicators, l2_indicators, l3_indicators


# =============================================================================
# Clopper-Pearson Confidence Bounds
# =============================================================================

def clopper_pearson_lower(
    k: int,
    n: int,
    alpha: float = 0.05
) -> float:
    """
    Compute Clopper-Pearson lower confidence bound for binomial proportion.
    
    For k successes out of n trials, returns the lower (1-alpha) bound.
    
    Uses the relationship between binomial and beta distributions:
        Lower bound = BetaInv(alpha; k, n-k+1)
    
    Args:
        k: Number of successes
        n: Number of trials
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        Lower confidence bound for the proportion k/n
    """
    if k == 0:
        return 0.0
    if k == n:
        return beta.ppf(alpha, k, 1)
    return beta.ppf(alpha, k, n - k + 1)


def clopper_pearson_upper(
    k: int,
    n: int,
    alpha: float = 0.05
) -> float:
    """
    Compute Clopper-Pearson upper confidence bound for binomial proportion.
    
    For k successes out of n trials, returns the upper (1-alpha) bound.
    
    Uses the relationship:
        Upper bound = BetaInv(1-alpha; k+1, n-k)
    
    Args:
        k: Number of successes
        n: Number of trials
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        Upper confidence bound for the proportion k/n
    """
    if k == n:
        return 1.0
    if k == 0:
        return beta.ppf(1 - alpha, 1, n)
    return beta.ppf(1 - alpha, k + 1, n - k)


def clopper_pearson_interval(
    k: int,
    n: int,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute two-sided Clopper-Pearson confidence interval.
    
    Args:
        k: Number of successes
        n: Number of trials
        alpha: Total significance level (default 0.05 for 95% CI)
    
    Returns:
        Tuple (lower, upper) confidence bounds
    """
    return (
        clopper_pearson_lower(k, n, alpha / 2),
        clopper_pearson_upper(k, n, alpha / 2)
    )


def estimate_gamma(
    runs: List[Dict],
    taus: np.ndarray,
    F_minus: float = 0.1,
    F_plus: float = 0.9,
    g_minus: float = 0.1,
    c_cr: float = 0.5,
    q_minus: float = 0.25,
    sigma_f: float = 0.1,
    sigma_cr: float = 0.1
) -> Dict[str, Any]:
    """
    Estimate gamma_t (partial witness frequency) on KM risk sets.
    
    For each generation t, computes:
        gamma_t = (# at-risk runs with L_t^partial) / (# at-risk runs)
    
    Args:
        runs: List of run dictionaries with 'history' key
        taus: Array of hitting times for these runs
        Other args: Threshold parameters
    
    Returns:
        Dictionary with:
        - generations: array of generation indices
        - gamma_t: array of gamma estimates
        - l2_rate: array of L2 satisfaction rates
        - l3_rate: array of L3 satisfaction rates
        - Y_t: array of risk set sizes
        - gamma_mean: overall mean gamma
    """
    # Check if history is available
    has_history = all('history' in r and r['history'] is not None for r in runs)
    if not has_history:
        return {
            'generations': None,
            'gamma_t': None,
            'l2_rate': None,
            'l3_rate': None,
            'Y_t': None,
            'gamma_mean': None,
            'error': 'No history logged in runs'
        }
    
    # Compute witness indicators for each run
    all_indicators = []
    all_l2 = []
    all_l3 = []
    
    for r in runs:
        ind, l2, l3 = compute_witness_indicators(
            r['history'], F_minus, F_plus, g_minus, c_cr, q_minus, sigma_f, sigma_cr
        )
        all_indicators.append(ind)
        all_l2.append(l2)
        all_l3.append(l3)
    
    # Find max generations
    max_gen = max(len(ind) for ind in all_indicators)
    
    # Estimate gamma_t on risk sets
    gamma_t = []
    l2_rate = []
    l3_rate = []
    Y_t_list = []
    
    for t in range(1, max_gen + 1):
        # Risk set: runs with tau >= t (not yet hit by generation t)
        at_risk_indices = [i for i, tau in enumerate(taus) if tau >= t]
        Y_t = len(at_risk_indices)
        
        if Y_t == 0:
            gamma_t.append(np.nan)
            l2_rate.append(np.nan)
            l3_rate.append(np.nan)
            Y_t_list.append(0)
            continue
        
        witness_count = 0
        l2_count = 0
        l3_count = 0
        
        for idx in at_risk_indices:
            if t - 1 < len(all_indicators[idx]):
                witness_count += all_indicators[idx][t - 1]
                l2_count += all_l2[idx][t - 1]
                l3_count += all_l3[idx][t - 1]
        
        gamma_t.append(witness_count / Y_t)
        l2_rate.append(l2_count / Y_t)
        l3_rate.append(l3_count / Y_t)
        Y_t_list.append(Y_t)
    
    gamma_t = np.array(gamma_t)
    
    return {
        'generations': np.arange(1, max_gen + 1),
        'gamma_t': gamma_t,
        'l2_rate': np.array(l2_rate),
        'l3_rate': np.array(l3_rate),
        'Y_t': np.array(Y_t_list),
        'gamma_mean': float(np.nanmean(gamma_t)),
        'l2_mean': float(np.nanmean(l2_rate)),
        'l3_mean': float(np.nanmean(l3_rate)),
    }


def estimate_gamma_with_ci(
    runs: List[Dict],
    taus: np.ndarray,
    alpha: float = 0.05,
    F_minus: float = 0.1,
    F_plus: float = 0.9,
    g_minus: float = 0.1,
    c_cr: float = 0.5,
    q_minus: float = 0.25,
    sigma_f: float = 0.1,
    sigma_cr: float = 0.1
) -> Dict[str, Any]:
    """
    Estimate gamma_t with Clopper-Pearson confidence bounds.
    
    Extends estimate_gamma() with conservative (1-alpha) confidence bounds.
    
    Under independent runs, M_t^L | Y_t ~ Binomial(Y_t, gamma_t^partial).
    
    Args:
        runs: List of run dictionaries with 'history' key
        taus: Array of hitting times for these runs
        alpha: Significance level for confidence bounds
        Other args: Threshold parameters
    
    Returns:
        Dictionary with all fields from estimate_gamma() plus:
        - gamma_lower: array of Clopper-Pearson lower bounds
        - gamma_upper: array of Clopper-Pearson upper bounds
        - M_t: array of witness counts (numerator)
    """
    # Check if history is available
    has_history = all('history' in r and r['history'] is not None for r in runs)
    if not has_history:
        return {
            'generations': None,
            'gamma_t': None,
            'gamma_lower': None,
            'gamma_upper': None,
            'M_t': None,
            'l2_rate': None,
            'l3_rate': None,
            'Y_t': None,
            'gamma_mean': None,
            'gamma_lower_mean': None,
            'error': 'No history logged in runs'
        }
    
    # Compute witness indicators for each run
    all_indicators = []
    all_l2 = []
    all_l3 = []
    
    for r in runs:
        ind, l2, l3 = compute_witness_indicators(
            r['history'], F_minus, F_plus, g_minus, c_cr, q_minus, sigma_f, sigma_cr
        )
        all_indicators.append(ind)
        all_l2.append(l2)
        all_l3.append(l3)
    
    # Find max generations
    max_gen = max(len(ind) for ind in all_indicators)
    
    # Estimate gamma_t on risk sets with confidence bounds
    gamma_t = []
    gamma_lower = []
    gamma_upper = []
    M_t_list = []
    l2_rate = []
    l3_rate = []
    Y_t_list = []
    
    for t in range(1, max_gen + 1):
        # Risk set: runs with tau >= t (not yet hit by generation t)
        at_risk_indices = [i for i, tau in enumerate(taus) if tau >= t]
        Y_t = len(at_risk_indices)
        
        if Y_t == 0:
            gamma_t.append(np.nan)
            gamma_lower.append(np.nan)
            gamma_upper.append(np.nan)
            M_t_list.append(0)
            l2_rate.append(np.nan)
            l3_rate.append(np.nan)
            Y_t_list.append(0)
            continue
        
        witness_count = 0
        l2_count = 0
        l3_count = 0
        
        for idx in at_risk_indices:
            if t - 1 < len(all_indicators[idx]):
                witness_count += int(all_indicators[idx][t - 1])
                l2_count += int(all_l2[idx][t - 1])
                l3_count += int(all_l3[idx][t - 1])
        
        # Point estimate
        gamma_t.append(witness_count / Y_t)
        M_t_list.append(witness_count)
        
        # Clopper-Pearson bounds
        lower, upper = clopper_pearson_interval(witness_count, Y_t, alpha)
        gamma_lower.append(lower)
        gamma_upper.append(upper)
        
        l2_rate.append(l2_count / Y_t)
        l3_rate.append(l3_count / Y_t)
        Y_t_list.append(Y_t)
    
    gamma_t = np.array(gamma_t)
    gamma_lower = np.array(gamma_lower)
    gamma_upper = np.array(gamma_upper)
    
    return {
        'generations': np.arange(1, max_gen + 1),
        'gamma_t': gamma_t,
        'gamma_lower': gamma_lower,
        'gamma_upper': gamma_upper,
        'M_t': np.array(M_t_list),
        'l2_rate': np.array(l2_rate),
        'l3_rate': np.array(l3_rate),
        'Y_t': np.array(Y_t_list),
        'gamma_mean': float(np.nanmean(gamma_t)),
        'gamma_lower_mean': float(np.nanmean(gamma_lower)),
        'gamma_upper_mean': float(np.nanmean(gamma_upper)),
        'l2_mean': float(np.nanmean(l2_rate)),
        'l3_mean': float(np.nanmean(l3_rate)),
        'alpha': alpha,
    }


def compute_theoretical_a_t(
    d: int,
    H: int = 6,
    p: float = 0.11,
    arc_rate: float = 2.6,
    g_minus: float = 0.1,
    Delta_F: float = 0.1,
    q_minus: float = 0.25,
    c_cr: float = 0.5,
    r: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute theoretical hazard floor a_t from equation (a_t-L).
    
    a_t = (1/H) * (1/m_t) * (1/(s1*(s2-1))) * (g^- * Delta_F) * (q^- * eta_r)
    
    Args:
        d: Dimension
        H: Memory size
        p: p-best fraction
        arc_rate: Archive rate
        g_minus, Delta_F, q_minus, c_cr: Witness thresholds
        r: Number of parent coordinates allowed (default: floor((d-1)/2))
    
    Returns:
        Dictionary with a_t and component values
    """
    N_init = 18 * d
    m_t = max(1, int(np.ceil(p * N_init)))
    s1 = N_init - 2
    s2 = N_init + int(arc_rate * N_init) - 2
    
    # Default r for c_cr = 0.5
    if r is None:
        r = (d - 1) // 2
    
    # eta_r = P(Bin(d-1, c_cr) >= d-r-1)
    eta_r = 1 - binom.cdf(d - r - 2, d - 1, c_cr)
    
    # Compute a_t
    a_t = (1/H) * (1/m_t) * (1/(s1 * (s2 - 1))) * (g_minus * Delta_F) * (q_minus * eta_r)
    
    return {
        'a_t': a_t,
        'd': d,
        'H': H,
        'N_init': N_init,
        'm_t': m_t,
        's1': s1,
        's2': s2,
        'r': r,
        'c_cr': c_cr,
        'eta_r': eta_r,
        'g_minus': g_minus,
        'Delta_F': Delta_F,
        'q_minus': q_minus,
    }
