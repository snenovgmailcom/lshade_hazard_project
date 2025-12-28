#!/usr/bin/env python3
"""
Kaplan-Meier survival analysis for L-SHADE hitting times.

This module provides functions to:
- Compute hitting times from convergence curves
- Estimate survival functions via Kaplan-Meier
- Compute geometric envelope rates and clustering indices
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any


# CEC2017 global optima
CEC2017_OPTIMA = {f"cec2017_f{i}": 100.0 * i for i in range(1, 31)}


def compute_hitting_times(
    curves: List[np.ndarray],
    f_global: float,
    eps_values: List[float]
) -> Dict[float, np.ndarray]:
    """
    Compute hitting times for multiple epsilon values.
    
    Args:
        curves: List of convergence curves (best-so-far per generation)
        f_global: Global optimum value
        eps_values: List of epsilon tolerances
    
    Returns:
        Dictionary mapping eps -> array of hitting times (np.inf if no hit)
    """
    result = {}
    for eps in eps_values:
        taus = []
        for curve in curves:
            curve = np.asarray(curve)
            hits = np.where(curve <= f_global + eps)[0]
            if len(hits) > 0:
                taus.append(hits[0] + 1)  # 1-indexed generations
            else:
                taus.append(np.inf)
        result[eps] = np.array(taus)
    return result


def kaplan_meier(
    taus: np.ndarray,
    B: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Kaplan-Meier survival estimator.
    
    Args:
        taus: Array of hitting times (np.inf for censored runs)
        B: Budget (censoring time). If None, use max finite tau + 10.
    
    Returns:
        times: Unique event times
        survival: S(t) at each time
        risk_sets: Y_t (number at risk) at each time
        events: d_t (number of events) at each time
    """
    taus = np.asarray(taus)
    
    # Handle censoring
    if B is None:
        finite_taus = taus[np.isfinite(taus)]
        B = int(finite_taus.max()) + 10 if len(finite_taus) > 0 else 1000
    
    T = np.minimum(taus, B)
    delta = (taus <= B).astype(int)  # 1 if hit, 0 if censored
    
    # Get unique event times (where delta=1)
    event_times = np.unique(T[delta == 1])
    event_times = event_times[event_times > 0]
    
    if len(event_times) == 0:
        return (
            np.array([0, B]),
            np.array([1.0, 1.0]),
            np.array([len(taus), 0]),
            np.array([0, 0])
        )
    
    times = []
    survival = []
    risk_sets = []
    events = []
    
    S = 1.0
    for t in event_times:
        Y_t = np.sum(T >= t)  # at risk
        d_t = np.sum((T == t) & (delta == 1))  # events
        
        if Y_t > 0:
            S = S * (1 - d_t / Y_t)
        
        times.append(t)
        survival.append(S)
        risk_sets.append(Y_t)
        events.append(d_t)
    
    return (
        np.array(times),
        np.array(survival),
        np.array(risk_sets),
        np.array(events)
    )


def compute_km_statistics(
    taus: np.ndarray,
    B: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive KM statistics for a set of hitting times.
    
    Args:
        taus: Array of hitting times (np.inf for censored)
        B: Budget for censoring
    
    Returns:
        Dictionary with:
        - n_runs, n_hits, hit_rate
        - T_first, tau_min, tau_med, tau_max
        - p_hat (MLE hazard)
        - a_valid (tightest geometric envelope)
        - clustering (a_valid / p_hat)
        - km_times, km_survival (for plotting)
    """
    taus = np.asarray(taus)
    finite_taus = taus[np.isfinite(taus)]
    
    n_runs = len(taus)
    n_hits = len(finite_taus)
    
    if n_hits == 0:
        return {
            'n_runs': n_runs,
            'n_hits': 0,
            'hit_rate': 0.0,
            'T_first': None,
            'tau_min': None,
            'tau_med': None,
            'tau_max': None,
            'p_hat': None,
            'a_valid': None,
            'clustering': None,
            'km_times': None,
            'km_survival': None,
        }
    
    T_first = int(finite_taus.min())
    tau_med = int(np.median(finite_taus))
    tau_max = int(finite_taus.max())
    
    times, S, Y, d = kaplan_meier(taus, B)
    
    # MLE hazard from T_first onwards
    mask = times >= T_first
    if mask.sum() > 0:
        total_events = d[mask].sum()
        total_exposure = Y[mask].sum()
        p_hat = total_events / total_exposure if total_exposure > 0 else 0
    else:
        p_hat = 0
    
    # Valid geometric envelope rate
    # a_valid = 1 - max_{n >= T} S_cond(n)^{1/(n-T+1)}
    S_T_minus_1 = 1.0  # S(T_first - 1) = 1 (no events before T_first)
    
    max_root = 0.0
    for i, t in enumerate(times):
        if t >= T_first and S[i] > 0:
            S_cond = S[i] / S_T_minus_1
            n_steps = t - T_first + 1
            root = S_cond ** (1.0 / n_steps)
            max_root = max(max_root, root)
    
    a_valid = 1 - max_root if max_root > 0 else 0
    
    # Clustering index
    clustering = a_valid / p_hat if p_hat > 0 else None
    
    return {
        'n_runs': n_runs,
        'n_hits': n_hits,
        'hit_rate': n_hits / n_runs,
        'T_first': T_first,
        'tau_min': int(finite_taus.min()),
        'tau_med': tau_med,
        'tau_max': tau_max,
        'p_hat': p_hat,
        'a_valid': a_valid,
        'clustering': clustering,
        'km_times': times,
        'km_survival': S,
    }


def compute_greenwood_variance(
    times: np.ndarray,
    survival: np.ndarray,
    risk_sets: np.ndarray,
    events: np.ndarray
) -> np.ndarray:
    """
    Compute Greenwood's variance estimate for KM survival.
    
    Args:
        times, survival, risk_sets, events: Output from kaplan_meier()
    
    Returns:
        Array of variance estimates at each time point
    """
    variance = np.zeros_like(survival)
    cumsum = 0.0
    
    for i in range(len(times)):
        Y_t = risk_sets[i]
        d_t = events[i]
        if Y_t > d_t and Y_t > 0:
            cumsum += d_t / (Y_t * (Y_t - d_t))
        variance[i] = survival[i]**2 * cumsum
    
    return variance
