# analysis/hazard_utils.py
"""
Common hazard / survival analysis utilities for L-SHADE experiments.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class EventData:
    y: np.ndarray       # observed time (hit or censor)
    delta: np.ndarray   # 1 if hit, 0 if censored
    c: np.ndarray       # censor time (max available)


def compute_event_data(
    curves: List[np.ndarray],
    f_star: float,
    epsilon: float
) -> EventData:
    """
    From best-so-far curves, compute first-hit times and censoring.
    Time is measured in generations (index in curve).
    """
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
        y=np.asarray(y, dtype=int),
        delta=np.asarray(delta, dtype=int),
        c=np.asarray(c, dtype=int),
    )


def kaplan_meier(
    y: np.ndarray,
    delta: np.ndarray,
    n_max: Optional[int] = None
):
    """
    Discrete-time Kaplan–Meier estimator.
    """
    if n_max is None:
        n_max = int(np.max(y))

    t_vals = np.arange(0, n_max + 1)
    n_at_risk = np.zeros(len(t_vals), dtype=int)
    d_events = np.zeros(len(t_vals), dtype=int)
    h_hat = np.zeros(len(t_vals), dtype=float)

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
        "n_max": n_max,
    }


def find_T(y: np.ndarray, delta: np.ndarray) -> Optional[int]:
    """
    First observed hit time T_first.
    """
    hits = y[delta == 1]
    return int(np.min(hits)) if len(hits) > 0 else None


def constant_hazard_mle(
    y: np.ndarray,
    delta: np.ndarray,
    T: int
) -> Optional[float]:
    """
    MLE for constant per-generation hazard on [T, B] with right censoring.
    """
    mask = (y >= T)
    if not np.any(mask):
        return None

    exposure = (y[mask] - T + 1).astype(float)
    events = delta[mask].astype(float)

    total_exposure = float(np.sum(exposure))
    total_events = float(np.sum(events))

    if total_exposure <= 0:
        return None

    return total_events / total_exposure


def find_max_valid_a(
    S_hat: np.ndarray,
    T: int
) -> Tuple[Optional[float], Optional[int]]:
    """
    Compute the tightest valid geometric envelope rate a_valid(T).

    Returns (a_valid, binding_n).
    """
    if T < 1 or T >= len(S_hat):
        return None, None

    S0 = S_hat[T - 1]
    if S0 <= 0:
        return None, None

    candidates = []

    for n in range(T, len(S_hat)):
        k = n - T + 1
        S_cond = S_hat[n] / S0

        if 0 < S_cond < 1:
            a_max = 1.0 - S_cond ** (1.0 / k)
            candidates.append((a_max, n))

    if not candidates:
        return None, None

    a_valid, binding_n = min(candidates, key=lambda x: x[0])
    return float(a_valid), int(binding_n)


def analyze_function(
    curves: List[np.ndarray],
    f_star: float,
    epsilon: float
) -> dict:
    """
    Full per-function analysis.
    """
    ev = compute_event_data(curves, f_star, epsilon)

    hits = int(np.sum(ev.delta == 1))
    cens = int(np.sum(ev.delta == 0))
    n_runs = len(ev.y)

    result = {
        "n_runs": n_runs,
        "hits": hits,
        "censored": cens,
        "hit_rate": hits / n_runs if n_runs > 0 else 0.0,
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

    if T is None or hits < 2:
        return result

    km = kaplan_meier(ev.y, ev.delta)

    p_cens = constant_hazard_mle(ev.y, ev.delta, T)
    result["p_cens"] = p_cens

    a_valid, _ = find_max_valid_a(km["S_hat"], T)
    result["a_valid"] = a_valid

    if p_cens is not None and a_valid is not None and p_cens > 0:
        result["ratio"] = a_valid / p_cens

    return result
