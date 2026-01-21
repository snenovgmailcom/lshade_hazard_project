#!/usr/bin/env python3
r"""
km_analysis.py

Kaplan–Meier (KM) and discrete-time hazard utilities used by the empirical
section of the paper.

This module implements the discrete-time notation from Appendix
"Discrete-time survival notation and Kaplan–Meier estimator" and the
tail-envelope diagnostics used later.

Conventions (consistent with CEC/L-SHADE logging):
- Generations are indexed by integers t = 0,1,...,B where B is the last logged
  generation (typically len(curve)-1).
- The hitting time tau is the smallest t such that f_best(t) <= f* + eps.
  If the target is never hit within budget, tau = +inf (right-censored).
- Observed time is \tilde{tau} = min(tau, B), and delta = 1{tau <= B}.

KM risk set and event counts:
- Y_t = #{runs with \tilde{tau} >= t}
- d_t = #{runs with \tilde{tau} = t and delta = 1}

The (discrete-time) KM survival is:
  \hat S(t) = prod_{s=0}^t (1 - d_s / Y_s), with \hat S(-1) := 1.

Tail "constant hazard" pooling (Appendix "Conservative post-T tail envelopes"):
Given a tail start T (integer, 0<=T<=B), define:
  D(T)      = sum_{t=T+1}^B d_t
  N_trial(T)= sum_{t=T+1}^B Y_{t-1} = sum_{s=T}^{B-1} Y_s
and the pooled MLE hazard:
  \hat a(T) = D(T) / N_trial(T).

We also provide:
- a_one_sided_lcb(T): one-sided Clopper–Pearson lower bound on a(T)
  (optional, for "validated" conservative rates).
- a_env(T): minimal geometric-envelope rate a such that for all t>=T,
    \hat S(t) <= \hat S(T) * (1-a)^{t-T}
  (deterministic diagnostic of temporal clustering / hazard nonstationarity).

These quantities are useful for reporting and for comparing with the paper's
theoretical per-generation hazard lower bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta


# ----------------------------
# Core utilities
# ----------------------------

def compute_hitting_times(curves: List[np.ndarray], f_star: float, eps: float) -> np.ndarray:
    """
    Compute per-run hitting times:
      tau = inf{ t : f_best(t) <= f_star + eps }.
    Returns an array of length R with tau in {0,1,...,B} or np.inf.
    """
    taus: List[float] = []
    threshold = f_star + float(eps)

    for curve in curves:
        # curve is best-so-far objective; assume 1D array indexed by generation
        hit_idx = np.where(np.asarray(curve) <= threshold)[0]
        if len(hit_idx) == 0:
            taus.append(np.inf)
        else:
            taus.append(float(hit_idx[0]))
    return np.asarray(taus, dtype=float)


@dataclass
class KMResult:
    times: np.ndarray        # integer times 0..B
    Y: np.ndarray            # risk set Y_t
    d: np.ndarray            # events d_t
    S: np.ndarray            # KM survival \hat S(t) for t=0..B
    hazard: np.ndarray       # discrete hazard estimate \hat p_t = d_t / Y_t (0 when Y_t=0)


def kaplan_meier_discrete(taus: np.ndarray, B: int) -> KMResult:
    """
    Discrete-time Kaplan–Meier estimator on t=0..B.

    Parameters
    ----------
    taus : array of shape (R,)
        Hitting times per run; may contain np.inf for censored runs.
    B : int
        Budget in generations (last logged generation index).

    Returns
    -------
    KMResult with arrays of length B+1.
    """
    taus = np.asarray(taus, dtype=float)
    if B < 0:
        raise ValueError("B must be nonnegative")

    # observed time and event indicator
    tilde = np.minimum(taus, float(B))
    delta = (taus <= float(B)).astype(int)

    times = np.arange(B + 1, dtype=int)

    # risk set and events
    # Y_t = #{tilde >= t}
    Y = np.array([(tilde >= t).sum() for t in times], dtype=int)
    # d_t = #{tilde == t and delta==1}
    d = np.array([((tilde == t) & (delta == 1)).sum() for t in times], dtype=int)

    # hazard p_t = d_t / Y_t (define 0 if Y_t=0)
    hazard = np.zeros_like(times, dtype=float)
    mask = Y > 0
    hazard[mask] = d[mask] / Y[mask]

    # KM survival
    S = np.ones_like(times, dtype=float)
    surv = 1.0
    for t in times:
        if Y[t] > 0:
            surv *= (1.0 - d[t] / Y[t])
        S[t] = surv

    return KMResult(times=times, Y=Y, d=d, S=S, hazard=hazard)


# ----------------------------
# Tail pooling and envelopes
# ----------------------------

def _tail_counts(km: KMResult, T: int) -> Tuple[int, int]:
    """
    Return (D(T), N_trial(T)) as defined in the module docstring.
    """
    if T < 0 or T > int(km.times[-1]):
        raise ValueError("T must satisfy 0 <= T <= B")

    B = int(km.times[-1])

    # D(T) = sum_{t=T+1}^B d_t
    D = int(km.d[T+1 : B+1].sum()) if T < B else 0

    # N_trial(T) = sum_{t=T+1}^B Y_{t-1} = sum_{s=T}^{B-1} Y_s
    N_trial = int(km.Y[T : B].sum()) if T < B else 0

    return D, N_trial


def pooled_tail_hazard(km: KMResult, T: int) -> float:
    """
    Pooled MLE hazard \hat a(T) = D(T) / N_trial(T).
    """
    D, N_trial = _tail_counts(km, T)
    if N_trial <= 0:
        return 0.0
    return float(D / N_trial)


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """
    One-sided Clopper–Pearson lower confidence bound for a Binomial(n, p):
      P(p >= L) >= 1 - alpha.

    For k=0, the lower bound is 0.
    """
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    if k > n:
        raise ValueError("k cannot exceed n")
    # One-sided lower bound: Beta(alpha; k, n-k+1)
    return float(beta.ppf(alpha, k, n - k + 1))


def pooled_tail_hazard_lcb(km: KMResult, T: int, alpha: float = 0.05) -> float:
    """
    One-sided (1-alpha) lower bound on the pooled tail hazard, by treating
    D(T) as Binomial(N_trial(T), a).
    """
    D, N_trial = _tail_counts(km, T)
    return clopper_pearson_lower(D, N_trial, alpha=alpha)


def geometric_envelope_rate(km: KMResult, T: int) -> float:
    """
    Event-time geometric envelope rate a_env(T).

    The paper's tail bound compares survival to a geometric decay.
    In finite-sample discrete-time KM curves, \hat S(t) is a step function and is
    often *flat* on generations with d_t=0. If we enforced the inequality at every
    generation, any post-T flat segment would force a_env(T)=0, which is not a
    useful diagnostic.

    Therefore we compute the envelope **only over event times** (generations where
    d_t>0), i.e. we maximize over t>T with d_t>0:
        a_env(T) = 1 - max_{t>T, d_t>0} ( S(t) / S(T) )^{1/(t-T)}.

    This is a deterministic shape diagnostic (not a confidence bound).
    Returns 0.0 if S(T)=0 or there are no events after T.
    """
    B = int(km.times[-1])
    if T < 0 or T > B:
        raise ValueError("T must satisfy 0 <= T <= B")
    if T == B:
        return 0.0
    S_T = float(km.S[T])
    if S_T <= 0.0:
        return 0.0

    roots = []
    for t in range(T + 1, B + 1):
        # Skip flat segments (no observed events) to avoid collapsing a_env to 0.
        if km.d[t] == 0:
            continue
        ratio = float(km.S[t]) / S_T
        ratio = min(max(ratio, 0.0), 1.0)
        if ratio >= 1.0 - 1e-15:
            # numerical safety; an event time should have ratio < 1
            continue
        root = 0.0 if ratio <= 0.0 else ratio ** (1.0 / (t - T))
        roots.append(root)

    max_root = max(roots) if roots else 1.0
    max_root = min(max(max_root, 0.0), 1.0)
    return float(1.0 - max_root)


# ----------------------------
# Reporting helper
# ----------------------------

def compute_km_statistics(
    taus: np.ndarray,
    B: int,
    tail_start: Optional[int] = None,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Compute headline KM / hazard statistics for a given tau sample.

    Parameters
    ----------
    taus : array-like
        Hitting times; np.inf for censored.
    B : int
        Budget generation index (last logged gen).
    tail_start : int or None
        Tail start T used in pooled hazard / envelopes. If None, we use
        T = tau_first (first observed hit). If there are no hits, use 0.
    alpha : float
        For one-sided Clopper–Pearson LCB on the pooled tail hazard.

    Returns a dict with keys:
      n_hits, tau_first, tau_median, tau_max,
      p_hat (pooled tail hazard), p_lcb (LCB),
      a_env (event-time geometric envelope rate),
      clustering (inverse coefficient of variation of inter-hit gaps).
    """
    taus = np.asarray(taus, dtype=float)
    finite = taus[np.isfinite(taus)]

    n_hits = int((taus <= float(B)).sum())

    if finite.size == 0:
        tau_first = np.nan
        tau_median = np.nan
        tau_max = np.nan
    else:
        tau_first = float(np.min(finite))
        tau_median = float(np.median(finite))
        tau_max = float(np.max(finite))

    T = tail_start
    if T is None:
        T = int(tau_first) if np.isfinite(tau_first) else 0
    T = int(np.clip(T, 0, B))

    km = kaplan_meier_discrete(taus, B=B)

    p_hat = pooled_tail_hazard(km, T=T)
    p_lcb = pooled_tail_hazard_lcb(km, T=T, alpha=alpha)
    a_env = geometric_envelope_rate(km, T=T)

    # Temporal clustering index based on inter-hit gaps among successful runs.
    # We use the inverse CV (mean / std), so smaller values indicate more bursty
    # / clustered hitting patterns (as in the empirical discussion).
    clustering = float("nan")
    if finite.size >= 3:
        gaps = np.diff(np.sort(finite))
        if gaps.size > 0:
            mu = float(np.mean(gaps))
            sigma = float(np.std(gaps))
            if sigma > 0:
                clustering = mu / sigma
            elif mu > 0:
                clustering = float("inf")

    return {
        "n_hits": n_hits,
        "tau_first": tau_first,
        "tau_median": tau_median,
        "tau_max": tau_max,
        "tail_start": float(T),
        "p_hat": p_hat,
        "p_lcb": p_lcb,
        "a_env": a_env,
        "clustering": clustering,
    }
