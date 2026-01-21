#!/usr/bin/env python3
r"""
witness_frequency.py

Memory-condition checks and witness-frequency estimators aligned with the paper.

This module provides:
- L2 (F-density) and L3 (CR-tail) checks for "empirically good memory"
  based on the actual L-SHADE sampling rules used in algorithms/lshade.py.
- Per-generation indicators and pooled frequency estimators (gamma, L2-rate, L3-rate)
  computed over at-risk runs (right-censoring handled via taus and budget).

Important implementation detail (matches algorithms/lshade.py):
- F is sampled by repeated Cauchy draws until F>0, then clipped: F := min(F, 1).
  Therefore, on the continuous region F in (0,1), the density is:

      g(F | mu) = cauchy_pdf(F; mu, sigma_f) / P(X>0),

  where X ~ Cauchy(mu, sigma_f). (There is additional point mass at F=1 due to clipping.)
  For intervals strictly below 1, the point mass is irrelevant.

- CR is sampled as Normal(mu, sigma_cr) and then clipped to [0,1]. For thresholds c_cr<1,
  P(CR >= c_cr) equals P(Z >= c_cr) for Z ~ Normal(mu, sigma_cr) (clipping does not
  reduce the upper-tail probability).

These correspond to the "good memory" condition (Definition / Eq. (good-memory-cond))
used in the theoretical hazard bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import cauchy, norm, beta, binom


# ----------------------------
# Default diagnostic thresholds (tune to match the paper constants)
# ----------------------------

GAMMA_THRESHOLDS: Dict[str, float] = {
    # Paper-style interval for "successful" / admissible F values
    "F_minus": 0.10,
    "F_plus": 0.90,
    # Uniform lower bound required on the F sampling density over [F_minus, F_plus]
    "g_minus": 0.10,
    # CR threshold and lower bound on its upper tail
    "c_cr": 0.50,
    "q_minus": 0.25,
    # Sampling scales used by L-SHADE
    "sigma_f": 0.10,
    "sigma_cr": 0.10,
    # L-SHADE settings used in theory tables / bounds
    "H": 6,
    "p_best": 0.11,
    "m_min": 4,
}

# ----------------------------
# L2/L3 checks for a single memory slot
# ----------------------------

def _as_float(x) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def check_L2(mu_f: float, F_minus: float, F_plus: float, g_minus: float, sigma_f: float = 0.10) -> bool:
    """
    L2: density lower bound for F.

    We compute the continuous density of the sampled F distribution on (0,1):
        g(F) = cauchy.pdf(F; mu_f, sigma_f) / P(X>0),
    where X ~ Cauchy(mu_f, sigma_f), because the algorithm rejects nonpositive samples.

    For unimodal densities, inf_{F in [F_minus, F_plus]} g(F) is attained at an endpoint,
    so we check the minimum density at the two endpoints.

    Returns True iff inf density >= g_minus.
    """
    mu_f = _as_float(mu_f)
    if not np.isfinite(mu_f):
        return False

    F_minus = float(F_minus)
    F_plus = float(F_plus)

    if not (0.0 < F_minus <= F_plus <= 1.0):
        raise ValueError("Require 0 < F_minus <= F_plus <= 1")

    # Normalization due to rejection of X<=0
    p_pos = 1.0 - float(cauchy.cdf(0.0, loc=mu_f, scale=sigma_f))
    if p_pos <= 0.0:
        return False

    d1 = float(cauchy.pdf(F_minus, loc=mu_f, scale=sigma_f)) / p_pos
    d2 = float(cauchy.pdf(F_plus, loc=mu_f, scale=sigma_f)) / p_pos

    return (min(d1, d2) >= float(g_minus))


def check_L3(mu_cr: float, c_cr: float, q_minus: float, sigma_cr: float = 0.10) -> bool:
    """
    L3: lower bound on the CR upper tail probability.

    The algorithm samples Z ~ Normal(mu_cr, sigma_cr) and clips to [0,1].
    For c_cr < 1, P(clipped(Z) >= c_cr) = P(Z >= c_cr).

    Returns True iff q(mu_cr) >= q_minus.
    """
    mu_cr = _as_float(mu_cr)
    if not np.isfinite(mu_cr):
        return False

    q = 1.0 - float(norm.cdf(c_cr, loc=mu_cr, scale=sigma_cr))
    return (q >= float(q_minus))


# ----------------------------
# Per-generation indicators for a run
# ----------------------------

@dataclass
class WitnessIndicators:
    L2: np.ndarray          # shape (T,), bool
    L3: np.ndarray          # shape (T,), bool
    gamma: np.ndarray       # shape (T,), bool (L2 & L3)


def compute_witness_indicators(
    memory_f: Sequence[Sequence[float]],
    memory_cr: Sequence[Sequence[float]],
    thresholds: Dict[str, float] = GAMMA_THRESHOLDS,
) -> WitnessIndicators:
    """
    Given logged memory arrays for a single run:
      memory_f[t]  = list/array length H of MF values at generation t
      memory_cr[t] = list/array length H of MCR values at generation t

    Return per-generation indicators:
      L2(t) = 1{ exists k : check_L2(MF[t,k]) }
      L3(t) = 1{ exists k : check_L3(MCR[t,k]) }

      gamma(t) = 1{ exists k : check_L2(MF[t,k]) AND check_L3(MCR[t,k]) }.

    The definition of "good memory" in the paper (Definition\ \ref{def:good-memory})
    requires that both conditions hold for the **same** memory slot, since L-SHADE
    samples (F, CR) from a single uniformly-chosen slot k each generation.
    """
    F_minus = thresholds["F_minus"]
    F_plus = thresholds["F_plus"]
    g_minus = thresholds["g_minus"]
    c_cr = thresholds["c_cr"]
    q_minus = thresholds["q_minus"]
    sigma_f = thresholds.get("sigma_f", 0.10)
    sigma_cr = thresholds.get("sigma_cr", 0.10)

    T = min(len(memory_f), len(memory_cr))
    L2_arr = np.zeros(T, dtype=bool)
    L3_arr = np.zeros(T, dtype=bool)
    gamma_arr = np.zeros(T, dtype=bool)

    for t in range(T):
        mf = list(memory_f[t])
        mcr = list(memory_cr[t])

        # L2: any slot satisfies density bound
        ok2 = any(
            check_L2(mu, F_minus=F_minus, F_plus=F_plus, g_minus=g_minus, sigma_f=sigma_f)
            for mu in mf
        )
        L2_arr[t] = ok2

        # L3: any slot satisfies CR tail bound
        ok3 = any(
            check_L3(mu, c_cr=c_cr, q_minus=q_minus, sigma_cr=sigma_cr)
            for mu in mcr
        )
        L3_arr[t] = ok3

        # Good memory (gamma): exists a single slot k where both are satisfied
        ok_gamma = False
        H = min(len(mf), len(mcr))
        for k in range(H):
            if check_L2(mf[k], F_minus=F_minus, F_plus=F_plus, g_minus=g_minus, sigma_f=sigma_f) and \
               check_L3(mcr[k], c_cr=c_cr, q_minus=q_minus, sigma_cr=sigma_cr):
                ok_gamma = True
                break
        gamma_arr[t] = ok_gamma

    return WitnessIndicators(L2=L2_arr, L3=L3_arr, gamma=gamma_arr)


# ----------------------------
# Pooled frequency estimation over at-risk runs
# ----------------------------

def estimate_gamma(
    taus: np.ndarray,
    indicators: List[WitnessIndicators],
    B: Optional[int] = None,
    start_gen: int = 0,
    end_gen: Optional[int] = None,
    alive_strict: bool = True,
) -> Dict[str, float]:
    """
    Pooled (time-weighted) estimates of L2, L3, and gamma over generations.

    This implements the pooled estimator described in Appendix (witness proxy),
    adapted to generic per-generation indicators:

      Y_t = #{runs with \tilde{tau} >= t}  (KM risk set)
      M_t^X = sum_r 1{alive at t} * 1{X holds at t in run r}
      \bar X = (sum_t M_t^X) / (sum_t Y_t)   (pooled time-weighted frequency)

    Where "alive at t" can be:
      - strict: \tilde{tau} > t  (recommended when X is meant to hold *before* hitting)
      - non-strict: \tilde{tau} >= t

    Parameters
    ----------
    taus : array (R,)
        Hitting times; np.inf for censored runs.
    indicators : list length R
        WitnessIndicators for each run.
    B : int or None
        Budget (last gen). If None, uses max available indicator length - 1.
    start_gen : int
        Start generation (inclusive) for pooling.
    end_gen : int or None
        End generation (inclusive). If None, uses B.
    alive_strict : bool
        Whether to use \tilde{tau} > t for "alive".

    Returns dict with:
      l2_rate, l3_rate, gamma, total_trials, total_hits_active
    """
    R = len(indicators)
    if R == 0:
        return {"l2_rate": np.nan, "l3_rate": np.nan, "gamma": np.nan, "total_trials": 0.0}

    taus = np.asarray(taus, dtype=float)

    # determine budget B from indicators if not provided
    max_T = min(ind.gamma.shape[0] for ind in indicators) - 1
    if B is None:
        B = max_T
    B = int(min(B, max_T))
    if B < 0:
        return {"l2_rate": np.nan, "l3_rate": np.nan, "gamma": np.nan, "total_trials": 0.0}

    end = B if end_gen is None else int(min(end_gen, B))
    start = int(max(0, start_gen))
    if end < start:
        return {"l2_rate": np.nan, "l3_rate": np.nan, "gamma": np.nan, "total_trials": 0.0}

    # observed time
    tilde = np.minimum(taus, float(B))

    total_risk = 0
    sum_L2 = 0
    sum_L3 = 0
    sum_gamma = 0

    for t in range(start, end + 1):
        if alive_strict:
            alive = tilde > float(t)
        else:
            alive = tilde >= float(t)

        idx = np.where(alive)[0]
        Y_t = int(idx.size)
        if Y_t == 0:
            continue

        total_risk += Y_t
        for r in idx:
            if indicators[r].L2[t]:
                sum_L2 += 1
            if indicators[r].L3[t]:
                sum_L3 += 1
            if indicators[r].gamma[t]:
                sum_gamma += 1

    # If there were no at-risk generations in the pooling window (e.g., all runs
    # hit at t=0 and alive_strict=True), these pooled quantities are undefined.
    if total_risk == 0:
        return {"l2_rate": np.nan, "l3_rate": np.nan, "gamma": np.nan, "total_trials": 0.0}

    return {
        "l2_rate": float(sum_L2 / total_risk),
        "l3_rate": float(sum_L3 / total_risk),
        "gamma": float(sum_gamma / total_risk),
        "total_trials": float(total_risk),
    }


# ----------------------------
# Confidence intervals for pooled gamma
# ----------------------------

def clopper_pearson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Two-sided Clopper–Pearson interval for Binomial(n,p) at confidence 1-alpha.
    """
    if n <= 0:
        return (0.0, 1.0)
    k = int(k)
    n = int(n)
    if k <= 0:
        lo = 0.0
    else:
        lo = float(beta.ppf(alpha / 2.0, k, n - k + 1))
    if k >= n:
        hi = 1.0
    else:
        hi = float(beta.ppf(1 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def estimate_gamma_with_ci(
    taus: np.ndarray,
    indicators: List[WitnessIndicators],
    B: Optional[int] = None,
    start_gen: int = 0,
    end_gen: Optional[int] = None,
    alpha: float = 0.05,
    alive_strict: bool = True,
) -> Dict[str, float]:
    """
    Same as estimate_gamma, but also returns a two-sided CP CI for the pooled gamma.
    """
    est = estimate_gamma(
        taus=taus,
        indicators=indicators,
        B=B,
        start_gen=start_gen,
        end_gen=end_gen,
        alive_strict=alive_strict,
    )

    n = int(est.get("total_trials", 0.0))
    k = int(round(est["gamma"] * n)) if n > 0 else 0
    lo, hi = clopper_pearson_interval(k, n, alpha=alpha)
    est.update({"gamma_lo": lo, "gamma_hi": hi, "k": float(k), "n": float(n)})
    return est


# ----------------------------
# Theoretical helper quantities (for bounds)
# ----------------------------

def eta_r(d: int, c_cr: float, r: Optional[int] = None) -> float:
    """
    eta_r(d,c_cr) from Lemma (eta-def):
      eta_r = P(Bin(d-1, c_cr) >= d-r-1).

    If r is None, we default to r = floor((d-1)/2).
    """
    d = int(d)
    if r is None:
        r = (d - 1) // 2
    r = int(r)
    thresh = d - r - 1
    n = d - 1
    # P(Bin(n,p) >= thresh)
    return float(binom.sf(thresh - 1, n, c_cr))


def compute_generic_a_t(
    N: int,
    A: int,
    H: int,
    p_best: float,
    m_min: int,
    g_minus: float,
    F_minus: float,
    F_plus: float,
    q_minus: float,
    c_cr: float,
    d: int,
    r: Optional[int] = None,
) -> float:
    """
    Generic (combinatorial) per-individual hazard lower bound (the small one),
    matching the structure of Proposition (gamma0-combinatorial) / Lemma (eta-def).

    This is the "a_t" that multiplies the witness probability gamma_t, with:
      a_t = (1/H) * g_minus*(F_plus-F_minus) * q_minus * eta_r * (1/(ceil(pN))) * ...
    and donor-selection factors based on minimal clique size m_min.

    NOTE: this is intentionally conservative; for Morse bounds use compute_morse_a_t.
    """
    N = int(N)
    A = int(A)
    H = int(H)
    m_min = int(m_min)
    pN = int(np.ceil(p_best * N)) if N > 0 else 1

    Delta_F = float(F_plus - F_minus)
    eta = eta_r(d=d, c_cr=c_cr, r=r)

    if N < 4 or m_min < 4:
        return 0.0

    # donor-selection probabilities (one possible conservative variant)
    # r1 from population excluding i and b -> size ~ (N-2)
    # r2 from pop+archive excluding i,b,r1 -> size ~ (N+A-3)
    # require r1,r2 in a cluster of size m_min: approximate with (m_min-2)/(N-2) * (m_min-3)/(N+A-3)
    denom1 = max(N - 2, 1)
    denom2 = max(N + A - 3, 1)
    p_pair = ((m_min - 2) / denom1) * ((m_min - 3) / denom2)

    return float((1.0 / H) * (g_minus * Delta_F) * (q_minus * eta) * (1.0 / pN) * p_pair)


def compute_morse_a_t(
    H: int,
    c_pair: float,
    g_minus: float,
    Delta_F: float,
    q_minus: float,
    eta: float,
) -> float:
    """
    Morse-style per-individual hazard lower bound (Theorem morse-hazard):
      \tilde a_t = (c_pair/H) * (g^- * Delta_F) * (q^- * eta).

    Here c_pair is the concentration-dependent donor-pair probability proxy
    (often lower-bounded by beta1*beta2 under the concentration assumption).
    """
    return float((c_pair / H) * (g_minus * Delta_F) * (q_minus * eta))
