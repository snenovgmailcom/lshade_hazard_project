#!/usr/bin/env python3
"""
gap_analysis.py

Compare empirical per-generation hazard estimates (from KM pooling) against
theoretical lower bounds derived in the paper.

This module focuses on the Morse-style lower bound (Theorem "morse-hazard"):

    \tilde a_t = (c_pair / H) * (g^- * Delta_F) * (q^- * eta_r)

and the per-generation (population-level) bound:

    p_t  >=  N_t * \tilde a_t    (optionally multiplied by an empirical gamma-rate)

where:
- H is the L-SHADE memory size.
- c_pair is the donor-pair concentration probability (often lower-bounded by beta1*beta2).
- Delta_F = F^+ - F^-.
- eta_r(d,c_cr) is the crossover tail probability from Lemma (eta-def).
- gamma is an empirical proxy for being in the "good memory" / witness-stable regime.

Because c_pair is not directly logged in most runs, we provide helper proxies:
- if beta1 and beta2 are available (fractions of eligible donors near best),
  use c_pair_proxy = beta1 * beta2;
- otherwise, use c_pair_proxy = beta1^2 (common approximation when archive is not logged).

The output of compute_gap_analysis is meant for table generation; it is not a proof.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    from .witness_frequency import eta_r, compute_morse_a_t, GAMMA_THRESHOLDS
except ImportError:  # pragma: no cover
    from witness_frequency import eta_r, compute_morse_a_t, GAMMA_THRESHOLDS


def compute_morse_bound(
    N_t: int,
    d: int,
    thresholds: Dict[str, float] = GAMMA_THRESHOLDS,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None,
    c_pair: Optional[float] = None,
    r: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute Morse-style per-individual and per-generation lower bounds.

    Returns dict:
      tilde_a (per-individual),
      p_lower = N_t * tilde_a  (per-generation),
      eta_r
    """
    H = int(thresholds["H"])
    F_minus = float(thresholds["F_minus"])
    F_plus = float(thresholds["F_plus"])
    g_minus = float(thresholds["g_minus"])
    q_minus = float(thresholds["q_minus"])
    c_cr = float(thresholds["c_cr"])

    Delta_F = float(F_plus - F_minus)
    eta = eta_r(d=d, c_cr=c_cr, r=r)

    if c_pair is None:
        b1 = float(beta1) if beta1 is not None and np.isfinite(beta1) else 0.0
        if beta2 is None or not np.isfinite(beta2):
            b2 = b1
        else:
            b2 = float(beta2)
        c_pair = float(max(min(b1 * b2, 1.0), 0.0))

    tilde_a = compute_morse_a_t(H=H, c_pair=c_pair, g_minus=g_minus, Delta_F=Delta_F, q_minus=q_minus, eta=eta)
    p_lower = float(int(N_t) * tilde_a)

    return {"tilde_a": float(tilde_a), "p_lower": float(p_lower), "eta_r": float(eta), "c_pair": float(c_pair)}


def compute_gap_analysis(
    km_stats: Dict[str, float],
    gamma_stats: Dict[str, float],
    N_tau_median: Optional[float],
    d: int,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None,
    thresholds: Dict[str, float] = GAMMA_THRESHOLDS,
    r: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compare empirical pooled hazard p_hat (from km_stats) to a Morse-style bound.

    Returns dict with:
      p_hat, gamma,
      tilde_a, p_lower, p_lower_gamma,
      gap = p_hat / p_lower_gamma (if p_lower_gamma>0), etc.
    """
    p_hat = float(km_stats.get("p_hat", 0.0))
    gamma = float(gamma_stats.get("gamma", 0.0))
    N = int(round(N_tau_median)) if N_tau_median is not None and np.isfinite(N_tau_median) else 0

    bound = compute_morse_bound(
        N_t=N,
        d=d,
        thresholds=thresholds,
        beta1=beta1,
        beta2=beta2,
        c_pair=None,
        r=r,
    )

    p_lower = float(bound["p_lower"])
    p_lower_gamma = float(gamma * p_lower)

    gap = float(p_hat / p_lower_gamma) if p_lower_gamma > 0 else float("inf")

    return {
        "p_hat": p_hat,
        "gamma": gamma,
        "N_tau": float(N),
        "tilde_a": float(bound["tilde_a"]),
        "p_lower": p_lower,
        "p_lower_gamma": p_lower_gamma,
        "gap_ratio": gap,
        "eta_r": float(bound["eta_r"]),
        "c_pair": float(bound["c_pair"]),
    }
