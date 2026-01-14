#!/usr/bin/env python3
"""
check_Gt.py

Unified witness-condition verification for L-SHADE empirical analysis.
PARALLELIZED version using multiprocessing across runs.

This script verifies the EMPIRICALLY CHECKABLE conditions from the main paper:
- G2-G3 (Concentration): cluster of size >= m with diameter <= r_conc around best
- G4 (Sublevel Progress): f_best <= f* + eps_out  
- G5 (Good Memory): existence of slot k with g_k^- >= g_thresh and q_{t,k} >= q_thresh

We define the empirical proxy:
    Ghat_t := (G2 ∧ G3 ∧ G4 ∧ G5)

Note: Condition (G1) - local basin membership - requires knowledge of the basin 
center x_j^* and strong-convexity radius r_0, which is unavailable for CEC 
black-box functions. We treat (G1) as an assumed modeling condition.

Usage:
    # Parallel (default: use all CPUs)
    python analysis/check_Gt.py \\
        --pkl experiments/D10/f22/f22.pkl \\
        --func f22 --dim 10 --eps_in 10 \\
        --r_conc 2.0 --jobs 16 \\
        --outdir results/witness/D10/f22

    # Sequential (for debugging)
    python analysis/check_Gt.py \\
        --pkl experiments/D10/f22/f22.pkl \\
        --func f22 --dim 10 --eps_in 10 \\
        --r_conc 2.0 --jobs 1 \\
        --outdir results/witness/D10/f22

Outputs:
    - per_generation.csv: per-generation condition checks for all runs
    - per_run_summary.csv: per-run aggregated statistics
    - summary.json: overall aggregated statistics
    - Ghat_rates.png: visualization of condition satisfaction over time
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import norm


# =============================================================================
# SECTION 1: G5 (Good Memory) Checks
# =============================================================================

def g_minus_at_slot(
    M_F_k: float,
    F_minus: float = 0.1,
    F_plus: float = 0.9,
    scale: float = 0.1
) -> float:
    """
    Compute infimum of truncated Cauchy density over [F_minus, F_plus].
    
    L-SHADE F sampling: Cauchy(M_F[k], scale), reject if <= 0, clip to 1.
    The density on (0,1) is:
        g(F) = cauchy.pdf(F; M_F_k, scale) / P(X > 0)
    where X ~ Cauchy(M_F_k, scale).
    
    Note: The induced F-law is a mixture: continuous on (0,1) and an atom at F=1
    from clipping X > 1. Since we certify F in [F_minus, F_plus] ⊂ (0,1), only
    the continuous part matters for our density bound.
    
    For unimodal Cauchy, infimum over [F_minus, F_plus] is at an endpoint.
    """
    if not np.isfinite(M_F_k):
        return 0.0
    
    from scipy.stats import cauchy
    p_pos = 1.0 - float(cauchy.cdf(0.0, loc=M_F_k, scale=scale))
    if p_pos <= 0:
        return 0.0
    
    g_lo = cauchy.pdf(F_minus, loc=M_F_k, scale=scale) / p_pos
    g_hi = cauchy.pdf(F_plus, loc=M_F_k, scale=scale) / p_pos
    
    return float(min(g_lo, g_hi))


def q_at_slot(
    M_CR_k: float,
    c_cr: float = 0.5,
    sigma: float = 0.1
) -> float:
    """
    Compute P(CR >= c_cr) for CLIPPED normal on [0, 1].
    
    IMPORTANT: Clipping != truncation!
    For c_cr in (0,1): P(clipped(X) >= c_cr) = P(X >= c_cr)
    """
    if not np.isfinite(M_CR_k):
        return 0.0
    
    if c_cr <= 0:
        return 1.0
    if c_cr >= 1:
        return 0.0
    
    return float(norm.sf(c_cr, loc=M_CR_k, scale=sigma))


def check_G5(
    memory_f: List[float],
    memory_cr: List[float],
    F_minus: float = 0.1,
    F_plus: float = 0.9,
    g_thresh: float = 0.01,
    c_cr: float = 0.5,
    q_thresh: float = 0.1,
    sigma_f: float = 0.1,
    sigma_cr: float = 0.1
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check G5 (Good Memory): existence of slot k satisfying both density bounds.
    """
    H = min(len(memory_f), len(memory_cr))
    
    g_values = []
    q_values = []
    good_slots = []
    
    for k in range(H):
        g_k = g_minus_at_slot(memory_f[k], F_minus, F_plus, sigma_f)
        q_k = q_at_slot(memory_cr[k], c_cr, sigma_cr)
        
        g_values.append(g_k)
        q_values.append(q_k)
        
        if g_k >= g_thresh and q_k >= q_thresh:
            good_slots.append(k)
    
    G5_holds = len(good_slots) > 0
    
    details = {
        "g_max": max(g_values) if g_values else 0.0,
        "q_max": max(q_values) if q_values else 0.0,
        "n_good_slots": len(good_slots),
    }
    
    return G5_holds, details


# =============================================================================
# SECTION 2: G2-G3 (Concentration) Checks
# =============================================================================

def compute_cluster(
    positions: np.ndarray,
    best_idx: int,
    r_conc: float
) -> Tuple[np.ndarray, int]:
    """
    Compute cluster around best individual with DIAMETER guarantee.
    
    To ensure diameter(C_t) <= r_conc, we use a ball of radius r_conc/2.
    """
    X = np.asarray(positions, dtype=float)
    N = X.shape[0]
    
    if N == 0:
        return np.array([], dtype=int), 0
    
    x_best = X[best_idx]
    distances = np.linalg.norm(X - x_best, axis=1)
    
    # Use radius = r_conc/2 to guarantee diameter <= r_conc
    r_ball = 0.5 * r_conc
    cluster_mask = distances <= r_ball
    cluster_indices = np.where(cluster_mask)[0]
    cluster_size = len(cluster_indices)
    
    return cluster_indices, cluster_size


def check_G2G3(
    positions: np.ndarray,
    fitness: np.ndarray,
    r_conc: float,
    m_min: int = 4
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check G2-G3 (Concentration):
        G2: |C_t| >= m_min and x_best in C_t
        G3: diam(C_t) <= r_conc (guaranteed by using radius r_conc/2)
    """
    X = np.asarray(positions, dtype=float)
    f = np.asarray(fitness, dtype=float)
    
    if X.shape[0] == 0 or f.shape[0] == 0:
        return False, {"cluster_size": 0, "best_idx": -1, "beta1": 0.0}
    
    best_idx = int(np.argmin(f))
    cluster_indices, cluster_size = compute_cluster(X, best_idx, r_conc)
    
    best_in_cluster = best_idx in cluster_indices
    G2G3_holds = (cluster_size >= m_min) and best_in_cluster
    
    details = {
        "cluster_size": cluster_size,
        "best_idx": best_idx,
        "beta1": cluster_size / X.shape[0] if X.shape[0] > 0 else 0.0,
    }
    
    return G2G3_holds, details


# =============================================================================
# SECTION 3: G4 (Sublevel Progress) Check
# =============================================================================

def check_G4(
    fitness: np.ndarray,
    f_star: float,
    eps_out: float
) -> Tuple[bool, float]:
    """
    Check G4 (Sublevel Progress): f_best <= f* + eps_out.
    """
    f = np.asarray(fitness, dtype=float)
    
    if f.size == 0:
        return False, np.inf
    
    f_best = float(np.min(f))
    G4_holds = f_best <= f_star + eps_out
    
    return G4_holds, f_best


# =============================================================================
# SECTION 4: Combined Ghat_t Check
# =============================================================================

@dataclass
class GhatParams:
    """Parameters for Ghat_t verification."""
    # G5 (Good Memory)
    F_minus: float = 0.1
    F_plus: float = 0.9
    g_thresh: float = 0.01
    c_cr: float = 0.5
    q_thresh: float = 0.1
    sigma_f: float = 0.1
    sigma_cr: float = 0.1
    
    # G2-G3 (Concentration)
    r_conc: float = 2.0  # DIAMETER bound; cluster uses radius = r_conc/2
    m_min: int = 4       # L-SHADE default N_min
    
    # G4 (Sublevel Progress)
    eps_out: float = 100.0  # Outer tolerance, should be > eps_in
    
    # gamma_0 computation
    H: int = 6
    p_best: float = 0.11


def check_Ghat(
    positions: np.ndarray,
    fitness: np.ndarray,
    memory_f: List[float],
    memory_cr: List[float],
    pop_size: int,
    archive_size: int,
    f_star: float,
    params: GhatParams
) -> Dict[str, Any]:
    """
    Check empirical witness-stable regime Ghat_t = (G2 ∧ G3 ∧ G4 ∧ G5).
    """
    G2G3_holds, G2G3_details = check_G2G3(
        positions, fitness, params.r_conc, params.m_min
    )
    
    G4_holds, f_best = check_G4(fitness, f_star, params.eps_out)
    
    G5_holds, G5_details = check_G5(
        memory_f, memory_cr,
        F_minus=params.F_minus,
        F_plus=params.F_plus,
        g_thresh=params.g_thresh,
        c_cr=params.c_cr,
        q_thresh=params.q_thresh,
        sigma_f=params.sigma_f,
        sigma_cr=params.sigma_cr
    )
    
    Ghat_holds = G2G3_holds and G4_holds and G5_holds
    
    gamma0 = 0.0
    if Ghat_holds:
        gamma0 = compute_gamma0(
            N_t=pop_size,
            A_t=archive_size,
            m=G2G3_details["cluster_size"],
            H=params.H,
            p_best=params.p_best,
            g_minus=params.g_thresh,
            Delta_F=params.F_plus - params.F_minus,
            q_minus=params.q_thresh
        )
    
    return {
        "Ghat": Ghat_holds,
        "G1_assumed": True,
        "G2G3": G2G3_holds,
        "G4": G4_holds,
        "G5": G5_holds,
        "f_best": f_best,
        "cluster_size": G2G3_details["cluster_size"],
        "beta1": G2G3_details["beta1"],
        "n_good_slots": G5_details["n_good_slots"],
        "g_max": G5_details["g_max"],
        "q_max": G5_details["q_max"],
        "N_t": pop_size,
        "A_t": archive_size,
        "gamma0": gamma0,
    }


# =============================================================================
# SECTION 5: T_wit Estimation (O(T) backward scan)
# =============================================================================

def estimate_Twit(Ghat_sequence: List[bool]) -> float:
    """
    Estimate stabilization time T_wit using O(T) backward scan.
    """
    n = len(Ghat_sequence)
    if n == 0:
        return float("inf")
    
    suffix_all_true = True
    Twit = float("inf")
    
    for t in range(n - 1, -1, -1):
        suffix_all_true = suffix_all_true and Ghat_sequence[t]
        if suffix_all_true:
            Twit = float(t)
    
    return Twit


def compute_T_conc_L(arr: np.ndarray, L: int) -> float:
    """
    Compute L-consecutive stability time for clustering.
    
    T_conc(L) = min{t : arr[t] = arr[t+1] = ... = arr[t+L-1] = True}
    
    This is more robust than suffix definition when clustering flickers.
    
    Parameters
    ----------
    arr : array of bool
        Per-generation clustering indicator.
    L : int
        Required consecutive generations (typically 5, 10, or 20).
    
    Returns
    -------
    T_conc : float
        First time with L consecutive True values, or inf if never.
    """
    n = len(arr)
    if n < L:
        return float("inf")
    
    # Sliding window: count consecutive True from each position
    consecutive = 0
    for t in range(n):
        if arr[t]:
            consecutive += 1
            if consecutive >= L:
                return float(t - L + 1)  # Start of the L-window
        else:
            consecutive = 0
    
    return float("inf")


# =============================================================================
# SECTION 6: gamma_0 Computation (WITHOUT eta_r)
# =============================================================================

def compute_gamma0(
    N_t: int,
    A_t: int,
    m: int,
    H: int = 6,
    p_best: float = 0.11,
    g_minus: float = 0.01,
    Delta_F: float = 0.8,
    q_minus: float = 0.1
) -> float:
    """
    Compute witness probability lower bound gamma_0(t).
    NOTE: No eta_r factor - that's in the Morse bound, not witness probability.
    """
    if N_t < 4 or m < 4:
        return 0.0
    
    memory_factor = (1.0 / H) * g_minus * Delta_F * q_minus
    
    pN = int(np.ceil(p_best * N_t))
    pbest_factor = 1.0 / max(pN, 1)
    
    denom1 = max(N_t - 2, 1)
    denom2 = max(N_t + A_t - 3, 1)
    
    r1_factor = (m - 2) / denom1 if m >= 2 else 0.0
    r2_factor = (m - 3) / denom2 if m >= 3 else 0.0
    
    gamma0 = memory_factor * pbest_factor * r1_factor * r2_factor
    
    return float(gamma0)


# =============================================================================
# SECTION 7: Single-run Analysis (worker function for parallel processing)
# =============================================================================

def analyze_run_worker(args: Tuple) -> Optional[Dict[str, Any]]:
    """
    Worker function to analyze a single run.
    
    This function is designed to be called by ProcessPoolExecutor.
    All arguments are passed as a tuple for compatibility with pool.map().
    
    Parameters
    ----------
    args : tuple
        (run_idx, run, f_star, params_dict, eps_in, max_gens)
    
    Returns
    -------
    result : dict or None
        Per-run results including DataFrame rows and summary statistics.
    """
    run_idx, run, f_star, params_dict, eps_in, max_gens = args
    
    # Reconstruct params from dict (dataclass not picklable across processes)
    params = GhatParams(**params_dict)
    
    hist = run.get("history", {})
    
    positions = hist.get("positions", [])
    fitness = hist.get("fitness", [])
    memory_f = hist.get("memory_f", [])
    memory_cr = hist.get("memory_cr", [])
    pop_size = hist.get("pop_size", [])
    archive_size = hist.get("archive_size", [])
    
    # Determine number of generations
    n_gens = min(len(positions), len(fitness), len(memory_f), len(memory_cr))
    if max_gens is not None:
        n_gens = min(n_gens, max_gens)
    
    if n_gens == 0:
        return None
    
    # Compute hitting time for this run
    curve = np.asarray(run.get("curve", []), dtype=float)
    hit_idx = np.where(curve <= f_star + eps_in)[0]
    tau = float(hit_idx[0]) if len(hit_idx) > 0 else np.inf
    
    # Analyze each generation
    rows = []
    for t in range(n_gens):
        pos_t = np.asarray(positions[t], dtype=float)
        fit_t = np.asarray(fitness[t], dtype=float)
        mf_t = list(memory_f[t]) if t < len(memory_f) else []
        mcr_t = list(memory_cr[t]) if t < len(memory_cr) else []
        N_t = int(pop_size[t]) if t < len(pop_size) else pos_t.shape[0]
        A_t = int(archive_size[t]) if t < len(archive_size) else 0
        
        result = check_Ghat(
            positions=pos_t,
            fitness=fit_t,
            memory_f=mf_t,
            memory_cr=mcr_t,
            pop_size=N_t,
            archive_size=A_t,
            f_star=f_star,
            params=params
        )
        
        result["t"] = t
        result["run"] = run_idx
        result["tau"] = tau
        rows.append(result)
    
    # Per-run summary
    df = pd.DataFrame(rows)
    
    # Compute tau_out: first t where f_best <= f* + eps_out
    f_best_series = df["f_best"].values
    eps_out = params.eps_out
    tau_out_idx = np.where(f_best_series <= f_star + eps_out)[0]
    tau_out = float(tau_out_idx[0]) if len(tau_out_idx) > 0 else np.inf
    
    # =========================================================================
    # FIRST OCCURRENCE TIMES
    # =========================================================================
    G2G3_arr = df["G2G3"].values.astype(bool)
    G4_arr = df["G4"].values.astype(bool)
    G5_arr = df["G5"].values.astype(bool)
    Ghat_arr = df["Ghat"].values.astype(bool)
    
    # First t where condition holds
    tau_G2G3_first = float(np.where(G2G3_arr)[0][0]) if G2G3_arr.any() else np.inf
    tau_G4_first = float(np.where(G4_arr)[0][0]) if G4_arr.any() else np.inf
    tau_G5_first = float(np.where(G5_arr)[0][0]) if G5_arr.any() else np.inf
    tau_Ghat_first = float(np.where(Ghat_arr)[0][0]) if Ghat_arr.any() else np.inf
    
    # =========================================================================
    # STABILIZATION TIMES (restricted to at-risk: must hold until tau)
    # tau_X_stable = inf{t : X(s) holds for all s in [t, tau)}
    # =========================================================================
    tau_int = int(tau) if np.isfinite(tau) else len(Ghat_arr)
    
    # Backward scan to find stabilization time
    def compute_stable_time(arr: np.ndarray, tau_int: int) -> float:
        """Find first t where arr[s]=True for all s in [t, tau_int)."""
        if tau_int <= 0:
            return np.inf
        # Restrict to at-risk period
        at_risk_arr = arr[:tau_int]
        if len(at_risk_arr) == 0:
            return np.inf
        
        # Backward scan: find first t where all suffix is True
        suffix_true = True
        stable_t = np.inf
        for t in range(len(at_risk_arr) - 1, -1, -1):
            suffix_true = suffix_true and at_risk_arr[t]
            if suffix_true:
                stable_t = float(t)
        return stable_t
    
    tau_G2G3_stable = compute_stable_time(G2G3_arr, tau_int)
    tau_G4_stable = compute_stable_time(G4_arr, tau_int)
    tau_G5_stable = compute_stable_time(G5_arr, tau_int)
    tau_Ghat_stable = compute_stable_time(Ghat_arr, tau_int)
    
    # =========================================================================
    # PHASE DURATIONS (only meaningful for successful runs)
    # =========================================================================
    if np.isfinite(tau) and np.isfinite(tau_Ghat_stable):
        phase1_duration = tau_Ghat_stable
        phase2_duration = tau - tau_Ghat_stable
        phase_ratio = phase2_duration / phase1_duration if phase1_duration > 0 else np.inf
    else:
        phase1_duration = np.nan
        phase2_duration = np.nan
        phase_ratio = np.nan
    
    # =========================================================================
    # L-CONSECUTIVE CLUSTERING STABILITY (more robust than suffix)
    # T_conc(L) = first t where C_t = C_{t+1} = ... = C_{t+L-1} = True
    # =========================================================================
    T_conc_5 = compute_T_conc_L(G2G3_arr, L=5)
    T_conc_10 = compute_T_conc_L(G2G3_arr, L=10)
    T_conc_20 = compute_T_conc_L(G2G3_arr, L=20)
    
    # Separation: Delta = tau_hit - T_conc(L)
    # Positive Delta means clustering happened before hitting
    if np.isfinite(tau):
        delta_5 = tau - T_conc_5 if np.isfinite(T_conc_5) else np.nan
        delta_10 = tau - T_conc_10 if np.isfinite(T_conc_10) else np.nan
        delta_20 = tau - T_conc_20 if np.isfinite(T_conc_20) else np.nan
    else:
        delta_5 = delta_10 = delta_20 = np.nan
    
    # =========================================================================
    # 3-STATE TRAJECTORY for occupancy plot
    # State 0: not clustered, not hit (C=0, H=0)
    # State 1: clustered, not hit (C=1, H=0)  
    # State 2: hit (H=1) - absorbing
    # =========================================================================
    H_arr = (df["f_best"].values <= f_star + eps_in).astype(int)
    states = np.zeros(len(df), dtype=int)
    hit_yet = False
    for t in range(len(df)):
        if hit_yet or H_arr[t]:
            states[t] = 2
            hit_yet = True
        elif G2G3_arr[t]:
            states[t] = 1
        else:
            states[t] = 0
    
    # At-risk statistics (t < tau)
    at_risk = df[df["t"] < df["tau"]]
    n_at_risk = len(at_risk)
    
    if n_at_risk > 0:
        G2G3_rate = at_risk["G2G3"].mean()
        G4_rate = at_risk["G4"].mean()
        G5_rate = at_risk["G5"].mean()
        Ghat_rate = at_risk["Ghat"].mean()
        gamma0_mean = at_risk["gamma0"].mean()
        
        # "Any" flags for sparse events (at-risk only)
        any_G4 = bool(at_risk["G4"].any())
        any_Ghat = bool(at_risk["Ghat"].any())
        any_gamma0_pos = bool((at_risk["gamma0"] > 0).any())
    else:
        G2G3_rate = G4_rate = G5_rate = Ghat_rate = gamma0_mean = np.nan
        any_G4 = any_Ghat = any_gamma0_pos = False
    
    # Estimate T_wit (stabilization over entire run, not just at-risk)
    Ghat_seq = df["Ghat"].tolist()
    Twit = estimate_Twit(Ghat_seq)
    
    run_summary = {
        "run": run_idx,
        "tau": tau,
        "tau_out": tau_out,
        "hit": np.isfinite(tau),
        "hit_out": np.isfinite(tau_out),
        "n_at_risk": n_at_risk,
        # Rates
        "G2G3_rate": G2G3_rate,
        "G4_rate": G4_rate,
        "G5_rate": G5_rate,
        "Ghat_rate": Ghat_rate,
        "gamma0_mean": gamma0_mean,
        # Any flags
        "any_G4": any_G4,
        "any_Ghat": any_Ghat,
        "any_gamma0_pos": any_gamma0_pos,
        # First occurrence times
        "tau_G2G3_first": tau_G2G3_first,
        "tau_G4_first": tau_G4_first,
        "tau_G5_first": tau_G5_first,
        "tau_Ghat_first": tau_Ghat_first,
        # Stabilization times (at-risk restricted)
        "tau_G2G3_stable": tau_G2G3_stable,
        "tau_G4_stable": tau_G4_stable,
        "tau_G5_stable": tau_G5_stable,
        "tau_Ghat_stable": tau_Ghat_stable,
        # L-consecutive clustering stability
        "T_conc_5": T_conc_5,
        "T_conc_10": T_conc_10,
        "T_conc_20": T_conc_20,
        # Separation: Delta = tau - T_conc(L)
        "delta_5": delta_5,
        "delta_10": delta_10,
        "delta_20": delta_20,
        # Phase durations
        "phase1_duration": phase1_duration,
        "phase2_duration": phase2_duration,
        "phase_ratio": phase_ratio,
        # T_wit (full run)
        "Twit": Twit,
    }
    
    # Add state to each row for occupancy analysis
    for i, row in enumerate(rows):
        row["state"] = int(states[i])
    
    return {
        "rows": rows,
        "run_summary": run_summary,
    }


# =============================================================================
# SECTION 8: Parallel Function Analysis
# =============================================================================

def analyze_function_parallel(
    runs: List[Dict],
    f_star: float,
    params: GhatParams,
    eps_in: float,
    max_gens: Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[pd.DataFrame], pd.DataFrame]:
    """
    Analyze all runs for a function with parallel processing.
    
    Parameters
    ----------
    runs : list of dict
        All runs for this function.
    f_star : float
        Global optimum.
    params : GhatParams
        Verification parameters.
    eps_in : float
        Target tolerance for hitting time.
    max_gens : int or None
        Maximum generations to analyze.
    n_jobs : int
        Number of parallel workers. -1 = all CPUs, 1 = sequential.
    verbose : bool
        Print progress.
    
    Returns
    -------
    summary : dict
        Aggregated statistics (mean/median of per-run rates).
    all_dfs : list of DataFrame
        Per-run DataFrames.
    per_run_df : DataFrame
        Per-run summary statistics.
    """
    n_runs = len(runs)
    
    # Determine number of workers
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = min(n_jobs, n_runs)
    
    # Convert params to dict for pickling
    params_dict = asdict(params)
    
    # Prepare arguments for workers
    worker_args = [
        (i, run, f_star, params_dict, eps_in, max_gens)
        for i, run in enumerate(runs)
    ]
    
    all_results = []
    
    if n_jobs == 1:
        # Sequential processing
        if verbose:
            print(f"Processing {n_runs} runs sequentially...")
        for i, args in enumerate(worker_args):
            result = analyze_run_worker(args)
            if result is not None:
                all_results.append(result)
            if verbose and (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_runs} runs")
    else:
        # Parallel processing
        if verbose:
            print(f"Processing {n_runs} runs with {n_jobs} workers...")
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            futures = {executor.submit(analyze_run_worker, args): i 
                       for i, args in enumerate(worker_args)}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_results.append(result)
                completed += 1
                if verbose and completed % 10 == 0:
                    print(f"  Completed {completed}/{n_runs} runs")
        
        if verbose:
            print(f"  Completed {n_runs}/{n_runs} runs")
    
    if not all_results:
        return {"error": "No valid runs"}, [], pd.DataFrame()
    
    # Collect results
    all_rows = []
    per_run_rows = []
    
    for result in all_results:
        all_rows.extend(result["rows"])
        per_run_rows.append(result["run_summary"])
    
    # Create DataFrames
    all_dfs = [pd.DataFrame(all_rows)]
    per_run_df = pd.DataFrame(per_run_rows)
    
    # Sort by run index
    per_run_df = per_run_df.sort_values("run").reset_index(drop=True)
    
    # Compute hitting times array
    taus = per_run_df["tau"].values
    n_hits = int(np.isfinite(taus).sum())
    
    # Outer hitting times
    taus_out = per_run_df["tau_out"].values
    n_hits_out = int(np.isfinite(taus_out).sum())
    
    # T_wit finite count
    Twit_values = per_run_df["Twit"].values
    n_finite_Twit = int(np.isfinite(Twit_values).sum())
    
    # =========================================================================
    # PHASE ANALYSIS: First occurrence and stabilization statistics
    # =========================================================================
    
    # First occurrence times (across all runs)
    tau_G2G3_first_arr = per_run_df["tau_G2G3_first"].values
    tau_G4_first_arr = per_run_df["tau_G4_first"].values
    tau_Ghat_first_arr = per_run_df["tau_Ghat_first"].values
    
    n_ever_G2G3 = int(np.isfinite(tau_G2G3_first_arr).sum())
    n_ever_G4 = int(np.isfinite(tau_G4_first_arr).sum())
    n_ever_Ghat = int(np.isfinite(tau_Ghat_first_arr).sum())
    
    # Stabilization times (at-risk restricted)
    tau_Ghat_stable_arr = per_run_df["tau_Ghat_stable"].values
    n_stable_Ghat = int(np.isfinite(tau_Ghat_stable_arr).sum())
    
    # Phase durations (only for successful runs with finite stabilization)
    phase1_arr = per_run_df["phase1_duration"].values
    phase2_arr = per_run_df["phase2_duration"].values
    phase_ratio_arr = per_run_df["phase_ratio"].values
    
    valid_phases = np.isfinite(phase1_arr) & np.isfinite(phase2_arr)
    n_valid_phases = int(valid_phases.sum())
    
    # Aggregate: mean/median of per-run rates
    summary = {
        "n_runs": n_runs,
        "n_hits": n_hits,
        "hit_rate": n_hits / n_runs if n_runs > 0 else 0.0,
        "tau_median": float(np.nanmedian(taus[np.isfinite(taus)])) if n_hits > 0 else np.nan,
        "tau_mean": float(np.nanmean(taus[np.isfinite(taus)])) if n_hits > 0 else np.nan,
        
        # Outer hitting (tau_out)
        "n_hits_out": n_hits_out,
        "hit_rate_out": n_hits_out / n_runs if n_runs > 0 else 0.0,
        "tau_out_median": float(np.nanmedian(taus_out[np.isfinite(taus_out)])) if n_hits_out > 0 else np.nan,
        "tau_out_mean": float(np.nanmean(taus_out[np.isfinite(taus_out)])) if n_hits_out > 0 else np.nan,
        
        # Per-run rate aggregation
        "G2G3_rate_mean": float(per_run_df["G2G3_rate"].mean()),
        "G2G3_rate_median": float(per_run_df["G2G3_rate"].median()),
        "G4_rate_mean": float(per_run_df["G4_rate"].mean()),
        "G4_rate_median": float(per_run_df["G4_rate"].median()),
        "G5_rate_mean": float(per_run_df["G5_rate"].mean()),
        "G5_rate_median": float(per_run_df["G5_rate"].median()),
        "Ghat_rate_mean": float(per_run_df["Ghat_rate"].mean()),
        "Ghat_rate_median": float(per_run_df["Ghat_rate"].median()),
        "gamma0_mean": float(per_run_df["gamma0_mean"].mean()),
        
        # "Any" counters for sparse events
        "runs_with_any_G4": int(per_run_df["any_G4"].sum()),
        "runs_with_any_Ghat": int(per_run_df["any_Ghat"].sum()),
        "runs_with_any_gamma0_pos": int(per_run_df["any_gamma0_pos"].sum()),
        
        # =====================================================================
        # PHASE ANALYSIS
        # =====================================================================
        
        # First occurrence counts
        "n_ever_G2G3": n_ever_G2G3,
        "n_ever_G4": n_ever_G4,
        "n_ever_Ghat": n_ever_Ghat,
        
        # First occurrence times (median over runs that have it)
        "tau_G2G3_first_median": float(np.nanmedian(tau_G2G3_first_arr[np.isfinite(tau_G2G3_first_arr)])) if n_ever_G2G3 > 0 else np.nan,
        "tau_G4_first_median": float(np.nanmedian(tau_G4_first_arr[np.isfinite(tau_G4_first_arr)])) if n_ever_G4 > 0 else np.nan,
        "tau_Ghat_first_median": float(np.nanmedian(tau_Ghat_first_arr[np.isfinite(tau_Ghat_first_arr)])) if n_ever_Ghat > 0 else np.nan,
        
        # Stabilization (Phase 2 entry)
        "n_stable_Ghat": n_stable_Ghat,
        "tau_Ghat_stable_median": float(np.nanmedian(tau_Ghat_stable_arr[np.isfinite(tau_Ghat_stable_arr)])) if n_stable_Ghat > 0 else np.nan,
        "tau_Ghat_stable_mean": float(np.nanmean(tau_Ghat_stable_arr[np.isfinite(tau_Ghat_stable_arr)])) if n_stable_Ghat > 0 else np.nan,
        
        # Phase durations
        "n_valid_phases": n_valid_phases,
        "phase1_median": float(np.nanmedian(phase1_arr[valid_phases])) if n_valid_phases > 0 else np.nan,
        "phase1_mean": float(np.nanmean(phase1_arr[valid_phases])) if n_valid_phases > 0 else np.nan,
        "phase2_median": float(np.nanmedian(phase2_arr[valid_phases])) if n_valid_phases > 0 else np.nan,
        "phase2_mean": float(np.nanmean(phase2_arr[valid_phases])) if n_valid_phases > 0 else np.nan,
        "phase_ratio_median": float(np.nanmedian(phase_ratio_arr[valid_phases])) if n_valid_phases > 0 else np.nan,
        "phase_ratio_mean": float(np.nanmean(phase_ratio_arr[valid_phases])) if n_valid_phases > 0 else np.nan,
        
        # =====================================================================
        # TWO-PERIOD ANALYSIS: L-consecutive clustering stability
        # =====================================================================
        # T_conc(L) = first t where C_t = C_{t+1} = ... = C_{t+L-1} = True
        "T_conc_10_median": float(per_run_df["T_conc_10"].replace(np.inf, np.nan).median()),
        "T_conc_10_mean": float(per_run_df["T_conc_10"].replace(np.inf, np.nan).mean()),
        "n_finite_T_conc_10": int(np.isfinite(per_run_df["T_conc_10"].values).sum()),
        
        # Separation: Delta = tau - T_conc(L)  [only for successful runs]
        # Positive Delta means clustering happened before hitting
        "delta_10_median": float(per_run_df["delta_10"].median()) if per_run_df["delta_10"].notna().any() else np.nan,
        "delta_10_mean": float(per_run_df["delta_10"].mean()) if per_run_df["delta_10"].notna().any() else np.nan,
        "n_positive_delta_10": int((per_run_df["delta_10"] > 0).sum()),
        "p_delta_positive_10": float((per_run_df["delta_10"] > 0).sum() / n_hits) if n_hits > 0 else np.nan,
        
        # T_wit statistics
        "n_finite_Twit": n_finite_Twit,
        "Twit_median": float(per_run_df["Twit"].replace(np.inf, np.nan).median()),
        "Twit_mean": float(per_run_df["Twit"].replace(np.inf, np.nan).mean()),
        "Twit_inf_rate": float((per_run_df["Twit"] == np.inf).mean()),
        
        "n_runs_with_at_risk": int((per_run_df["n_at_risk"] > 0).sum()),
    }
    
    return summary, all_dfs, per_run_df


# =============================================================================
# SECTION 9: Visualization
# =============================================================================

def plot_condition_rates(
    combined_df: pd.DataFrame,
    out_png: Path,
    title: str = "Condition Satisfaction Rates"
) -> None:
    """Plot condition satisfaction rates over generations."""
    if combined_df.empty:
        return
    
    grouped = combined_df.groupby("t").agg({
        "G2G3": "mean",
        "G4": "mean",
        "G5": "mean",
        "Ghat": "mean",
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(grouped["t"], grouped["G2G3"], label="G2-G3 (Concentration)", alpha=0.8)
    plt.plot(grouped["t"], grouped["G4"], label="G4 (Sublevel)", alpha=0.8)
    plt.plot(grouped["t"], grouped["G5"], label="G5 (Good Memory)", alpha=0.8)
    plt.plot(grouped["t"], grouped["Ghat"], label=r"$\hat{G}_t$ (Combined)", linewidth=2, color="black")
    
    plt.xlabel("Generation t")
    plt.ylabel("Satisfaction Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_gamma0_evolution(
    combined_df: pd.DataFrame,
    out_png: Path,
    title: str = r"Witness Probability $\gamma_0$"
) -> None:
    """Plot gamma_0 evolution over generations."""
    if combined_df.empty:
        return
    
    pos_gamma = combined_df[combined_df["gamma0"] > 0]
    if pos_gamma.empty:
        return
    
    grouped = pos_gamma.groupby("t").agg({
        "gamma0": ["mean", "median"],
    }).reset_index()
    grouped.columns = ["t", "mean", "median"]
    
    plt.figure(figsize=(10, 6))
    
    plt.semilogy(grouped["t"], grouped["mean"], label="Mean", alpha=0.8)
    plt.semilogy(grouped["t"], grouped["median"], label="Median", alpha=0.8)
    
    plt.xlabel("Generation t")
    plt.ylabel(r"$\gamma_0$ (log scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_phase_transition(
    combined_df: pd.DataFrame,
    per_run_df: pd.DataFrame,
    out_png: Path,
    title: str = "Phase Transition Analysis"
) -> None:
    """
    Plot phase transition: P(Ghat | t) over generations.
    
    Shows sigmoid-like transition from exploration to exploitation phase.
    Also shows vertical lines for median transition times.
    """
    if combined_df.empty:
        return
    
    # Compute P(condition | t) for each generation
    grouped = combined_df.groupby("t").agg({
        "G2G3": "mean",
        "G4": "mean",
        "G5": "mean",
        "Ghat": "mean",
    }).reset_index()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top panel: Condition probabilities over time
    ax1 = axes[0]
    ax1.plot(grouped["t"], grouped["Ghat"], label=r"$P(\hat{G}_t)$", linewidth=2, color="black")
    ax1.plot(grouped["t"], grouped["G2G3"], label="P(G2G3)", alpha=0.7)
    ax1.plot(grouped["t"], grouped["G4"], label="P(G4)", alpha=0.7)
    ax1.plot(grouped["t"], grouped["G5"], label="P(G5)", alpha=0.7)
    
    # Add vertical lines for median transition times
    tau_Ghat_stable_arr = per_run_df["tau_Ghat_stable"].values
    finite_stable = tau_Ghat_stable_arr[np.isfinite(tau_Ghat_stable_arr)]
    if len(finite_stable) > 0:
        med_stable = np.median(finite_stable)
        ax1.axvline(med_stable, color="red", linestyle="--", linewidth=2, 
                    label=f"Median $\\tau_{{stable}}^{{\\hat{{G}}}}$ = {med_stable:.0f}")
    
    tau_arr = per_run_df["tau"].values
    finite_tau = tau_arr[np.isfinite(tau_arr)]
    if len(finite_tau) > 0:
        med_tau = np.median(finite_tau)
        ax1.axvline(med_tau, color="green", linestyle="--", linewidth=2,
                    label=f"Median $\\tau$ = {med_tau:.0f}")
    
    ax1.set_ylabel("Probability")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Bottom panel: Number of runs at risk
    at_risk_counts = combined_df.groupby("t").apply(
        lambda x: (x["t"] < x["tau"]).sum()
    ).reset_index(name="n_at_risk")
    
    ax2 = axes[1]
    ax2.fill_between(at_risk_counts["t"], at_risk_counts["n_at_risk"], alpha=0.5, label="Runs at risk")
    ax2.set_xlabel("Generation t")
    ax2.set_ylabel("Number of runs at risk")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_phase_durations(
    per_run_df: pd.DataFrame,
    out_png: Path,
    title: str = "Phase Duration Analysis"
) -> None:
    """
    Plot phase duration scatter and histograms.
    
    Shows relationship between Phase 1 (exploration) and Phase 2 (exploitation).
    """
    # Filter to runs with valid phase data
    valid = per_run_df[
        np.isfinite(per_run_df["phase1_duration"]) & 
        np.isfinite(per_run_df["phase2_duration"])
    ].copy()
    
    if len(valid) < 2:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Scatter plot of Phase 1 vs Phase 2
    ax1 = axes[0]
    ax1.scatter(valid["phase1_duration"], valid["phase2_duration"], alpha=0.6, s=50)
    
    # Add diagonal line (equal phases)
    max_val = max(valid["phase1_duration"].max(), valid["phase2_duration"].max())
    ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label="Equal phases")
    
    ax1.set_xlabel("Phase 1 (Exploration)")
    ax1.set_ylabel("Phase 2 (Exploitation)")
    ax1.set_title("Phase 1 vs Phase 2 Duration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Histogram of phase ratio
    ax2 = axes[1]
    ratios = valid["phase_ratio"].values
    ratios = ratios[np.isfinite(ratios) & (ratios < 100)]  # Clip extreme values
    if len(ratios) > 0:
        ax2.hist(ratios, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(np.median(ratios), color='red', linestyle='--', 
                    label=f"Median = {np.median(ratios):.2f}")
    ax2.set_xlabel("Phase Ratio (Phase 2 / Phase 1)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Phase Ratios")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Timeline visualization
    ax3 = axes[2]
    valid_sorted = valid.sort_values("tau").reset_index(drop=True)
    for i, row in valid_sorted.iterrows():
        p1 = row["phase1_duration"]
        p2 = row["phase2_duration"]
        ax3.barh(i, p1, color="steelblue", alpha=0.7, label="Phase 1" if i == 0 else "")
        ax3.barh(i, p2, left=p1, color="coral", alpha=0.7, label="Phase 2" if i == 0 else "")
    
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Run (sorted by τ)")
    ax3.set_title("Phase Timeline per Run")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(title, fontsize=14, y=1.02)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()


def plot_state_occupancy(
    combined_df: pd.DataFrame,
    out_png: Path,
    title: str = "3-State Occupancy Over Time"
) -> None:
    """
    Plot 3-state occupancy as stacked area chart.
    
    State 0: not clustered, not hit (exploration)
    State 1: clustered, not hit (exploitation phase)
    State 2: hit (absorbing)
    
    A clear "State 1 plateau" demonstrates two distinct periods.
    """
    if combined_df.empty or "state" not in combined_df.columns:
        return
    
    # Compute fraction in each state per generation
    state_counts = combined_df.groupby("t")["state"].value_counts().unstack(fill_value=0)
    
    # Ensure all states are present
    for s in [0, 1, 2]:
        if s not in state_counts.columns:
            state_counts[s] = 0
    state_counts = state_counts[[0, 1, 2]]
    
    # Convert to fractions
    state_fracs = state_counts.div(state_counts.sum(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Stacked area plot
    ax.stackplot(
        state_fracs.index,
        state_fracs[0], state_fracs[1], state_fracs[2],
        labels=[
            "State 0: Not clustered, not hit",
            "State 1: Clustered, not hit",
            "State 2: Hit (absorbing)"
        ],
        colors=["lightcoral", "steelblue", "forestgreen"],
        alpha=0.8
    )
    
    ax.set_xlabel("Generation t")
    ax.set_ylabel("Fraction of runs")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_stratified_hazard(
    combined_df: pd.DataFrame,
    out_png: Path,
    title: str = "Stratified Hazard Analysis"
) -> None:
    """
    Plot empirical hazard stratified by clustering status.
    
    h_t^(0): hazard when C_t = 0 (not clustered)
    h_t^(1): hazard when C_t = 1 (clustered)
    
    If h^(1) >> h^(0), clustering enables hitting.
    """
    if combined_df.empty:
        return
    
    # Need to identify hitting events
    # A hit at generation t means: tau == t (this run hit at generation t)
    combined_df = combined_df.copy()
    combined_df["hit_at_t"] = combined_df["t"] == combined_df["tau"]
    combined_df["at_risk"] = combined_df["t"] < combined_df["tau"]
    
    # Compute hazard by generation and clustering status
    # For each (t, C_t), count:
    #   - n_at_risk: runs with at_risk=True
    #   - n_hits: runs with hit_at_t=True (actually tau == t+1, but we approximate)
    
    # Bin generations to get smoother estimates
    max_t = int(combined_df["t"].max())
    bin_size = max(1, max_t // 50)  # ~50 bins
    combined_df["t_bin"] = (combined_df["t"] // bin_size) * bin_size
    
    # Stratify by G2G3 (clustering)
    results = []
    for clustered in [False, True]:
        subset = combined_df[combined_df["G2G3"] == clustered]
        
        for t_bin in sorted(subset["t_bin"].unique()):
            bin_data = subset[subset["t_bin"] == t_bin]
            n_at_risk = bin_data["at_risk"].sum()
            n_hits = bin_data["hit_at_t"].sum()
            
            if n_at_risk > 10:  # Require minimum sample size
                hazard = n_hits / n_at_risk
                results.append({
                    "t_bin": t_bin,
                    "clustered": clustered,
                    "n_at_risk": n_at_risk,
                    "n_hits": n_hits,
                    "hazard": hazard
                })
    
    if not results:
        return
    
    hazard_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel 1: Hazard curves
    ax1 = axes[0]
    for clustered, label, color in [(False, "Not clustered (C=0)", "lightcoral"),
                                     (True, "Clustered (C=1)", "steelblue")]:
        data = hazard_df[hazard_df["clustered"] == clustered]
        if not data.empty:
            ax1.plot(data["t_bin"], data["hazard"], 'o-', label=label, color=color, alpha=0.8)
    
    ax1.set_xlabel("Generation (binned)")
    ax1.set_ylabel("Empirical hazard $h_t$")
    ax1.set_title("Hazard by Clustering Status")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Panel 2: Hazard ratio
    ax2 = axes[1]
    pivot = hazard_df.pivot(index="t_bin", columns="clustered", values="hazard")
    if False in pivot.columns and True in pivot.columns:
        ratio = pivot[True] / pivot[False].replace(0, np.nan)
        ratio = ratio.dropna()
        if not ratio.empty:
            ax2.bar(ratio.index, ratio.values, width=bin_size*0.8, alpha=0.7, color="purple")
            ax2.axhline(1, color='black', linestyle='--', label="HR=1 (no effect)")
            median_hr = ratio.median()
            ax2.axhline(median_hr, color='red', linestyle='--', 
                        label=f"Median HR = {median_hr:.1f}")
    
    ax2.set_xlabel("Generation (binned)")
    ax2.set_ylabel("Hazard Ratio $h^{(1)}/h^{(0)}$")
    ax2.set_title("Hazard Ratio: Clustered / Not Clustered")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.suptitle(title, fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_separation_histogram(
    per_run_df: pd.DataFrame,
    out_png: Path,
    title: str = "Separation Analysis: Δ = τ - T_conc"
) -> None:
    """
    Plot histogram of separation Δ = τ_hit - T_conc(L).
    
    Positive Δ means clustering happened before hitting.
    A large positive median Δ with high p_{Δ>0} proves two-period structure.
    """
    delta_10 = per_run_df["delta_10"].dropna().values
    
    if len(delta_10) < 2:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: Histogram of Δ
    ax1 = axes[0]
    ax1.hist(delta_10, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label="Δ=0 (simultaneous)")
    ax1.axvline(np.median(delta_10), color='green', linestyle='--', linewidth=2,
                label=f"Median Δ = {np.median(delta_10):.0f}")
    
    p_positive = (delta_10 > 0).mean()
    ax1.set_xlabel("Δ = τ - T_conc(10)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Separation Distribution (p_{{Δ>0}} = {p_positive:.2%})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: T_conc vs τ scatter
    ax2 = axes[1]
    valid = per_run_df[
        np.isfinite(per_run_df["T_conc_10"]) & 
        np.isfinite(per_run_df["tau"])
    ]
    if not valid.empty:
        ax2.scatter(valid["T_conc_10"], valid["tau"], alpha=0.6, s=50)
        
        # Add diagonal (T_conc = τ)
        max_val = max(valid["T_conc_10"].max(), valid["tau"].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', label="T_conc = τ")
        
        ax2.set_xlabel("T_conc(10) (Clustering stabilizes)")
        ax2.set_ylabel("τ (Hitting time)")
        ax2.set_title("Clustering Time vs Hitting Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =============================================================================
# SECTION 10: Main CLI
# =============================================================================

def load_pkl(pkl_path: Path) -> Dict:
    """Load PKL file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_func_key(data: Dict, func: str) -> str:
    """Get function key from data dict."""
    func = func.strip()
    if func in data:
        return func
    
    if func.startswith("f") and func[1:].isdigit():
        key = f"cec2017_{func}"
        if key in data:
            return key
    
    for k in data.keys():
        if k.endswith(f"_{func}") or k == func:
            return k
    
    raise KeyError(f"Function {func} not found in PKL. Available: {list(data.keys())}")


def f_star_from_func(func: str) -> float:
    """Get f* from function name (CEC2017 convention: f* = 100*i)."""
    import re
    m = re.search(r"f(\d+)", func)
    if not m:
        raise ValueError(f"Cannot parse function id from {func}")
    return float(100 * int(m.group(1)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified witness-condition verification for L-SHADE (Ghat_t) - PARALLEL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parallel with 16 workers:
  python check_Gt.py --pkl f22.pkl --func f22 --dim 10 --eps_in 10 --eps_out 100 --r_conc 2.0 --jobs 16 --outdir results/

  # Use all CPUs:
  python check_Gt.py --pkl f22.pkl --func f22 --dim 10 --eps_in 10 --r_conc 2.0 --jobs -1 --outdir results/

  # Sequential (for debugging):
  python check_Gt.py --pkl f22.pkl --func f22 --dim 10 --eps_in 10 --r_conc 2.0 --jobs 1 --outdir results/

Notes:
  - Ghat_t = (G2 ∧ G3 ∧ G4 ∧ G5) is the empirical proxy for G_t
  - G1 (local basin membership) is assumed, not verified
  - r_conc is a DIAMETER bound; cluster uses radius = r_conc/2
  - eps_out should be > eps_in for meaningful at-risk analysis
        """
    )
    
    # Input/output
    ap.add_argument("--pkl", required=True, help="Path to PKL file")
    ap.add_argument("--func", required=True, help="Function name (e.g., f1, f22)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    
    # Problem parameters
    ap.add_argument("--dim", type=int, required=True, help="Dimension")
    ap.add_argument("--eps_in", type=float, required=True, 
                    help="Target tolerance for hitting time tau (inner)")
    ap.add_argument("--eps_out", type=float, default=None,
                    help="Outer tolerance for G4 (default: 4 * eps_in)")
    
    # G5 parameters
    ap.add_argument("--F_minus", type=float, default=0.1)
    ap.add_argument("--F_plus", type=float, default=0.9)
    ap.add_argument("--g_thresh", type=float, default=0.01)
    ap.add_argument("--c_cr", type=float, default=0.5)
    ap.add_argument("--q_thresh", type=float, default=0.1)
    
    # G2-G3 parameters
    ap.add_argument("--r_conc", type=float, default=2.0,
                    help="Concentration DIAMETER bound (default: 2.0)")
    ap.add_argument("--m_min", type=int, default=4)
    
    # Parallelization
    ap.add_argument("--jobs", type=int, default=-1,
                    help="Number of parallel workers. -1 = all CPUs, 1 = sequential (default: -1)")
    
    # Analysis parameters
    ap.add_argument("--max_gens", type=int, default=None)
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = ap.parse_args()
    
    # Handle eps_out default
    if args.eps_out is None:
        args.eps_out = 4.0 * args.eps_in
        if not args.quiet:
            print(f"\n[INFO] eps_out not provided; using eps_out = 4 × eps_in = {args.eps_out}")
    
    # Validate eps_out > eps_in
    if args.eps_out <= args.eps_in:
        print(f"\n{'!'*70}")
        print(f"WARNING: eps_out ({args.eps_out}) <= eps_in ({args.eps_in})")
        print(f"G4 will NEVER hold before hitting time tau!")
        print(f"Recommendation: use eps_out > eps_in (e.g., eps_out = 4*eps_in)")
        print(f"{'!'*70}\n")
    
    # Load data
    pkl_path = Path(args.pkl)
    if not args.quiet:
        print(f"\nLoading {pkl_path}...")
    data = load_pkl(pkl_path)
    func_key = get_func_key(data, args.func)
    runs = data[func_key]
    f_star = f_star_from_func(func_key)
    
    if not args.quiet:
        print(f"\n{'='*70}")
        print(f"WITNESS CONDITION VERIFICATION (Ghat_t) - PARALLEL")
        print(f"{'='*70}")
        print(f"Function: {func_key}")
        print(f"f*: {f_star}")
        print(f"Runs: {len(runs)}")
        print(f"Dimension: {args.dim}")
        print(f"eps_in: {args.eps_in}, eps_out: {args.eps_out}")
        print(f"r_conc: {args.r_conc} (diameter; radius = {args.r_conc/2})")
        print(f"Jobs: {args.jobs if args.jobs != -1 else 'all CPUs'}")
        print(f"\nNote: Ghat_t = (G2 ∧ G3 ∧ G4 ∧ G5). G1 is assumed.")
    
    # Build parameters
    params = GhatParams(
        F_minus=args.F_minus,
        F_plus=args.F_plus,
        g_thresh=args.g_thresh,
        c_cr=args.c_cr,
        q_thresh=args.q_thresh,
        r_conc=args.r_conc,
        m_min=args.m_min,
        eps_out=args.eps_out,
    )
    
    # Run analysis
    summary, all_dfs, per_run_df = analyze_function_parallel(
        runs=runs,
        f_star=f_star,
        params=params,
        eps_in=args.eps_in,
        max_gens=args.max_gens,
        n_jobs=args.jobs,
        verbose=not args.quiet
    )
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary["function"] = func_key
    summary["eps_in"] = args.eps_in
    summary["eps_out"] = args.eps_out
    summary["dim"] = args.dim
    summary["r_conc"] = args.r_conc
    summary["m_min"] = args.m_min
    
    summary_path = outdir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, 
                  default=lambda x: float(x) if isinstance(x, np.floating) else 
                                    (None if (isinstance(x, float) and np.isnan(x)) else x))
    
    if not args.quiet:
        print(f"\n{'='*70}")
        print("SUMMARY (per-run rate aggregation)")
        print(f"{'='*70}")
        for k, v in summary.items():
            if isinstance(v, float):
                if np.isnan(v):
                    print(f"  {k}: nan")
                else:
                    print(f"  {k}: {v:.4g}")
            else:
                print(f"  {k}: {v}")
    
    # Save per-run summary
    if not per_run_df.empty:
        per_run_path = outdir / "per_run_summary.csv"
        per_run_df.to_csv(per_run_path, index=False)
        if not args.quiet:
            print(f"\nSaved: {per_run_path}")
    
    # Save per-generation data
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values(["run", "t"]).reset_index(drop=True)
        csv_path = outdir / "per_generation.csv"
        combined.to_csv(csv_path, index=False)
        if not args.quiet:
            print(f"Saved: {csv_path}")
        
        # Generate plots
        plot_condition_rates(
            combined, 
            outdir / "Ghat_rates.png",
            title=f"{func_key} Condition Satisfaction (eps_in={args.eps_in}, eps_out={args.eps_out})"
        )
        if not args.quiet:
            print(f"Saved: {outdir / 'Ghat_rates.png'}")
        
        plot_gamma0_evolution(
            combined,
            outdir / "gamma0_evolution.png",
            title=f"{func_key} Witness Probability (eps_in={args.eps_in})"
        )
        if not args.quiet:
            print(f"Saved: {outdir / 'gamma0_evolution.png'}")
        
        # Phase transition analysis plots
        plot_phase_transition(
            combined,
            per_run_df,
            outdir / "phase_transition.png",
            title=f"{func_key} Phase Transition (eps_in={args.eps_in}, eps_out={args.eps_out})"
        )
        if not args.quiet:
            print(f"Saved: {outdir / 'phase_transition.png'}")
        
        plot_phase_durations(
            per_run_df,
            outdir / "phase_durations.png",
            title=f"{func_key} Phase Durations"
        )
        if not args.quiet:
            print(f"Saved: {outdir / 'phase_durations.png'}")
        
        # Two-period analysis plots
        plot_state_occupancy(
            combined,
            outdir / "state_occupancy.png",
            title=f"{func_key} 3-State Occupancy (eps_in={args.eps_in})"
        )
        if not args.quiet:
            print(f"Saved: {outdir / 'state_occupancy.png'}")
        
        plot_stratified_hazard(
            combined,
            outdir / "stratified_hazard.png",
            title=f"{func_key} Stratified Hazard Analysis"
        )
        if not args.quiet:
            print(f"Saved: {outdir / 'stratified_hazard.png'}")
        
        plot_separation_histogram(
            per_run_df,
            outdir / "separation_analysis.png",
            title=f"{func_key} Separation: Δ = τ - T_conc(10)"
        )
        if not args.quiet:
            print(f"Saved: {outdir / 'separation_analysis.png'}")
    
    if not args.quiet:
        print(f"\nSaved: {summary_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
