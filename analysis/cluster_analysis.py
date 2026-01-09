#!/usr/bin/env python3
r"""
cluster_analysis.py

Population-concentration diagnostics aligned with the paper's "checkable proxy"
for the witness regime (Appendix app:witness-proxy).

Key observable:
  \widehat n_clust(t) = # { i in P^{(t)} : ||x_i^{(t)} - x_best^{(t)}|| <= r_conc }.

We report:
- n_clust(t) and beta1(t) = n_clust(t) / N_t
- diameter of the within-ball subset (optional; note diameter can be up to 2 r_conc)
- (optional) strict G2∧G3 checks via clique existence are in cluster_analysis_corrected.py

This module is designed to be used both standalone and from combined_analysis.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Core concentration metrics
# ----------------------------

def concentration_count(
    positions: np.ndarray,
    best_idx: int,
    r_conc: float,
) -> Tuple[int, float, float]:
    """
    Compute (n_clust, beta1, diameter) for a single generation.

    Parameters
    ----------
    positions : array (N,d)
    best_idx : int
        Index of the best individual in this generation.
    r_conc : float
        Concentration radius.

    Returns
    -------
    n_clust : int
        Number of individuals within r_conc of the best (including the best).
    beta1 : float
        Fraction n_clust / N.
    diameter : float
        Max pairwise distance among the within-ball subset (0 if subset size < 2).
        Note: This is a diagnostic; the within-ball subset does NOT guarantee
        diameter <= r_conc (it can be up to 2*r_conc).
    """
    X = np.asarray(positions, dtype=float)
    N = X.shape[0]
    if N == 0:
        return (0, 0.0, float("nan"))

    best_idx = int(best_idx)
    if best_idx < 0 or best_idx >= N:
        raise IndexError("best_idx out of range")

    x_best = X[best_idx]
    dists = np.linalg.norm(X - x_best, axis=1)

    mask = dists <= float(r_conc)
    idx = np.where(mask)[0]
    n_clust = int(idx.size)
    beta1 = float(n_clust / N)

    if n_clust < 2:
        diameter = 0.0
    else:
        Y = X[idx]
        # Compute max pairwise distance (O(m^2), m = n_clust)
        # For typical n_clust small, this is fine; for large n_clust, this can be slow.
        diff = Y[:, None, :] - Y[None, :, :]
        dist_mat = np.sqrt(np.sum(diff * diff, axis=2))
        diameter = float(np.max(dist_mat))

    return (n_clust, beta1, diameter)


# ----------------------------
# Per-run / per-epsilon summaries
# ----------------------------

def cluster_stats_at_hitting(
    run: Dict,
    tau: int,
    r_conc: float,
) -> Optional[Dict[str, float]]:
    """
    Compute concentration diagnostics at the hitting generation 'tau' for a single run.

    The run dict is assumed to have:
      run["history"]["positions"][t] : (N_t, d)
      run["history"]["fitness"][t]   : (N_t,)

    Returns a dict with n_clust, beta1, diameter; or None if missing data.
    """
    hist = run.get("history", {})
    pos_hist = hist.get("positions", None)
    fit_hist = hist.get("fitness", None)
    if pos_hist is None or fit_hist is None:
        return None
    if tau is None or not np.isfinite(tau):
        return None
    tau = int(tau)
    if tau < 0 or tau >= len(pos_hist):
        return None

    positions = np.asarray(pos_hist[tau])
    fitness = np.asarray(fit_hist[tau])
    if positions.ndim != 2 or fitness.ndim != 1:
        return None
    best_idx = int(np.argmin(fitness))

    n_clust, beta1, diameter = concentration_count(positions, best_idx=best_idx, r_conc=r_conc)

    return {
        "n_clust": float(n_clust),
        "beta1": float(beta1),
        "diameter": float(diameter),
        "N": float(positions.shape[0]),
    }


def summarize_cluster_at_hitting(
    runs: List[Dict],
    taus: np.ndarray,
    r_conc: float,
) -> Dict[str, float]:
    """
    Aggregate cluster statistics at hitting across runs (median of per-run values).

    Returns dict with median n_clust, median beta1, median diameter, and the number of hits used.
    """
    vals = []
    for run, tau in zip(runs, taus):
        if not np.isfinite(tau):
            continue
        st = cluster_stats_at_hitting(run, tau=int(tau), r_conc=r_conc)
        if st is not None:
            vals.append(st)

    if not vals:
        return {"n_hits": 0, "n_clust_median": np.nan, "beta1_median": np.nan, "diameter_median": np.nan}

    df = pd.DataFrame(vals)
    return {
        "n_hits": int(df.shape[0]),
        "n_clust_median": float(df["n_clust"].median()),
        "beta1_median": float(df["beta1"].median()),
        "diameter_median": float(df["diameter"].median()),
    }


# ----------------------------
# Distance ECDF diagnostic
# ----------------------------

def plot_distance_ecdf_at_hitting(
    runs: List[Dict],
    taus: np.ndarray,
    out_png: Path,
    max_points: int = 200_000,
) -> None:
    """
    Plot the ECDF of distances-to-best, pooling across runs at the hitting time.

    We collect distances from best to all population members at time tau for each run.
    """
    dist_list: List[np.ndarray] = []

    for run, tau in zip(runs, taus):
        if not np.isfinite(tau):
            continue
        hist = run.get("history", {})
        pos_hist = hist.get("positions", None)
        fit_hist = hist.get("fitness", None)
        if pos_hist is None or fit_hist is None:
            continue
        tau = int(tau)
        if tau < 0 or tau >= len(pos_hist):
            continue

        X = np.asarray(pos_hist[tau], dtype=float)
        f = np.asarray(fit_hist[tau], dtype=float)
        if X.ndim != 2 or f.ndim != 1:
            continue
        b = int(np.argmin(f))
        x_best = X[b]
        dists = np.linalg.norm(X - x_best, axis=1)
        dist_list.append(dists)

    if not dist_list:
        return

    d_all = np.concatenate(dist_list, axis=0)
    if d_all.size > max_points:
        rng = np.random.default_rng(0)
        d_all = rng.choice(d_all, size=max_points, replace=False)

    d_sorted = np.sort(d_all)
    y = np.linspace(0, 1, d_sorted.size, endpoint=False)

    plt.figure()
    plt.plot(d_sorted, y)
    plt.xlabel("Distance to best ||x - x_best||")
    plt.ylabel("ECDF")
    plt.title("Distance-to-best ECDF at hitting time (pooled)")
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
