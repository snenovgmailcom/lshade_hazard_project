#!/usr/bin/env python3
"""
morse_validate.py

Quasi-Morse validation script aligned with:
  - main: paper3d.tex (Theorem 5: thm:morse-hazard)
  - supp: morse-validation.tex

Validates the concentration-based hazard bound by tracking three critical times:
  - tau_deep: First time (C1) is satisfied (witness enters A_{eps/4})
  - tau_C2:   First time (C2) is satisfied (c1 >= 1, first neighbor in r_conc)
  - tau_pair: First time c1 >= 2 (pair bound becomes non-vacuous)

Key fixes implemented:
  Fix 1: Correct reference point (use witness = best deep hitter)
  Fix 2: Scan forward for tau_C2 (not just check at tau_deep)
  Fix 3: Track tau_pair (when c_pair > 0 becomes possible)

Usage:
  python morse_validate.py --dim 10 --func all --base experiments --outdir out --plots

  python morse_validate.py --dim 10 --func f1,f3,f5,f11 --base experiments \\
    --margin 60 --deep_ratio 0.25 --F_minus 0.1 --Delta_F 0.8 --outdir out --plots

Outputs:
  Per-run CSV:  {outdir}/morse_validate_D{D}_{func}.per_run.csv
  Summary CSV:  {outdir}/morse_validate_D{D}_{func}.summary.csv
  Plots (opt):  {outdir}/*.png

Expected PKL format:
  base/D{dim}/f{fid}/f{fid}.pkl
  Pickle contains dict with key "cec2017_f{fid}" -> list of runs.
  Each run dict should have:
    run["history"]["fitness"][t]    list/array length N
    run["history"]["positions"][t]  array (N,d)
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CEC2017 evaluator (same import pattern as benchmark.py) ---
from cec2017.functions import (
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
    f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
    f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
)

CEC_FUN_LIST = [
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
    f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
    f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
]


def cec_wrap(f, x: np.ndarray) -> float:
    """Allow (d,) or (n,d). Return scalar float."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    val = f(x)
    try:
        return float(val[0])
    except Exception:
        return float(val)


CEC_FUNCS = {i: partial(cec_wrap, fun) for i, fun in enumerate(CEC_FUN_LIST, start=1)}


# =============================================================================
# Utilities / IO
# =============================================================================

def ts(msg: str) -> str:
    return f"[{time.strftime('%H:%M:%S')}] {msg}"


def get_f_star_cec2017(fid: int) -> float:
    # Your convention (consistent with your logs)
    return 100.0 * fid


def load_runs(dim: int, fid: int, base: Path) -> Optional[List[Dict[str, Any]]]:
    pkl = base / f"D{dim}" / f"f{fid}" / f"f{fid}.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return data.get(f"cec2017_f{fid}", None)


def first_hit_time(fitness_hist: List[Any], thr: float) -> Optional[int]:
    """First index t where min(f_t) <= thr."""
    for t, f_t in enumerate(fitness_hist):
        arr = np.asarray(f_t, dtype=float)
        if arr.size and float(np.min(arr)) <= thr:
            return t
    return None


def extract_archive_positions(hist: Dict[str, Any], t: int) -> Optional[np.ndarray]:
    for key in ("archive_positions", "A_positions", "archive_pos", "archive"):
        seq = hist.get(key)
        if isinstance(seq, list) and t < len(seq):
            try:
                A_t = np.asarray(seq[t], dtype=float)
                if A_t.ndim == 2 and A_t.shape[0] > 0:
                    return A_t
            except Exception:
                pass
    return None


def extract_archive_size(hist: Dict[str, Any], t: int, X_arch: Optional[np.ndarray]) -> int:
    if isinstance(X_arch, np.ndarray):
        return int(X_arch.shape[0])
    for key in ("archive_size", "A_size", "archive_len"):
        seq = hist.get(key)
        if isinstance(seq, list) and t < len(seq):
            try:
                return int(seq[t])
            except Exception:
                pass
        if isinstance(seq, (int, float)):
            return int(seq)
    return 0


# =============================================================================
# Math layer
# =============================================================================

@dataclass
class ValidatorConfig:
    # eps_in Option 2
    margin: float = 60.0
    deep_ratio: float = 0.25
    q_out: float = 0.90  # quantile of (f-f*) at tau_deep to define eps_out

    # r_conc definition
    F_minus: float = 0.1
    Delta_F: float = 0.8
    r_safe_factor: float = float(1.0 - 1.0 / np.sqrt(2.0))

    # Local L estimation (isotropic)
    L_n_dirs: int = 64
    L_h: float = 1e-3        # relative to domain width (upper-lower)
    L_quantile: float = 0.90 # 0.90 robust; 1.0 -> max

    # Box bounds for CEC (used for step clipping)
    lower: float = -100.0
    upper: float = 100.0


class MathValidator:
    def __init__(self, cfg: ValidatorConfig):
        self.cfg = cfg

    # ---------- eps_in (Option 2) ----------
    def compute_eps_in_margin(self, runs: List[Dict[str, Any]], f_star: float) -> float:
        """
        eps_in := max{ margin, median_r(e_final^(r)) + margin }
        where e_final^(r) = final best - f_star.
        """
        errs: List[float] = []
        for run in runs:
            curve = run.get("curve", None)
            if isinstance(curve, (list, tuple)) and len(curve) > 0:
                e = float(curve[-1]) - f_star
                errs.append(max(0.0, e))
                continue

            hist = run.get("history", {})
            fitness_hist = hist.get("fitness", [])
            if fitness_hist:
                last = np.asarray(fitness_hist[-1], dtype=float)
                if last.size:
                    e = float(np.min(last)) - f_star
                    errs.append(max(0.0, e))

        if not errs:
            return float(self.cfg.margin)

        med = float(np.median(np.asarray(errs, dtype=float)))
        return float(max(self.cfg.margin, med + self.cfg.margin))

    # ---------- witness index ----------
    def select_witness_index(self, f_t: np.ndarray, f_star: float, eps_deep: float) -> Optional[int]:
        """
        Return witness index i:
          i = argmin{ f_t[j] : f_t[j] <= f* + eps_deep }
        """
        thr_deep = f_star + eps_deep
        deep_idx = np.where(f_t <= thr_deep)[0]
        if deep_idx.size == 0:
            return None
        i = int(deep_idx[np.argmin(f_t[deep_idx])])
        return i

    # ---------- eps_out per run (optional outer zone) ----------
    def eps_out_at_tau(self, f_t: np.ndarray, f_star: float, eps_in: float) -> float:
        errs = np.maximum(0.0, f_t - f_star)
        return float(max(eps_in, np.quantile(errs, self.cfg.q_out)))

    # ---------- local L estimator (isotropic, stable) ----------
    def estimate_L_local(self, x0: np.ndarray, f_fun) -> Tuple[float, Dict[str, float]]:
        """
        Stable local curvature proxy (one-sided):
          Q_k = (f(x0 + h u_k) - f(x0)) / h^2,  ||u_k||=1
          L_hat = 2 * quantile_q({Q_k})

        - avoids cancellation of symmetric second differences
        - clips step so x0 + h u remains inside [lower, upper]^d
        """
        lo, hi = float(self.cfg.lower), float(self.cfg.upper)
        d = int(x0.size)
        width = float(hi - lo)

        base_h = float(self.cfg.L_h * width)
        if not np.isfinite(base_h) or base_h <= 0:
            base_h = 1e-3

        rng = np.random.default_rng(12345)
        f0 = float(f_fun(x0))

        Qs: List[float] = []
        used_hs: List[float] = []
        invalid = 0
        nonpos = 0

        for _ in range(int(self.cfg.L_n_dirs)):
            u = rng.normal(size=d)
            nu = float(np.linalg.norm(u))
            if nu == 0.0:
                invalid += 1
                continue
            u /= nu

            # clip h so x0 + h u stays inside box
            h = base_h
            for j in range(d):
                uj = float(u[j])
                if uj > 0:
                    h = min(h, (hi - float(x0[j])) / uj)
                elif uj < 0:
                    h = min(h, (float(x0[j]) - lo) / (-uj))
            if not np.isfinite(h) or h <= 0:
                invalid += 1
                continue

            fp = float(f_fun(x0 + h * u))
            Qk = (fp - f0) / (h * h)  # (fp-f0)/||delta||^2 since ||u||=1

            if np.isfinite(Qk) and Qk > 0:
                Qs.append(float(Qk))
                used_hs.append(float(h))
            else:
                nonpos += 1

        if not Qs:
            return float("nan"), {
                "n_pos": 0.0,
                "n_total": float(self.cfg.L_n_dirs),
                "n_invalid": float(invalid),
                "n_nonpos": float(nonpos),
                "h_med": np.nan,
                "Q_med": np.nan,
                "L_q": float(self.cfg.L_quantile),
            }

        arr = np.asarray(Qs, dtype=float)
        q = float(self.cfg.L_quantile)
        if q >= 1.0:
            Q_hat = float(np.max(arr))
        else:
            Q_hat = float(np.quantile(arr, q))

        L_hat = float(2.0 * Q_hat)

        diag = {
            "n_pos": float(arr.size),
            "n_total": float(self.cfg.L_n_dirs),
            "n_invalid": float(invalid),
            "n_nonpos": float(nonpos),
            "h_med": float(np.median(np.asarray(used_hs, dtype=float))) if used_hs else np.nan,
            "Q_med": float(np.median(arr)),
            "L_q": float(q),
        }
        return L_hat, diag

    # ---------- scan forward for tau_C2 and tau_pair ----------
    def scan_tau_C2(
        self,
        fitness_hist: List[Any],
        positions_hist: List[Any],
        start_t: int,
        f_star: float,
        eps_deep: float,
        r_conc: float,
    ) -> Dict[str, Any]:
        """
        Scan t = start_t..T for first time where:
          - deep hitters exist at t
          - min distance from witness to others <= r_conc

        Also track tau_pair:
          first time the 2nd-nearest neighbor distance <= r_conc
          (equivalently c1 >= 2 in the population excluding the witness)

        Returns a dict with:
          tau_C2, tau_pair (Optional[int])
          min_dist_at_tau_deep
          min_dist_at_tau_C2
          second_dist_at_tau_pair
          beta_at_tau_C2, beta_at_tau_pair
          c1_at_tau_C2, c1_at_tau_pair
        """
        T = min(len(fitness_hist), len(positions_hist))
        if start_t >= T:
            return {
                "tau_C2": None,
                "tau_pair": None,
                "min_dist_at_tau_deep": np.nan,
                "min_dist_at_tau_C2": np.nan,
                "second_dist_at_tau_pair": np.nan,
                "beta_at_tau_C2": np.nan,
                "beta_at_tau_pair": np.nan,
                "c1_at_tau_C2": np.nan,
                "c1_at_tau_pair": np.nan,
            }

        # compute min_dist at tau_deep
        f0 = np.asarray(fitness_hist[start_t], dtype=float)
        X0 = np.asarray(positions_hist[start_t], dtype=float)
        w0 = self.select_witness_index(f0, f_star, eps_deep)
        if w0 is None or X0.ndim != 2 or X0.shape[0] != f0.size:
            min_dist_tau_deep = np.nan
        else:
            mask0 = np.ones(f0.size, dtype=bool)
            mask0[w0] = False
            if np.any(mask0):
                d0 = np.linalg.norm(X0[mask0] - X0[w0], axis=1)
                min_dist_tau_deep = float(np.min(d0)) if d0.size else np.nan
            else:
                min_dist_tau_deep = np.nan

        tau_C2: Optional[int] = None
        tau_pair: Optional[int] = None
        min_dist_at_tau_C2 = np.nan
        second_dist_at_tau_pair = np.nan
        beta_at_tau_C2 = np.nan
        beta_at_tau_pair = np.nan
        c1_at_tau_C2 = np.nan
        c1_at_tau_pair = np.nan

        # scan for tau_C2 and tau_pair
        for t in range(start_t, T):
            f_t = np.asarray(fitness_hist[t], dtype=float)
            X_t = np.asarray(positions_hist[t], dtype=float)
            if X_t.ndim != 2 or X_t.shape[0] != f_t.size or f_t.size < 2:
                continue

            w = self.select_witness_index(f_t, f_star, eps_deep)
            if w is None:
                continue

            mask = np.ones(f_t.size, dtype=bool)
            mask[w] = False
            if not np.any(mask):
                continue

            d = np.linalg.norm(X_t[mask] - X_t[w], axis=1)
            if d.size == 0:
                continue

            d_sorted = np.sort(d)
            d1 = float(d_sorted[0])
            d2 = float(d_sorted[1]) if d_sorted.size >= 2 else float("inf")

            c1 = int(np.sum(d <= r_conc))
            s1 = int(f_t.size - 1)
            beta = float(c1 / s1) if s1 > 0 else np.nan

            # First neighbor inside r_conc => tau_C2
            if tau_C2 is None and d1 <= r_conc:
                tau_C2 = int(t)
                min_dist_at_tau_C2 = float(d1)
                beta_at_tau_C2 = float(beta)
                c1_at_tau_C2 = float(c1)

            # Second neighbor inside r_conc => tau_pair (>=2 donors)
            if tau_pair is None and d2 <= r_conc:
                tau_pair = int(t)
                second_dist_at_tau_pair = float(d2)
                beta_at_tau_pair = float(beta)
                c1_at_tau_pair = float(c1)

            if (tau_C2 is not None) and (tau_pair is not None):
                break

        return {
            "tau_C2": tau_C2,
            "tau_pair": tau_pair,
            "min_dist_at_tau_deep": float(min_dist_tau_deep),
            "min_dist_at_tau_C2": float(min_dist_at_tau_C2),
            "second_dist_at_tau_pair": float(second_dist_at_tau_pair),
            "beta_at_tau_C2": float(beta_at_tau_C2),
            "beta_at_tau_pair": float(beta_at_tau_pair),
            "c1_at_tau_C2": float(c1_at_tau_C2),
            "c1_at_tau_pair": float(c1_at_tau_pair),
        }

    # ---------- per-run analysis ----------
    def analyze_run(self, run: Dict[str, Any], fid: int, f_fun) -> Optional[Dict[str, Any]]:
        hist = run.get("history", {})
        fitness_hist = hist.get("fitness", [])
        positions_hist = hist.get("positions", [])
        if not fitness_hist or not positions_hist:
            return None

        f_star = get_f_star_cec2017(fid)

        eps_in = float(run.get("_eps_in", np.nan))
        if not np.isfinite(eps_in) or eps_in <= 0:
            return None

        eps_deep = float(self.cfg.deep_ratio * eps_in)
        tau_deep = first_hit_time(fitness_hist, f_star + eps_deep)
        if tau_deep is None:
            return None

        f_t = np.asarray(fitness_hist[tau_deep], dtype=float)
        X_t = np.asarray(positions_hist[tau_deep], dtype=float)
        if f_t.size == 0 or X_t.ndim != 2 or X_t.shape[0] != f_t.size:
            return None

        N = int(f_t.size)
        if N < 4:
            return None

        witness_idx = self.select_witness_index(f_t, f_star, eps_deep)
        if witness_idx is None:
            return None

        x_witness = X_t[witness_idx].copy()

        # eps_out (only for zone plot; can be equal to eps_in sometimes)
        eps_out = self.eps_out_at_tau(f_t, f_star, eps_in)

        # local L at witness at tau_deep
        L_hat, L_diag = self.estimate_L_local(x0=x_witness, f_fun=f_fun)
        if not np.isfinite(L_hat) or L_hat <= 0:
            return None

        r_safe = float(self.cfg.r_safe_factor * np.sqrt((2.0 * eps_in) / L_hat))
        denom = 2.0 * (self.cfg.F_minus + self.cfg.Delta_F)
        r_conc = float(r_safe / denom) if denom > 0 else float("inf")

        # zones at tau_deep
        thr_deep = f_star + eps_deep
        thr_in = f_star + eps_in
        thr_out = f_star + eps_out

        n_deep = int(np.sum(f_t <= thr_deep))
        n_in = int(np.sum(f_t <= thr_in))
        n_out = int(np.sum(f_t <= thr_out))
        n_in_shell = int(n_in - n_deep)
        n_out_shell = int(n_out - n_in)
        n_outside = int(N - n_out)

        # distances at tau_deep from witness to others
        mask = np.ones(N, dtype=bool)
        mask[witness_idx] = False
        X_others = X_t[mask]
        dS1 = np.linalg.norm(X_others - x_witness, axis=1) if X_others.size else np.asarray([], dtype=float)

        distS1_min = float(np.min(dS1)) if dS1.size else np.nan
        distS1_q10 = float(np.quantile(dS1, 0.10)) if dS1.size else np.nan
        distS1_med = float(np.median(dS1)) if dS1.size else np.nan

        # beta at tau_deep (population only)
        s1 = int(N - 1)
        c1 = int(np.sum(dS1 <= r_conc)) if dS1.size else 0
        beta1_hat = float(c1 / s1) if s1 > 0 else np.nan

        # optional archive-based beta2/c_pair at tau_deep (kept for continuity)
        X_arch = extract_archive_positions(hist, tau_deep)
        arch_M = extract_archive_size(hist, tau_deep, X_arch)
        distA_min = distA_q10 = distA_med = np.nan

        s2 = float(s1 + arch_M)
        c2 = float(c1)
        if X_arch is not None and X_arch.ndim == 2 and X_arch.shape[0] > 0:
            dA = np.linalg.norm(X_arch - x_witness, axis=1)
            if dA.size:
                distA_min = float(np.min(dA))
                distA_q10 = float(np.quantile(dA, 0.10))
                distA_med = float(np.median(dA))
            c2 += float(np.sum(dA <= r_conc))

        beta2_hat = float(c2 / s2) if s2 > 0 else np.nan
        if c2 > 1 and s2 > 1 and s1 > 0:
            c_pair_lb_hat = float((c1 / s1) * ((c2 - 1.0) / (s2 - 1.0)))
        else:
            c_pair_lb_hat = 0.0

        # helper: compute archive-augmented c_pair bound at time t (using witness-at-t)
        def cpair_at_time(t: Optional[int]) -> Tuple[float, float, float, float, float]:
            """
            Returns (c_pair_lb, c1_t, c2_t, s1_t, s2_t). NaNs if t is None/invalid.
            """
            if t is None:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            if t < 0 or t >= min(len(fitness_hist), len(positions_hist)):
                return (np.nan, np.nan, np.nan, np.nan, np.nan)

            f_tt = np.asarray(fitness_hist[t], dtype=float)
            X_tt = np.asarray(positions_hist[t], dtype=float)
            if X_tt.ndim != 2 or X_tt.shape[0] != f_tt.size or f_tt.size < 2:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)

            w = self.select_witness_index(f_tt, f_star, eps_deep)
            if w is None:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            xw = X_tt[w]

            mask_tt = np.ones(f_tt.size, dtype=bool)
            mask_tt[w] = False
            dpop = np.linalg.norm(X_tt[mask_tt] - xw, axis=1) if np.any(mask_tt) else np.asarray([], dtype=float)

            s1_t = float(f_tt.size - 1)
            c1_t = float(np.sum(dpop <= r_conc)) if dpop.size else 0.0

            X_arch_t = extract_archive_positions(hist, t)
            arch_M_t = extract_archive_size(hist, t, X_arch_t)
            s2_t = float(s1_t + arch_M_t)

            c2_t = float(c1_t)
            if X_arch_t is not None and X_arch_t.ndim == 2 and X_arch_t.shape[0] > 0:
                dA_t = np.linalg.norm(X_arch_t - xw, axis=1)
                c2_t += float(np.sum(dA_t <= r_conc))

            if s1_t > 0 and s2_t > 1 and c2_t > 1:
                c_pair_t = float((c1_t / s1_t) * ((c2_t - 1.0) / (s2_t - 1.0)))
            else:
                c_pair_t = 0.0
            return (c_pair_t, c1_t, c2_t, s1_t, s2_t)

        # scan forward for tau_C2 and tau_pair
        scan = self.scan_tau_C2(
            fitness_hist=fitness_hist,
            positions_hist=positions_hist,
            start_t=int(tau_deep),
            f_star=float(f_star),
            eps_deep=float(eps_deep),
            r_conc=float(r_conc),
        )
        tau_C2 = scan["tau_C2"]
        tau_pair = scan["tau_pair"]

        C2_achieved = 1 if tau_C2 is not None else 0
        pair_achieved = 1 if tau_pair is not None else 0
        gap_C2 = (float(tau_C2) - float(tau_deep)) if tau_C2 is not None else np.nan
        gap_pair = (float(tau_pair) - float(tau_deep)) if tau_pair is not None else np.nan

        cpair_tau_C2, c1_tau_C2, c2_tau_C2, s1_tau_C2, s2_tau_C2 = cpair_at_time(tau_C2)
        cpair_tau_pair, c1_tau_pair, c2_tau_pair, s1_tau_pair, s2_tau_pair = cpair_at_time(tau_pair)

        return {
            "func": f"f{fid}",
            "fid": int(fid),
            "run_id": int(run.get("run_id", -1)),

            # (C1) time
            "tau_deep": int(tau_deep),

            # (C2) time
            "tau_C2": float(tau_C2) if tau_C2 is not None else np.nan,
            "gap_C2": float(gap_C2),
            "C2_achieved": int(C2_achieved),

            # pair-ready time (>=2 donors)
            "tau_pair": float(tau_pair) if tau_pair is not None else np.nan,
            "gap_pair": float(gap_pair),
            "pair_achieved": int(pair_achieved),

            # eps and radii
            "eps_in": float(eps_in),
            "eps_deep": float(eps_deep),
            "eps_out": float(eps_out),
            "L_hat": float(L_hat),
            "r_safe": float(r_safe),
            "r_conc": float(r_conc),

            # witness index at tau_deep
            "witness_idx_tau_deep": int(witness_idx),

            # zone masses at tau_deep
            "N": int(N),
            "n_deep": int(n_deep),
            "n_in_shell": int(n_in_shell),
            "n_out_shell": int(n_out_shell),
            "n_outside": int(n_outside),

            # concentration at tau_deep (population)
            "s1": float(s1),
            "c1": float(c1),
            "beta1_hat": float(beta1_hat),

            # archive (optional) at tau_deep
            "arch_M": int(arch_M),
            "s2": float(s2),
            "c2": float(c2),
            "beta2_hat": float(beta2_hat),
            "c_pair_lb_hat": float(c_pair_lb_hat),

            # distance diagnostics at tau_deep (population & archive)
            "distS1_min": float(distS1_min),
            "distS1_q10": float(distS1_q10),
            "distS1_med": float(distS1_med),
            "distA_min": float(distA_min),
            "distA_q10": float(distA_q10),
            "distA_med": float(distA_med),

            # scan diagnostics
            "min_dist_at_tau_deep": float(scan["min_dist_at_tau_deep"]),
            "min_dist_at_tau_C2": float(scan["min_dist_at_tau_C2"]),
            "second_dist_at_tau_pair": float(scan["second_dist_at_tau_pair"]),
            "beta_at_tau_C2": float(scan["beta_at_tau_C2"]),
            "beta_at_tau_pair": float(scan["beta_at_tau_pair"]),
            "c1_at_tau_C2": float(scan["c1_at_tau_C2"]),
            "c1_at_tau_pair": float(scan["c1_at_tau_pair"]),

            # c_pair lower bounds at scan times (archive-augmented)
            "c_pair_lb_at_tau_C2": float(cpair_tau_C2),
            "c_pair_lb_at_tau_pair": float(cpair_tau_pair),
            "c1_tau_C2": float(c1_tau_C2),
            "c2_tau_C2": float(c2_tau_C2),
            "s1_tau_C2": float(s1_tau_C2),
            "s2_tau_C2": float(s2_tau_C2),
            "c1_tau_pair": float(c1_tau_pair),
            "c2_tau_pair": float(c2_tau_pair),
            "s1_tau_pair": float(s1_tau_pair),
            "s2_tau_pair": float(s2_tau_pair),

            # L diagnostics
            "L_n_pos": float(L_diag.get("n_pos", np.nan)),
            "L_n_total": float(L_diag.get("n_total", np.nan)),
            "L_n_invalid": float(L_diag.get("n_invalid", np.nan)),
            "L_n_nonpos": float(L_diag.get("n_nonpos", np.nan)),
            "L_h_med": float(L_diag.get("h_med", np.nan)),
            "Q_med": float(L_diag.get("Q_med", np.nan)),
            "L_quantile": float(L_diag.get("L_q", np.nan)),
        }


# =============================================================================
# IO / plots
# =============================================================================

class InputOutputPipeline:
    def __init__(self, outdir: Path, verbose: bool = True):
        self.outdir = outdir
        self.verbose = verbose

    def write_csvs(self, per_run_df: pd.DataFrame, dim: int, func_tag: str) -> Tuple[Path, Path]:
        self.outdir.mkdir(parents=True, exist_ok=True)
        per_run_path = self.outdir / f"morse_validate_D{dim}_{func_tag}.per_run.csv"
        summary_path = self.outdir / f"morse_validate_D{dim}_{func_tag}.summary.csv"
        per_run_df.to_csv(per_run_path, index=False)

        summary_df = self.make_summary(per_run_df)
        summary_df.to_csv(summary_path, index=False)

        if self.verbose:
            print(ts(f"Saved: {per_run_path}"))
            print(ts(f"Saved: {summary_path}"))
        return per_run_path, summary_path

    @staticmethod
    def make_summary(per_run_df: pd.DataFrame) -> pd.DataFrame:
        def q(x, p):
            return x.quantile(p) if len(x) else np.nan

        g = per_run_df.groupby("func", dropna=False)

        summary = g.agg(
            n_runs_ok=("tau_deep", "count"),
            N_med=("N", "median"),

            tau_deep_med=("tau_deep", "median"),

            # tau_C2 stats
            C2_achieved_frac=("C2_achieved", "mean"),
            tau_C2_med=("tau_C2", "median"),
            gap_C2_med=("gap_C2", "median"),

            # tau_pair stats (>=2 donors)
            pair_achieved_frac=("pair_achieved", "mean"),
            tau_pair_med=("tau_pair", "median"),
            gap_pair_med=("gap_pair", "median"),

            eps_in=("eps_in", "median"),
            eps_out_med=("eps_out", "median"),
            L_hat_med=("L_hat", "median"),
            L_hat_q80=("L_hat", lambda x: q(x, 0.80)),
            r_safe_med=("r_safe", "median"),
            r_conc_med=("r_conc", "median"),

            beta1_med=("beta1_hat", "median"),
            beta2_med=("beta2_hat", "median"),
            c_pair_lb_med=("c_pair_lb_hat", "median"),

            # scan diagnostics
            min_dist_tau_deep_med=("min_dist_at_tau_deep", "median"),
            min_dist_tau_C2_med=("min_dist_at_tau_C2", "median"),
            beta_at_tau_C2_med=("beta_at_tau_C2", "median"),
            second_dist_tau_pair_med=("second_dist_at_tau_pair", "median"),
            beta_at_tau_pair_med=("beta_at_tau_pair", "median"),

            # c_pair at scan times
            c_pair_lb_at_tau_C2_med=("c_pair_lb_at_tau_C2", "median"),
            c_pair_lb_at_tau_pair_med=("c_pair_lb_at_tau_pair", "median"),

            # zones
            n_deep_med=("n_deep", "median"),
            n_in_shell_med=("n_in_shell", "median"),
            n_out_shell_med=("n_out_shell", "median"),
            n_outside_med=("n_outside", "median"),

            # distance diagnostics
            distS1_min_med=("distS1_min", "median"),
            distS1_q10_med=("distS1_q10", "median"),
            distS1_med_med=("distS1_med", "median"),

            # L diagnostics
            L_n_pos_med=("L_n_pos", "median"),
            L_h_med=("L_h_med", "median"),
            Q_med_med=("Q_med", "median"),
        ).reset_index()

        return summary

    def plot_quick(self, summary_df: pd.DataFrame, dim: int, func_tag: str) -> None:
        out1 = self.outdir / f"morse_validate_D{dim}_{func_tag}.cpair.png"
        out2 = self.outdir / f"morse_validate_D{dim}_{func_tag}.zones.png"

        df = summary_df.copy().sort_values("c_pair_lb_med", ascending=True)

        # plot 1: c_pair median at tau_deep
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df) + 1)))
        ax.barh(df["func"], df["c_pair_lb_med"])
        ax.set_xlabel(r"median $\widehat{c}_{\mathrm{pair}}$ lower bound at $\tau_{\mathrm{deep}}$")
        ax.set_title(f"CEC2017 witness-regime strength (D={dim})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out1, dpi=200, bbox_inches="tight")
        plt.close(fig)

        if self.verbose:
            print(ts(f"Saved: {out1}"))

        # plot 1b: c_pair at tau_pair (>=2 donors)
        if "c_pair_lb_at_tau_pair_med" in df.columns:
            out1b = self.outdir / f"morse_validate_D{dim}_{func_tag}.cpair_taupair.png"
            df_sorted = df.sort_values("c_pair_lb_at_tau_pair_med", ascending=True)
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df_sorted) + 1)))
            ax.barh(df_sorted["func"], df_sorted["c_pair_lb_at_tau_pair_med"])
            ax.set_xlabel(r"median $\widehat{c}_{\mathrm{pair}}$ lower bound at $\tau_{\mathrm{pair}}$")
            ax.set_title(f"CEC2017 pair-ready witness strength (D={dim})")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(out1b, dpi=200, bbox_inches="tight")
            plt.close(fig)
            if self.verbose:
                print(ts(f"Saved: {out1b}"))

        # plot 2: zones stacked fractions
        N = df["N_med"].astype(float).replace(0, 1.0).values
        deep = df["n_deep_med"].astype(float).values / N
        in_shell = df["n_in_shell_med"].astype(float).values / N
        out_shell = df["n_out_shell_med"].astype(float).values / N
        outside = df["n_outside_med"].astype(float).values / N

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df) + 1)))
        left = np.zeros_like(deep)
        ax.barh(df["func"], deep, left=left, label=r"$A_{\mathrm{deep}}$")
        left += deep
        ax.barh(df["func"], in_shell, left=left, label=r"$A_{\mathrm{in}}\setminus A_{\mathrm{deep}}$")
        left += in_shell
        ax.barh(df["func"], out_shell, left=left, label=r"$A_{\mathrm{out}}\setminus A_{\mathrm{in}}$")
        left += out_shell
        ax.barh(df["func"], outside, left=left, label=r"outside $A_{\mathrm{out}}$")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(r"fraction of population at $\tau_{\mathrm{deep}}$")
        ax.set_title(f"Zone mass at τ_deep (D={dim})")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close(fig)

        if self.verbose:
            print(ts(f"Saved: {out2}"))


# =============================================================================
# Main
# =============================================================================

def parse_func_list(s: str) -> Tuple[List[int], str]:
    s0 = s.strip().lower()
    if s0 == "all":
        return list(range(1, 31)), "all"
    parts = [p for p in s0.replace(" ", "").split(",") if p]
    fids = [int(p.replace("f", "")) for p in parts]
    tag = "_".join([f"f{fid}" for fid in fids])
    return fids, tag


def main() -> None:
    ap = argparse.ArgumentParser(description="Morse / witness regime validation (Fix1 + Fix2 + Fix3).")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--func", type=str, required=True, help="all or comma list: f1,f2,...")
    ap.add_argument("--base", type=str, default="experiments")
    ap.add_argument("--outdir", type=str, default="out_morse_validate")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--plots", action="store_true")

    # eps_in Option 2
    ap.add_argument("--margin", type=float, default=60.0)
    ap.add_argument("--deep_ratio", type=float, default=0.25)
    ap.add_argument("--q_out", type=float, default=0.90)

    # r_conc
    ap.add_argument("--F_minus", type=float, default=0.1)
    ap.add_argument("--Delta_F", type=float, default=0.8)
    ap.add_argument("--r_safe_factor", type=float, default=float(1.0 - 1.0 / np.sqrt(2.0)))

    # local L parameters
    ap.add_argument("--L_n_dirs", type=int, default=64)
    ap.add_argument("--L_h", type=float, default=1e-3)
    ap.add_argument("--L_quantile", type=float, default=0.90)

    # bounds
    ap.add_argument("--lower", type=float, default=-100.0)
    ap.add_argument("--upper", type=float, default=100.0)

    args = ap.parse_args()
    verbose = not args.quiet

    fids, func_tag = parse_func_list(args.func)
    base = Path(args.base)
    outdir = Path(args.outdir)

    cfg = ValidatorConfig(
        margin=args.margin,
        deep_ratio=args.deep_ratio,
        q_out=args.q_out,
        F_minus=args.F_minus,
        Delta_F=args.Delta_F,
        r_safe_factor=args.r_safe_factor,
        L_n_dirs=args.L_n_dirs,
        L_h=args.L_h,
        L_quantile=args.L_quantile,
        lower=args.lower,
        upper=args.upper,
    )

    validator = MathValidator(cfg)
    io = InputOutputPipeline(outdir=outdir, verbose=verbose)

    per_run_rows: List[Dict[str, Any]] = []

    for k, fid in enumerate(fids, start=1):
        runs = load_runs(args.dim, fid, base)
        if not runs:
            if verbose:
                print(ts(f"[{k}/{len(fids)}] f{fid}: not found"))
            continue

        f_star = get_f_star_cec2017(fid)
        eps_in = validator.compute_eps_in_margin(runs, f_star)

        # Attach eps_in to runs
        for ridx, run in enumerate(runs):
            run["_eps_in"] = eps_in
            if "run_id" not in run:
                run["run_id"] = ridx

        if verbose:
            print(ts(f"[{k}/{len(fids)}] f{fid}: runs={len(runs)} eps_in={eps_in:.6g}"))

        f_fun = CEC_FUNCS[fid]

        for run in runs:
            row = validator.analyze_run(run, fid=fid, f_fun=f_fun)
            if row is not None:
                per_run_rows.append(row)

    if not per_run_rows:
        print("No valid runs (no τ_deep hits or missing history keys).")
        return

    per_run_df = pd.DataFrame(per_run_rows)
    per_run_path, summary_path = io.write_csvs(per_run_df, dim=args.dim, func_tag=func_tag)

    summary_df = pd.read_csv(summary_path)
    if verbose:
        print("\nSummary (first rows):")
        print(summary_df.head(12).to_string(index=False))

    if args.plots:
        io.plot_quick(summary_df, dim=args.dim, func_tag=func_tag)


if __name__ == "__main__":
    main()
