#!/usr/bin/env python3
"""gamma0_validate.py

Validate / audit Proposition \ref{prop:gamma0-combinatorial} (paper3d.tex)
using already computed run-level results.

This script *does not* simulate the probability of \mathcal{L}_t. It only
computes the analytical lower bound \gamma_0(t) from Eq. (\ref{eq:gamma0-formula}).

Eq. (\ref{eq:gamma0-formula})
----------------------------
On G_{t-1}:
  P(\mathcal{L}_t | \mathcal{F}_{t-1}) >= \gamma_0(t-1), where

  \gamma_0(t-1) = (1/H) * g^- * (F^+ - F^-) * q^- 
                 * 1/ceil(p N_{t-1})
                 * (m-2)/(N_{t-1}-2)
                 * (m-3)/(N_{t-1}+A_{t-1}-3).

In practice, we evaluate this bound at specific timestamps you already computed
(e.g., tau_deep, tau_C2, tau_pair) using the corresponding N_t, A_t extracted
from your per-run CSV. Interpreting \gamma_0(t) as the bound for generation t+1
is consistent with the proposition (shift by one generation).

Inputs
------
A per-run CSV produced by your validation pipeline, ideally the Fix123 version of
morse_validate.py. The script will try to locate the necessary columns, and will
fallback gracefully if some columns are missing.

Outputs
-------
Two CSVs are written:
  - gamma0_validate.<tag>.per_run.csv (input rows + gamma0 columns)
  - gamma0_validate.<tag>.summary.csv (function-level aggregates)

Optionally, plots can be produced (bar chart of median log10 gamma0).

Limitations
-----------
We cannot *verify* the full witness-stable regime G_t from this CSV alone
(e.g., basin membership and "good memory" are not logged). This script audits
and reports the *numerical/combinatorial* side of Proposition \ref{prop:gamma0-combinatorial}.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Config + math
# =============================================================================

@dataclass
class Gamma0Config:
    # L-SHADE / paper parameters
    p: float
    H: int
    m_req: int
    F_minus: float
    F_plus: float
    g_minus: float
    q_minus: float

    # Optional time-uniform bound inputs
    N_max: Optional[int] = None
    A_max: Optional[int] = None

    def validate(self) -> None:
        if not (0.0 < self.p <= 1.0):
            raise ValueError(f"p must be in (0,1], got {self.p}")
        if self.H <= 0:
            raise ValueError(f"H must be positive, got {self.H}")
        if self.m_req < 4:
            raise ValueError(
                f"m_req must be >= 4 (need 2 donors besides b,i), got {self.m_req}"
            )
        if not (0.0 < self.F_minus < self.F_plus <= 1.0):
            raise ValueError(
                f"Require 0 < F_minus < F_plus <= 1, got {self.F_minus}, {self.F_plus}"
            )
        if self.g_minus <= 0.0:
            raise ValueError(f"g_minus must be >0, got {self.g_minus}")
        if self.q_minus <= 0.0:
            raise ValueError(f"q_minus must be >0, got {self.q_minus}")

    @property
    def const_factor(self) -> float:
        """(1/H) * g^- * (F^+ - F^-) * q^-"""
        return (1.0 / float(self.H)) * float(self.g_minus) * float(self.F_plus - self.F_minus) * float(self.q_minus)


def _first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_float_array(x: pd.Series) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").astype(float).to_numpy()


def gamma0_comb(N: np.ndarray, A: np.ndarray, m: np.ndarray, p: float) -> np.ndarray:
    """Combinatorial part (no memory/F/CR constants):

        1/ceil(pN) * (m-2)/(N-2) * (m-3)/(N+A-3)

    We clamp negative values to 0 (e.g. m<3 gives no 2-donor event).
    """

    N = np.asarray(N, dtype=float)
    A = np.asarray(A, dtype=float)
    m = np.asarray(m, dtype=float)

    out = np.full_like(N, np.nan, dtype=float)

    valid = (
        np.isfinite(N)
        & np.isfinite(A)
        & np.isfinite(m)
        & (N > 2.0)
        & ((N + A) > 3.0)
        & (p > 0.0)
        & (p <= 1.0)
    )

    if not np.any(valid):
        return out

    Nv = N[valid]
    Av = A[valid]
    mv = m[valid]

    denom_pbest = np.maximum(np.ceil(p * Nv), 1.0)
    term_pbest = 1.0 / denom_pbest

    term_r1 = (mv - 2.0) / (Nv - 2.0)
    term_r2 = (mv - 3.0) / (Nv + Av - 3.0)

    comb = term_pbest * term_r1 * term_r2
    comb = np.maximum(comb, 0.0)

    out[valid] = comb
    return out


def gamma0_full(N: np.ndarray, A: np.ndarray, m: np.ndarray, cfg: Gamma0Config) -> np.ndarray:
    return cfg.const_factor * gamma0_comb(N=N, A=A, m=m, p=cfg.p)


# =============================================================================
# Extraction from morse_validate per-run CSV
# =============================================================================

@dataclass
class TimeSlice:
    tag: str  # e.g. "tau_deep", "tau_C2", "tau_pair"
    N: np.ndarray
    A: np.ndarray
    c1: np.ndarray


def extract_timeslice(df: pd.DataFrame, tag: str) -> TimeSlice:
    """Extract (N_t, A_t, c1_t) for a given tag from a morse_validate per_run.csv.

    Supported tags:
      - tau_deep: uses N, arch_M, c1
      - tau_C2:   uses s1_tau_C2/s2_tau_C2 and c1_at_tau_C2 or c1_tau_C2
      - tau_pair: uses s1_tau_pair/s2_tau_pair and c1_at_tau_pair or c1_tau_pair

    If columns are missing, returns NaNs.
    """
    n = len(df)
    nan = np.full(n, np.nan, dtype=float)

    if tag == "tau_deep":
        N_col = _first_existing(df, ["N", "N_tau_deep"])
        if N_col is not None:
            N = _to_float_array(df[N_col])
        else:
            s1_col = _first_existing(df, ["s1"])
            N = _to_float_array(df[s1_col]) + 1.0 if s1_col is not None else nan

        A_col = _first_existing(df, ["arch_M", "A_tau_deep"])
        if A_col is not None:
            A = _to_float_array(df[A_col])
        else:
            s1_col = _first_existing(df, ["s1"])
            s2_col = _first_existing(df, ["s2"])
            if s1_col is not None and s2_col is not None:
                A = _to_float_array(df[s2_col]) - _to_float_array(df[s1_col])
            else:
                A = nan

        c1_col = _first_existing(df, ["c1", "c1_tau_deep"])
        c1 = _to_float_array(df[c1_col]) if c1_col is not None else nan

        return TimeSlice(tag=tag, N=N, A=A, c1=c1)

    if tag == "tau_C2":
        s1_col = _first_existing(df, ["s1_tau_C2"])
        s2_col = _first_existing(df, ["s2_tau_C2"])
        if s1_col is not None and s2_col is not None:
            s1 = _to_float_array(df[s1_col])
            s2 = _to_float_array(df[s2_col])
            N = s1 + 1.0
            A = s2 - s1
        else:
            N = nan
            A = nan

        c1_col = _first_existing(df, ["c1_at_tau_C2", "c1_tau_C2"])
        c1 = _to_float_array(df[c1_col]) if c1_col is not None else nan

        return TimeSlice(tag=tag, N=N, A=A, c1=c1)

    if tag == "tau_pair":
        s1_col = _first_existing(df, ["s1_tau_pair"])
        s2_col = _first_existing(df, ["s2_tau_pair"])
        if s1_col is not None and s2_col is not None:
            s1 = _to_float_array(df[s1_col])
            s2 = _to_float_array(df[s2_col])
            N = s1 + 1.0
            A = s2 - s1
        else:
            N = nan
            A = nan

        c1_col = _first_existing(df, ["c1_at_tau_pair", "c1_tau_pair"])
        c1 = _to_float_array(df[c1_col]) if c1_col is not None else nan

        return TimeSlice(tag=tag, N=N, A=A, c1=c1)

    raise ValueError(f"Unsupported tag: {tag}")


# =============================================================================
# Summary + plots
# =============================================================================

class Gamma0IO:
    def __init__(self, outdir: Path, verbose: bool = True):
        self.outdir = outdir
        self.verbose = verbose

    def write(self, per_run: pd.DataFrame, tag: str) -> Tuple[Path, Path]:
        self.outdir.mkdir(parents=True, exist_ok=True)
        per_path = self.outdir / f"gamma0_validate.{tag}.per_run.csv"
        sum_path = self.outdir / f"gamma0_validate.{tag}.summary.csv"

        per_run.to_csv(per_path, index=False)
        summary = self.make_summary(per_run)
        summary.to_csv(sum_path, index=False)

        if self.verbose:
            print(f"Saved: {per_path}")
            print(f"Saved: {sum_path}")
        return per_path, sum_path

    @staticmethod
    def make_summary(df: pd.DataFrame) -> pd.DataFrame:
        def q(x: pd.Series, p: float) -> float:
            x = x.dropna()
            return float(x.quantile(p)) if len(x) else float("nan")

        g = df.groupby("func", dropna=False)

        # Keep columns small but informative.
        out = g.agg(
            n_runs=("run_id", "count"),

            # G2 proxy: cluster size >= m_req
            G2_ok_frac=("G2_ok", "mean"),
            cluster_M_med=("cluster_M", "median"),

            # gamma0 (theorem, only when G2_ok)
            gamma0_med=("gamma0_full_valid", "median"),
            gamma0_q10=("gamma0_full_valid", lambda x: q(x, 0.10)),
            gamma0_q90=("gamma0_full_valid", lambda x: q(x, 0.90)),

            # gamma0_hat (empirical m_hat)
            gamma0_hat_med=("gamma0_hat_full", "median"),
            gamma0_hat_q10=("gamma0_hat_full", lambda x: q(x, 0.10)),
            gamma0_hat_q90=("gamma0_hat_full", lambda x: q(x, 0.90)),

            # sizes
            N_med=("N_t", "median"),
            A_med=("A_t", "median"),
        ).reset_index()

        return out

    def plot_bar(self, summary: pd.DataFrame, tag: str) -> None:
        """Plot median gamma0 (log10) across functions."""
        outpath = self.outdir / f"gamma0_validate.{tag}.gamma0_bar.png"

        df = summary.copy().sort_values("gamma0_hat_med", ascending=True)
        vals = df["gamma0_hat_med"].to_numpy(dtype=float)
        # avoid -inf
        vals = np.where(vals > 0, vals, np.nan)
        logv = np.log10(vals)

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df) + 1)))
        ax.barh(df["func"], logv)
        ax.set_xlabel(r"median $\log_{10}(\hat\gamma_0)$")
        ax.set_title(f"gamma0 empirical bound at {tag}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)

        if self.verbose:
            print(f"Saved: {outpath}")


# =============================================================================
# Main
# =============================================================================


def _infer_tag_from_path(path: Path) -> str:
    # For nicer filenames only
    stem = path.stem
    if "D" in stem:
        return stem
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Proposition gamma0-combinatorial from morse_validate outputs.")
    ap.add_argument("--in_per_run", type=str, required=True, help="Path to morse_validate per_run.csv (Fix123 recommended).")
    ap.add_argument("--outdir", type=str, default="out_gamma0")
    ap.add_argument("--tag", type=str, default="auto", help="Label for outputs (default: auto from input filename).")

    # Proposition parameters
    ap.add_argument("--p", type=float, required=True, help="p-best fraction p.")
    ap.add_argument("--H", type=int, required=True, help="Memory size H.")
    ap.add_argument("--m", type=int, default=4, help="Minimum cluster size m (>=4).")
    ap.add_argument("--F_minus", type=float, required=True)
    ap.add_argument("--F_plus", type=float, required=True)
    ap.add_argument("--g_minus", type=float, default=1.0, help="Lower bound g^- on Cauchy density in [F-,F+].")
    ap.add_argument("--q_minus", type=float, default=1.0, help="Lower bound q^- on P(CR>=c_cr) for a good slot.")

    # Optional time-uniform bounds
    ap.add_argument("--N_max", type=int, default=None)
    ap.add_argument("--A_max", type=int, default=None)

    # Which time slice(s) to audit
    ap.add_argument("--times", type=str, default="tau_deep,tau_C2,tau_pair", help="Comma list from {tau_deep,tau_C2,tau_pair}.")

    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    verbose = not args.quiet

    in_path = Path(args.in_per_run)
    outdir = Path(args.outdir)
    tag = args.tag if args.tag != "auto" else _infer_tag_from_path(in_path)

    cfg = Gamma0Config(
        p=args.p,
        H=args.H,
        m_req=args.m,
        F_minus=args.F_minus,
        F_plus=args.F_plus,
        g_minus=args.g_minus,
        q_minus=args.q_minus,
        N_max=args.N_max,
        A_max=args.A_max,
    )
    cfg.validate()

    df = pd.read_csv(in_path)
    if "func" not in df.columns:
        raise ValueError("Input per_run.csv must contain a 'func' column (e.g. f1,f2,...)")
    if "run_id" not in df.columns:
        # not fatal; create one
        df = df.copy()
        df["run_id"] = np.arange(len(df), dtype=int)

    times = [t.strip() for t in args.times.split(",") if t.strip()]

    # For data-based N_max/A_max if user did not provide.
    N_max_obs = -math.inf
    A_max_obs = -math.inf

    io = Gamma0IO(outdir=outdir, verbose=verbose)

    for ttag in times:
        ts = extract_timeslice(df, ttag)

        # cluster size estimate: M = 1 + c1 (since c1 excludes the best/witness)
        cluster_M = 1.0 + ts.c1
        m_hat_int = np.floor(cluster_M + 1e-9)

        # core sizes
        N_t = ts.N
        A_t = ts.A

        # Update observed maxima
        if np.any(np.isfinite(N_t)):
            N_max_obs = max(N_max_obs, float(np.nanmax(N_t)))
        if np.any(np.isfinite(A_t)):
            A_max_obs = max(A_max_obs, float(np.nanmax(A_t)))

        # G2 proxy (only checks size, not diameter etc)
        G2_ok = (m_hat_int >= float(cfg.m_req)).astype(int)

        # gamma0 with theorem m (only when G2_ok)
        m_fixed = np.full_like(N_t, float(cfg.m_req), dtype=float)
        gamma0_full_all = gamma0_full(N=N_t, A=A_t, m=m_fixed, cfg=cfg)
        gamma0_full_valid = np.where(G2_ok == 1, gamma0_full_all, np.nan)

        # empirical gamma0_hat with m_hat (always defined; becomes 0 if m_hat<4 via clamping)
        gamma0_hat_full = gamma0_full(N=N_t, A=A_t, m=m_hat_int, cfg=cfg)

        # Compose per-run output for this time slice
        out = df.copy()
        out["time_tag"] = ttag
        out["N_t"] = N_t
        out["A_t"] = A_t
        out["cluster_M"] = cluster_M
        out["m_hat"] = m_hat_int
        out["G2_ok"] = G2_ok

        out["gamma0_const"] = cfg.const_factor
        out["gamma0_comb_fixedm"] = gamma0_comb(N=N_t, A=A_t, m=m_fixed, p=cfg.p)
        out["gamma0_full_valid"] = gamma0_full_valid

        out["gamma0_comb_hat"] = gamma0_comb(N=N_t, A=A_t, m=m_hat_int, p=cfg.p)
        out["gamma0_hat_full"] = gamma0_hat_full

        # Write per-run + summary for this time slice
        per_path, sum_path = io.write(out, tag=f"{tag}.{ttag}")

        # Print one-line diagnostics
        if verbose:
            summary = pd.read_csv(sum_path)
            # Top few functions with largest gamma0_hat_med
            top = summary.sort_values("gamma0_hat_med", ascending=False).head(8)
            print("\nTop functions by gamma0_hat_med at", ttag)
            print(top[["func", "G2_ok_frac", "gamma0_hat_med", "gamma0_med", "N_med", "A_med"]].to_string(index=False))

        if args.plots:
            summary = pd.read_csv(sum_path)
            io.plot_bar(summary, tag=f"{tag}.{ttag}")

    # Compute time-uniform gamma0_min
    if cfg.N_max is None:
        N_max_use = int(N_max_obs) if np.isfinite(N_max_obs) else None
    else:
        N_max_use = int(cfg.N_max)

    if cfg.A_max is None:
        A_max_use = int(A_max_obs) if np.isfinite(A_max_obs) else None
    else:
        A_max_use = int(cfg.A_max)

    if verbose:
        print("\n--- gamma0 constants ---")
        print(f"const_factor = (1/H)*g^-*(F^+-F^-)*q^- = {cfg.const_factor:.6g}")
        print(f"Using m = {cfg.m_req}, p = {cfg.p}, H = {cfg.H}")
        if N_max_use is not None and A_max_use is not None:
            gamma0_min = float(gamma0_full(
                N=np.asarray([float(N_max_use)]),
                A=np.asarray([float(A_max_use)]),
                m=np.asarray([float(cfg.m_req)]),
                cfg=cfg,
            )[0])
            gamma0_min_comb = float(gamma0_comb(
                N=np.asarray([float(N_max_use)]),
                A=np.asarray([float(A_max_use)]),
                m=np.asarray([float(cfg.m_req)]),
                p=cfg.p,
            )[0])
            print(f"N_max = {N_max_use} ({'user' if cfg.N_max is not None else 'observed'}), A_max = {A_max_use} ({'user' if cfg.A_max is not None else 'observed'})")
            print(f"gamma0_min_comb = {gamma0_min_comb:.6g}")
            print(f"gamma0_min_full = {gamma0_min:.6g}")
        else:
            print("Could not infer N_max/A_max from data; provide --N_max and --A_max to compute gamma0_min.")


if __name__ == "__main__":
    main()
