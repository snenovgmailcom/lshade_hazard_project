#!/usr/bin/env python3
"""
cluster_analysis_corrected.py

Strict G2 ∧ G3 proxy via m-clique existence check at the *hitting time*.

This is stricter than the ball-based concentration count in cluster_analysis.py.
For a given generation t and radius r_conc, we build the graph whose vertices
are population members and edges connect pairs with distance <= r_conc.

We say "G2∧G3 holds (m-clique)" if there exists a clique of size m that contains
the current best individual.

This script reproduces the "CORRECTED m-clique check" workflow used in the logs.

Note:
- The paper's Appendix also defines a more easily checkable proxy based on
  n_clust(t) counts around the best. That proxy is implemented in cluster_analysis.py.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_pkl(pkl_path: Path) -> Dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _func_key(data: Dict, func: str) -> str:
    func = func.strip()
    if func in data:
        return func
    if func.startswith("f") and func[1:].isdigit():
        k = f"cec2017_{func}"
        if k in data:
            return k
    # fall back: try exact match ignoring prefix
    for k in data.keys():
        if k.endswith(f"_{func}") or k.endswith(func):
            return k
    raise KeyError(f"Function {func} not found in PKL keys.")


def f_star_from_key(key: str) -> float:
    # CEC2017 convention: optimum is 100*i
    import re
    m = re.search(r"f(\d+)", key)
    if not m:
        raise ValueError(f"Cannot parse function id from key={key}")
    i = int(m.group(1))
    return float(100 * i)


def compute_hitting_times(curves: List[np.ndarray], f_star: float, eps: float) -> np.ndarray:
    taus: List[float] = []
    thr = f_star + float(eps)
    for curve in curves:
        hit = np.where(np.asarray(curve) <= thr)[0]
        taus.append(float(hit[0]) if hit.size else np.inf)
    return np.asarray(taus, dtype=float)


def has_m_clique_containing_best(positions: np.ndarray, best_idx: int, r_conc: float, m: int = 4) -> bool:
    """
    Check existence of an m-clique containing best_idx in the threshold graph dist<=r_conc.
    Currently optimized for m=4 (triangle among neighbors).
    """
    X = np.asarray(positions, dtype=float)
    N = X.shape[0]
    b = int(best_idx)

    if m <= 1:
        return True
    if m == 2:
        # need at least one neighbor
        d = np.linalg.norm(X - X[b], axis=1)
        return bool(np.any((d <= r_conc) & (np.arange(N) != b)))

    # neighbors of best (excluding itself)
    d_best = np.linalg.norm(X - X[b], axis=1)
    neigh = np.where((d_best <= r_conc) & (np.arange(N) != b))[0]
    if neigh.size < (m - 1):
        return False

    if m != 4:
        # Generic fallback: build adjacency among neighbors and brute-force combinations.
        # Only intended for small neighbor sets.
        Y = X[neigh]
        diff = Y[:, None, :] - Y[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        adj = dist <= r_conc
        # search for (m-1)-clique in neighbors
        import itertools
        for comb in itertools.combinations(range(neigh.size), m - 1):
            ok = True
            for i in range(len(comb)):
                for j in range(i + 1, len(comb)):
                    if not adj[comb[i], comb[j]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
        return False

    # m=4: need a triangle among neighbors
    Y = X[neigh]
    diff = Y[:, None, :] - Y[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    adj = dist <= r_conc
    np.fill_diagonal(adj, False)

    # For each edge (i,j), check if there exists k adjacent to both
    # Vectorized: compute common = adj @ adj (boolean) but careful; use logical intersection.
    M = adj.shape[0]
    for i in range(M):
        # candidates connected to i
        nbr_i = np.where(adj[i])[0]
        if nbr_i.size < 2:
            continue
        for j in nbr_i:
            if j <= i:
                continue
            # common neighbor of i and j
            if np.any(adj[i] & adj[j]):
                return True
    return False


def beta1_ball(positions: np.ndarray, best_idx: int, r_conc: float) -> float:
    X = np.asarray(positions, dtype=float)
    b = int(best_idx)
    d = np.linalg.norm(X - X[b], axis=1)
    return float(np.mean(d <= r_conc))


def analyze_function(
    runs: List[Dict],
    f_star: float,
    eps_list: List[float],
    r_list: List[float],
    outdir: Path,
    m: int = 4,
) -> pd.DataFrame:
    out_rows = []
    curves = [np.asarray(r["curve"], dtype=float) for r in runs]
    B = int(max(len(c) for c in curves) - 1)

    for eps in eps_list:
        taus = compute_hitting_times(curves, f_star=f_star, eps=eps)
        for r_conc in r_list:
            sat = []
            beta1s = []
            for run, tau in zip(runs, taus):
                if not np.isfinite(tau):
                    sat.append(False)
                    continue
                tau = int(tau)
                hist = run.get("history", {})
                pos_hist = hist.get("positions", None)
                fit_hist = hist.get("fitness", None)
                if pos_hist is None or fit_hist is None or tau >= len(pos_hist):
                    sat.append(False)
                    continue
                X = np.asarray(pos_hist[tau], dtype=float)
                f = np.asarray(fit_hist[tau], dtype=float)
                b = int(np.argmin(f))
                sat.append(has_m_clique_containing_best(X, b, r_conc=r_conc, m=m))
                beta1s.append(beta1_ball(X, b, r_conc=r_conc))

            n_hits = int(np.isfinite(taus).sum())
            rate = float(np.mean(sat)) if n_hits > 0 else 0.0
            beta1_med = float(np.median(beta1s)) if beta1s else np.nan
            out_rows.append({"eps": eps, "r_conc": r_conc, "g2g3_rate": rate, "n_hits": n_hits, "beta1_median": beta1_med})

    df = pd.DataFrame(out_rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "cluster_results.csv", index=False)

    # Pivot table for convenience
    pivot = df.pivot(index="eps", columns="r_conc", values="g2g3_rate")
    pivot.to_csv(outdir / "g2g3_rate_table.csv")

    pivot_beta = df.pivot(index="eps", columns="r_conc", values="beta1_median")
    pivot_beta.to_csv(outdir / "beta1_table.csv")

    # simple visualization
    plt.figure()
    for eps in eps_list:
        sub = df[df["eps"] == eps].sort_values("r_conc")
        plt.plot(sub["r_conc"], sub["g2g3_rate"], marker="o", label=f"eps={eps:g}")
    plt.xlabel("r_conc")
    plt.ylabel("G2∧G3 rate (m-clique containing best)")
    plt.title(f"G2∧G3 rate at hitting time (m={m})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "g2g3_rate_vs_r.png", dpi=200)
    plt.close()

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=str)
    ap.add_argument("--func", required=True, type=str, help="e.g. f5 or cec2017_f5")
    ap.add_argument("--dim", required=True, type=int)
    ap.add_argument("--outdir", required=True, type=str)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--eps", type=str, default="0.01,1,10,100,400", help="comma-separated")
    ap.add_argument("--r", type=str, default="0.5,1,2,5,10,20", help="comma-separated")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    data = _load_pkl(pkl_path)
    key = _func_key(data, args.func)
    runs = data[key]
    f_star = f_star_from_key(key)

    eps_list = [float(x) for x in args.eps.split(",") if x.strip()]
    r_list = [float(x) for x in args.r.split(",") if x.strip()]

    outdir = Path(args.outdir) / f"D{args.dim}" / args.func.replace("cec2017_", "")
    print("\n" + "=" * 70)
    print(f"CORRECTED G2 ∧ G3 ANALYSIS: {key}")
    print(f"Output: {outdir}")
    print(f"(Using {args.m}-clique existence check)")
    print("=" * 70 + "\n")

    analyze_function(runs, f_star=f_star, eps_list=eps_list, r_list=r_list, outdir=outdir, m=args.m)

    print(f"Saved: {outdir / 'cluster_results.csv'}")
    print(f"Saved: {outdir / 'g2g3_rate_table.csv'}")
    print(f"Saved: {outdir / 'beta1_table.csv'}")
    print(f"Saved: {outdir / 'g2g3_rate_vs_r.png'}")


if __name__ == "__main__":
    main()
