#!/usr/bin/env python3
"""
Unified Benchmark: PURE classical L-SHADE on CEC2017.

- Uses a global ProcessPoolExecutor over ALL (function, seed) tasks.
- Same CLI and output structure as q_benchmark.py
- Results go by default to: experiments/D{dim}/
"""

import os
import sys
import time
import argparse
import pickle
from pathlib import Path
import concurrent.futures

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import classical LSHADE
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
ALGO = ROOT / "algorithms"
sys.path.insert(0, str(ALGO))

try:
    from lshade import LSHADE  # type: ignore
except ImportError:
    print("ERROR: lshade.py not found in algorithms/ directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load CEC2017 functions
# ---------------------------------------------------------------------------

try:
    from cec2017.functions import (
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
        f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
        f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
    )
    from functools import partial
except ImportError:
    print("ERROR: cec2017 package not found! Install with: pip install cec2017")
    sys.exit(1)


def cec_wrap(f, x):
    """Ensure CEC function returns a scalar float."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    val = f(x)
    try:
        return float(val[0])
    except Exception:
        return float(val)


CEC2017_FUNCTIONS = {
    f"cec2017_f{i}": {
        "func": partial(cec_wrap, f),
        "f_global": 100.0 * i,
    }
    for i, f in enumerate(
        [
            f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
            f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
            f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
        ],
        start=1,
    )
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pad_and_envelope(curves):
    if not curves:
        return np.array([]), np.array([]), np.array([]), 0
    lengths = [len(c) for c in curves]
    target = int(np.median(lengths))
    arr = np.array([
        np.pad(c, (0, max(0, target - len(c))), "edge")[:target]
        for c in curves
    ])
    med = np.median(arr, axis=0)
    mn = np.min(arr, axis=0)
    mx = np.max(arr, axis=0)
    return med, mn, mx, target


def plot_envelope(curves, out_path, title):
    med, mn, mx, L = pad_and_envelope(curves)
    if L == 0:
        return
    iters = np.arange(1, L + 1)
    plt.figure()
    plt.plot(iters, med, label="Median best-so-far")
    plt.fill_between(iters, mn, mx, alpha=0.25, label="Min–Max")
    plt.xlabel("Generation")
    plt.ylabel("Best-so-far Fitness")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def convert_history_for_pickle(history):
    """
    Convert history arrays to lists for efficient pickling.
    
    Handles all logged data:
    - memory_f, memory_cr: list of arrays -> list of lists
    - pop_size, archive_size: list of ints (unchanged)
    - positions: list of (N_t, d) arrays -> list of lists
    - fitness: list of (N_t,) arrays -> list of lists
    - x_best: list of (d,) arrays -> list of lists
    - f_best: list of floats (unchanged)
    - all_F, all_CR: list of (N_t,) arrays -> list of lists
    - successful_F, successful_CR, delta_f: list of arrays -> list of lists
    """
    converted = {}
    
    for key, val in history.items():
        if isinstance(val, list) and len(val) > 0:
            first = val[0]
            if isinstance(first, np.ndarray):
                # Convert numpy arrays to lists
                converted[key] = [arr.tolist() for arr in val]
            else:
                # Keep as-is (scalars, already lists)
                converted[key] = val
        else:
            converted[key] = val
    
    return converted

# ---------------------------------------------------------------------------
# Run a single LSHADE instance
# ---------------------------------------------------------------------------

def run_single(seed,
               fname,
               dim,
               popsize,
               init_factor,
               max_evals,
               N_min,
               disp):
    """
    One (function, seed) LSHADE run.
    """
    info = CEC2017_FUNCTIONS[fname]
    func = info["func"]
    bounds = (-100, 100)

    ps = int(init_factor * dim) if popsize == -1 else popsize
    bounds_list = [bounds] * dim

    solver = LSHADE(
        func=func,
        bounds=bounds_list,
        popsize=ps,
        N_min=N_min,
        max_evals=max_evals,
        atol=0.0,
        seed=seed,
        disp=disp,
    )

    t0 = time.perf_counter()
    res = solver.solve()
    wall = time.perf_counter() - t0

    curve = np.array(res.convergence if len(res.convergence) > 0 else [])
    if curve.size > 0:
        curve = np.minimum.accumulate(curve)

    final_pop = int(res.final_pop_size)

    # Validation: final_pop should be >= N_min
    if final_pop < N_min:
        print(f"[WARNING] {fname} seed={seed}: final_pop={final_pop} < N_min={N_min}")

    # Convert history arrays to lists for pickling
    history = convert_history_for_pickle(res.history)

    return {
        "function": fname,
        "seed": seed,
        "best": float(res.fun),
        "nfev": int(res.nfev),
        "nit": int(res.nit),
        "wall": float(wall),
        "curve": curve,
        "init_pop": ps,
        "final_pop": final_pop,
        "history": history,
    }


def run_single_entry(args):
    return run_single(*args)

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results):
    best_vals = np.array([r["best"] for r in results])
    nfevs = np.array([r["nfev"] for r in results])
    nits = np.array([r["nit"] for r in results])
    walls = np.array([r["wall"] for r in results])
    final_pops = np.array([r["final_pop"] for r in results])
    curves = [r["curve"] for r in results]

    return {
        "best_vals": best_vals,
        "nfevs": nfevs,
        "nits": nits,
        "walls": walls,
        "final_pops": final_pops,
        "curves": curves,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Benchmark pure LSHADE on CEC2017")

    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--max-evals", type=int, default=None)
    ap.add_argument("--runs", type=int, default=51)
    ap.add_argument("--popsize", type=int, default=-1)
    ap.add_argument("--init-factor", type=float, default=18.0)
    ap.add_argument("--N-min", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--functions", type=str, default="all")
    ap.add_argument("--disp", action="store_true")
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--figs", action="store_true", help="Generate plots")

    args = ap.parse_args()

    if args.max_evals is None:
        args.max_evals = 10000 * args.dim

    # Default outdir based on dim if not supplied
    if args.outdir is None:
        args.outdir = os.path.join("experiments", f"D{args.dim}")
    os.makedirs(args.outdir, exist_ok=True)

    # Select functions
    if args.functions.lower() == "all":
        func_names = list(CEC2017_FUNCTIONS.keys())
    else:
        indices = []
        for x in args.functions.split(","):
            x = x.strip()
            if x.startswith("f"):
                x = x[1:]
            indices.append(int(x))
        func_names = [f"cec2017_f{i}" for i in indices]

    seeds = [args.seed_start + i for i in range(args.runs)]

    # Build all tasks across functions × seeds
    tasks = []
    for fname in func_names:
        for s in seeds:
            tasks.append(
                (
                    s,
                    fname,
                    args.dim,
                    args.popsize,
                    args.init_factor,
                    args.max_evals,
                    args.N_min,
                    args.disp,
                )
            )

    total_tasks = len(tasks)
    print(f"Total tasks (functions × runs): {total_tasks}")

    max_workers = min(args.jobs, os.cpu_count() or args.jobs)
    print(f"Using up to {max_workers} worker processes.")

    results_by_fn = {fname: [] for fname in func_names}

    t0_global = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_single_entry, t) for t in tasks]
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                r = fut.result()
                results_by_fn[r["function"]].append(r)
            except Exception as e:
                print(f"  ERROR in run: {e}")

            if idx % 20 == 0 or idx == total_tasks:
                print(f"  Global progress: {idx}/{total_tasks}")

    wall_global = time.perf_counter() - t0_global
    print(f"Total wall time (all functions): {wall_global:.2f} s")

    # Aggregate per function
    summary_rows = []
    raw_results = {}

    for fname in func_names:
        info = CEC2017_FUNCTIONS[fname]
        f_global = info["f_global"]
        fn_results = results_by_fn[fname]

        if not fn_results:
            print(f"[WARNING] No results for {fname}")
            continue

        ag = aggregate(fn_results)
        best_vals = ag["best_vals"]
        curves = ag["curves"]
        final_pops = ag["final_pops"]

        errors = best_vals - f_global

        print("\n====================================================")
        print(f"Function: {fname}  | Optimum = {f_global}")
        print("====================================================")
        print(f"  Mean error  : {errors.mean():.4e}")
        print(f"  Best error  : {errors.min():.4e}")
        print(f"  Worst error : {errors.max():.4e}")
        print(f"  Final pop   : {final_pops.mean():.1f} (min={final_pops.min()}, max={final_pops.max()})")

        if args.figs:
            outfile = os.path.join(args.outdir, f"{fname}_lshade_envelope.png")
            plot_envelope(curves, outfile, f"{fname} (LSHADE)")
            print(f"  Plot saved: {outfile}")

        row = {
            "function": fname,
            "dim": args.dim,
            "runs": args.runs,
            "N_init": args.init_factor * args.dim
                     if args.popsize == -1 else args.popsize,
            "N_min": args.N_min,
            "best_mean": float(best_vals.mean()),
            "best_median": float(np.median(best_vals)),
            "best_std": float(best_vals.std()),
            "best_min": float(best_vals.min()),
            "best_max": float(best_vals.max()),
            "error_mean": float(errors.mean()),
            "error_median": float(np.median(errors)),
            "final_pop_mean": float(final_pops.mean()),
            "final_pop_min": int(final_pops.min()),
            "final_pop_max": int(final_pops.max()),
            "wall_total": float(wall_global),
        }
        summary_rows.append(row)
        raw_results[fname] = fn_results

    df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.outdir, "summary_lshade.csv")
    df.to_csv(summary_csv, index=False)

    raw_pkl = os.path.join(args.outdir, "raw_results_lshade.pkl")
    with open(raw_pkl, "wb") as f:
        pickle.dump(raw_results, f)

    print("\n====================================================")
    print("Benchmark complete!")
    print("====================================================")
    print(f"Summary CSV: {summary_csv}")
    print(f"Raw pickle: {raw_pkl}")


if __name__ == "__main__":
    main()
