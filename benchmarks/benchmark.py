#!/usr/bin/env python3
"""
Unified Benchmark: PURE classical L-SHADE on CEC2017.

- Processes functions sequentially, parallelizing seeds within each function.
- Saves PKL immediately after each function completes.
- Rebuilds summary CSV from saved PKLs at the end.

Outputs:
    - Per-function PKL: experiments/D{dim}/f{i}/f{i}.pkl
    - Summary CSV: experiments/D{dim}/summary_lshade.csv

Usage:
    # Run single function
    python benchmark.py --dim 10 --functions f1 --runs 51
    
    # Run all functions
    python benchmark.py --dim 10 --functions all --runs 51 --jobs 160
    
    # Run multiple dimensions
    for d in 10 30 50 100; do
        python benchmark.py --dim $d --functions all --runs 51 --jobs 160
    done
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
    """
    converted = {}
    
    for key, val in history.items():
        if isinstance(val, list) and len(val) > 0:
            first = val[0]
            if isinstance(first, np.ndarray):
                converted[key] = [arr.tolist() for arr in val]
            else:
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

    if final_pop < N_min:
        print(f"[WARNING] {fname} seed={seed}: final_pop={final_pop} < N_min={N_min}")

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


def build_summary_row(fname, fn_results, dim, runs, popsize, init_factor, N_min, wall_fn):
    """Build a summary row for one function from its results."""
    info = CEC2017_FUNCTIONS[fname]
    f_global = info["f_global"]
    
    ag = aggregate(fn_results)
    best_vals = ag["best_vals"]
    final_pops = ag["final_pops"]
    errors = best_vals - f_global
    
    return {
        "function": fname,
        "dim": dim,
        "runs": runs,
        "N_init": init_factor * dim if popsize == -1 else popsize,
        "N_min": N_min,
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
        "wall_fn": float(wall_fn),
    }


def build_summary_from_pkls(outdir, func_names, dim, runs, popsize, init_factor, N_min):
    """Rebuild summary CSV by loading all per-function PKLs."""
    summary_rows = []
    
    for fname in func_names:
        func_short = fname.replace("cec2017_", "")
        pkl_path = os.path.join(outdir, func_short, f"{func_short}.pkl")
        
        if not os.path.exists(pkl_path):
            print(f"  [WARNING] Missing PKL for {fname}, skipping in summary")
            continue
        
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        fn_results = data[fname]
        
        # Wall time not available when rebuilding, set to 0
        row = build_summary_row(fname, fn_results, dim, runs, popsize, init_factor, N_min, wall_fn=0.0)
        summary_rows.append(row)
        
        print(f"  Loaded {func_short}: {len(fn_results)} runs")
    
    return summary_rows

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
    ap.add_argument("--summary-only", action="store_true", 
                    help="Only rebuild summary CSV from existing PKLs")

    args = ap.parse_args()

    if args.max_evals is None:
        args.max_evals = 10000 * args.dim

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

    # -------------------------------------------------------------------------
    # Summary-only mode: rebuild CSV from existing PKLs
    # -------------------------------------------------------------------------
    if args.summary_only:
        print("Rebuilding summary from existing PKLs...")
        summary_rows = build_summary_from_pkls(
            args.outdir, func_names, args.dim, args.runs,
            args.popsize, args.init_factor, args.N_min
        )
        
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            summary_csv = os.path.join(args.outdir, "summary_lshade.csv")
            df.to_csv(summary_csv, index=False)
            print(f"\nSummary CSV saved: {summary_csv}")
        else:
            print("\nNo PKLs found, no summary generated.")
        return

    # -------------------------------------------------------------------------
    # Normal mode: run experiments
    # -------------------------------------------------------------------------
    seeds = [args.seed_start + i for i in range(args.runs)]
    max_workers = min(args.jobs, os.cpu_count() or args.jobs)
    
    print(f"Functions to run: {len(func_names)}")
    print(f"Runs per function: {args.runs}")
    print(f"Using up to {max_workers} worker processes")
    print(f"Output directory: {args.outdir}")
    print()

    t0_global = time.perf_counter()
    summary_rows = []

    # -------------------------------------------------------------------------
    # Process each function sequentially, parallelize seeds within
    # -------------------------------------------------------------------------
    for func_idx, fname in enumerate(func_names, 1):
        info = CEC2017_FUNCTIONS[fname]
        f_global = info["f_global"]
        func_short = fname.replace("cec2017_", "")
        
        print(f"\n{'='*60}")
        print(f"[{func_idx}/{len(func_names)}] {fname} (optimum={f_global})")
        print(f"{'='*60}")
        
        # Build tasks for this function only
        tasks = [
            (s, fname, args.dim, args.popsize, args.init_factor,
             args.max_evals, args.N_min, args.disp)
            for s in seeds
        ]
        
        fn_results = []
        t0_fn = time.perf_counter()
        
        # Run all seeds for this function in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_single_entry, t) for t in tasks]
            for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    r = fut.result()
                    fn_results.append(r)
                except Exception as e:
                    print(f"  ERROR in seed run: {e}")
                
                if idx % 10 == 0 or idx == len(tasks):
                    print(f"  Progress: {idx}/{len(tasks)} seeds completed")
        
        wall_fn = time.perf_counter() - t0_fn
        
        if not fn_results:
            print(f"  [WARNING] No results for {fname}")
            continue
        
        # Print function summary
        ag = aggregate(fn_results)
        best_vals = ag["best_vals"]
        errors = best_vals - f_global
        final_pops = ag["final_pops"]
        
        print(f"  Mean error  : {errors.mean():.4e}")
        print(f"  Best error  : {errors.min():.4e}")
        print(f"  Worst error : {errors.max():.4e}")
        print(f"  Final pop   : {final_pops.mean():.1f} (min={final_pops.min()}, max={final_pops.max()})")
        print(f"  Wall time   : {wall_fn:.1f}s")
        
        # Generate plot if requested
        if args.figs:
            outfile = os.path.join(args.outdir, f"{fname}_lshade_envelope.png")
            plot_envelope(ag["curves"], outfile, f"{fname} (LSHADE)")
            print(f"  Plot saved  : {outfile}")
        
        # ---------------------------------------------------------------------
        # Save PKL immediately after function completes
        # ---------------------------------------------------------------------
        func_dir = os.path.join(args.outdir, func_short)
        os.makedirs(func_dir, exist_ok=True)
        
        pkl_path = os.path.join(func_dir, f"{func_short}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({fname: fn_results}, f)
            f.flush()
            os.fsync(f.fileno())  # Ensure data hits disk
        
        size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
        print(f"  PKL saved   : {pkl_path} ({size_mb:.1f} MB)")
        
        # Build summary row
        row = build_summary_row(
            fname, fn_results, args.dim, args.runs,
            args.popsize, args.init_factor, args.N_min, wall_fn
        )
        summary_rows.append(row)
        
        # Clear memory
        del fn_results
        del ag

    # -------------------------------------------------------------------------
    # Save summary CSV
    # -------------------------------------------------------------------------
    wall_global = time.perf_counter() - t0_global
    
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.outdir, "summary_lshade.csv")
        df.to_csv(summary_csv, index=False)
        
        print(f"\n{'='*60}")
        print("Benchmark complete!")
        print(f"{'='*60}")
        print(f"Total wall time: {wall_global:.1f}s")
        print(f"Summary CSV: {summary_csv}")
        print(f"Per-function PKLs: {args.outdir}/f*/f*.pkl")
    else:
        print("\n[WARNING] No results collected, no summary generated.")


if __name__ == "__main__":
    main()
