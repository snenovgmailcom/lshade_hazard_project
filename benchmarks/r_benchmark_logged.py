#!/usr/bin/env python3
"""
Benchmark L-SHADE with population logging for ℓ analysis.

Runs fewer runs (e.g., 5-10) but logs full population history.
"""

import os
import sys
import time
import argparse
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ALGO = ROOT / "algorithms"
ANALYSIS = ROOT / "analysis"
sys.path.insert(0, str(ALGO))
sys.path.insert(0, str(ANALYSIS))

from r_lshade_logged import LSHADE_Logged
from compute_ell import analyze_all_generations

# CEC2017 functions
try:
    from cec2017.functions import f1, f2, f3
    from functools import partial
except ImportError:
    print("ERROR: cec2017 package not found!")
    sys.exit(1)


def cec_wrap(f, x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    val = f(x)
    try:
        return float(val[0])
    except Exception:
        return float(val)


# For F1 (shifted sphere), optimum is at origin shifted
# CEC2017 F1: f* = 100, x* is shifted (we approximate as 0 for sphere analysis)
CEC2017_F1 = {
    "func": partial(cec_wrap, f1),
    "f_star": 100.0,
    "x_star_approx": None,  # Will be set based on dimension
}


def run_logged(seed, dim, max_evals, log_every=1, log_max_gens=None):
    """Run one L-SHADE with logging."""
    
    func = CEC2017_F1["func"]
    f_star = CEC2017_F1["f_star"]
    bounds = [(-100, 100)] * dim
    
    solver = LSHADE_Logged(
        func=func,
        bounds=bounds,
        popsize=18 * dim,
        N_min=4,
        max_evals=max_evals,
        seed=seed,
        disp=False,
        log_every=log_every,
        log_max_gens=log_max_gens,
    )
    
    t0 = time.perf_counter()
    res = solver.solve()
    wall = time.perf_counter() - t0
    
    return {
        "seed": seed,
        "best": res.fun,
        "nit": res.nit,
        "nfev": res.nfev,
        "wall": wall,
        "convergence": res.convergence,
        "generation_logs": res.generation_logs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--max-evals", type=int, default=100000)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--log-max-gens", type=int, default=None)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--epsilon", type=float, default=1e-2)
    ap.add_argument("--outdir", type=str, default="experiments/ell_analysis")
    ap.add_argument("--n-sample-tuples", type=int, default=1000)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    f_star = 100.0  # CEC2017 F1
    epsilon = args.epsilon
    
    print(f"Running {args.runs} logged L-SHADE runs on F1 (D={args.dim})")
    print(f"Max evals: {args.max_evals}, log every {args.log_every} generations")
    print(f"Target: f* + ε = {f_star} + {epsilon} = {f_star + epsilon}")
    print()
    
    all_results = []
    all_ell_analyses = []
    
    for r in range(args.runs):
        seed = args.seed_start + r
        print(f"Run {r+1}/{args.runs} (seed={seed})...", end=" ", flush=True)
        
        result = run_logged(
            seed=seed,
            dim=args.dim,
            max_evals=args.max_evals,
            log_every=args.log_every,
            log_max_gens=args.log_max_gens,
        )
        
        print(f"best={result['best']:.6e}, nit={result['nit']}, "
              f"logged {len(result['generation_logs'])} gens")
        
        # Analyze ℓ for this run
        # Note: For F1, we don't have exact x_star, so use sampling
        # But first let's just collect the logs
        all_results.append(result)
    
    # Save raw results with logs
    pkl_path = os.path.join(args.outdir, f"logged_runs_D{args.dim}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "args": vars(args),
            "results": all_results,
        }, f)
    print(f"\nSaved: {pkl_path}")
    
    # Now analyze ℓ using sampling (since we don't have exact x_star for CEC2017)
    print("\nAnalyzing ℓ statistics...")
    
    func = CEC2017_F1["func"]
    
    for r, result in enumerate(all_results):
        print(f"\nRun {r+1}: analyzing {len(result['generation_logs'])} generations...")
        
        ell_analysis = analyze_all_generations(
            generation_logs=result['generation_logs'],
            f_star=f_star,
            epsilon=epsilon,
            func=func,
            use_analytic=False,  # Use sampling since we don't have x_star
            n_sample_tuples=args.n_sample_tuples,
            seed=args.seed_start + r,
        )
        
        all_ell_analyses.append(ell_analysis)
        
        # Summary statistics
        gammas = [ea['gamma_hat'] for ea in ell_analysis]
        ell_means = [ea['ell_mean'] for ea in ell_analysis if ea['ell_mean'] > 0]
        
        print(f"  γ̂ range: [{min(gammas):.4f}, {max(gammas):.4f}]")
        print(f"  Mean γ̂: {np.mean(gammas):.4f}")
        if ell_means:
            print(f"  Mean ℓ (when positive): {np.mean(ell_means):.6f}")
    
    # Save ℓ analysis
    ell_pkl = os.path.join(args.outdir, f"ell_analysis_D{args.dim}.pkl")
    with open(ell_pkl, "wb") as f:
        pickle.dump({
            "args": vars(args),
            "ell_analyses": all_ell_analyses,
        }, f)
    print(f"\nSaved: {ell_pkl}")


if __name__ == "__main__":
    main()
