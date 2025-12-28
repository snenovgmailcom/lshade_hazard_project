#!/usr/bin/env python3
"""
Compute η_r(d, c_0) - the dimension-friendly crossover bound.
"""

from scipy.stats import binom
import numpy as np

def eta_r(d, c0, r):
    """
    Compute η_r(d, c_0) = P(Binomial(d-1, 1-c0) ≤ r)
    = sum_{k=0}^r C(d-1,k) (1-c0)^k c0^{d-1-k}
    """
    return binom.cdf(r, d-1, 1-c0)


def compare_bounds(d, c0):
    """Compare old vs new bound."""
    old = c0 ** (d-1)
    r_half = (d-1) // 2
    new = eta_r(d, c0, r_half)
    
    print(f"d={d}, c0={c0}:")
    print(f"  Old bound: c0^(d-1) = {old:.2e}")
    print(f"  New bound: eta_{r_half}(d,c0) = {new:.4f}")
    print(f"  Improvement: {new/old:.1e}x")
    return old, new


if __name__ == "__main__":
    print("=" * 50)
    print("COMPARISON: OLD vs NEW CROSSOVER BOUND")
    print("=" * 50)
    print()
    
    for d in [10, 20, 30, 50, 100]:
        compare_bounds(d, 0.5)
        print()
    
    # Table for paper
    print("=" * 50)
    print("TABLE FOR PAPER")
    print("=" * 50)
    print()
    print(f"{'d':>5} {'c0':>6} {'r':>5} {'c0^(d-1)':>15} {'eta_r':>10} {'Improvement':>15}")
    print("-" * 60)
    
    for d in [10, 20, 30, 50, 100]:
        c0 = 0.5
        r = (d-1) // 2
        old = c0 ** (d-1)
        new = eta_r(d, c0, r)
        improvement = new / old
        print(f"{d:>5} {c0:>6} {r:>5} {old:>15.2e} {new:>10.4f} {improvement:>15.1e}")
    
    # Vary r for fixed d
    print()
    print("=" * 50)
    print("EFFECT OF r FOR d=10, c0=0.5")
    print("=" * 50)
    print()
    print(f"{'r':>5} {'eta_r':>10} {'vs c0^(d-1)':>15}")
    print("-" * 35)
    
    d = 10
    c0 = 0.5
    old = c0 ** (d-1)
    
    for r in range(d):
        new = eta_r(d, c0, r)
        print(f"{r:>5} {new:>10.6f} {new/old:>15.1f}x")
    
    # Vary c0
    print()
    print("=" * 50)
    print("EFFECT OF c0 FOR d=10, r=4")
    print("=" * 50)
    print()
    print(f"{'c0':>6} {'c0^(d-1)':>15} {'eta_4':>10} {'Improvement':>15}")
    print("-" * 50)
    
    d = 10
    r = 4
    
    for c0 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        old = c0 ** (d-1)
        new = eta_r(d, c0, r)
        print(f"{c0:>6} {old:>15.6f} {new:>10.4f} {new/old:>15.1f}x")
