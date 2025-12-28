# analysis/combine_budgets.py
"""
Combine L-SHADE results across different budget levels (100k, 1M, 200M).
For each function, pick data from minimal budget where hit rate > 50%.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from analysis.hazard_utils import analyze_function, compute_event_data, kaplan_meier

# ======================================================
# CONFIG
# ======================================================
CEC2017_OPTIMA = {f"cec2017_f{i}": 100.0 * i for i in range(1, 31)}
EPSILON = 1e-2

EXPERIMENTS = {
    "100k": "experiments/r_lshade_D10_nfev_100000/raw_results_lshade.pkl",
    "1M": "experiments/r_lshade_D10_nfev_1000000/raw_results_lshade.pkl",
    "200M": "experiments/r_lshade_D10_nfev_200000000/raw_results_lshade.pkl",
}

BUDGET_TO_NFEV = {
    "100k": 100_000,
    "1M": 1_000_000,
    "200M": 200_000_000,
}

def short_name(fname):
    """cec2017_f5 -> f5"""
    return fname.replace("cec2017_", "")

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ======================================================
# HELPERS
# ======================================================
def load_data(pkl_path: str) -> dict:
    p = Path(pkl_path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ======================================================
# MAIN
# ======================================================
def main():
    print("=" * 120)
    print("COMBINED BUDGET ANALYSIS: L-SHADE, D=10, ε=1e-2")
    print("Selecting minimal budget with hit rate > 50%")
    print("=" * 120)

    # Load all available data
    all_data = {}
    for label, path in EXPERIMENTS.items():
        data = load_data(path)
        if data is not None:
            all_data[label] = data
            print(f"Loaded {label}: {path}")
        else:
            print(f"Missing {label}: {path}")

    if not all_data:
        print("No data found!")
        return

    budgets = ["100k", "1M", "200M"]
    budgets = [b for b in budgets if b in all_data]

    # All 30 functions
    funcs = [f"cec2017_f{i}" for i in range(1, 31)]

    # Analyze each function at each budget
    all_results = {label: {} for label in budgets}
    for label in budgets:
        data = all_data[label]
        for fname in funcs:
            if fname in data:
                f_star = CEC2017_OPTIMA[fname]
                curves = [run["curve"] for run in data[fname]]
                r = analyze_function(curves, f_star, EPSILON)
                all_results[label][fname] = r

    # Select best budget for each function (minimal with rate > 50%)
    selected = {}  # fname -> (budget_label, result)
    for fname in funcs:
        chosen_budget = None
        chosen_result = None
        
        for b in budgets:
            if fname in all_results[b]:
                r = all_results[b][fname]
                if r["hit_rate"] > 0.5:
                    chosen_budget = b
                    chosen_result = r
                    break  # take first (minimal) budget with > 50%
        
        # If none > 50%, take the highest budget available
        if chosen_budget is None:
            for b in reversed(budgets):
                if fname in all_results[b]:
                    chosen_budget = b
                    chosen_result = all_results[b][fname]
                    break
        
        selected[fname] = (chosen_budget, chosen_result)

    # Print table
    print("\n")
    print(f"{'Func':<6} {'NFEV':>12} {'Hits':>7} {'Rate':>6} {'T':>8} "
          f"{'τ_min':>8} {'τ_med':>8} {'τ_max':>8} "
          f"{'p_hat':>10} {'a_valid':>10} {'a/p':>8}")
    print("-" * 115)

    for fname in funcs:
        budget, r = selected[fname]
        sname = short_name(fname)
        
        if r is None:
            print(f"{sname:<6} {'--':>12} {'--':>7} {'--':>6} {'--':>8} "
                  f"{'--':>8} {'--':>8} {'--':>8} "
                  f"{'--':>10} {'--':>10} {'--':>8}")
            continue

        nfev = BUDGET_TO_NFEV[budget]
        hits_str = f"{r['hits']}/{r['n_runs']}"
        rate_str = f"{r['hit_rate']:.1%}"
        T_str = f"{r['T']}" if r["T"] is not None else "--"
        tau_min = f"{r['tau_min']}" if r["tau_min"] is not None else "--"
        tau_med = f"{r['tau_median']:.0f}" if r["tau_median"] is not None else "--"
        tau_max = f"{r['tau_max']}" if r["tau_max"] is not None else "--"
        p_str = f"{r['p_cens']:.4e}" if r["p_cens"] is not None else "--"
        a_str = f"{r['a_valid']:.4e}" if r["a_valid"] is not None else "--"
        ratio_str = f"{r['ratio']:.4f}" if r["ratio"] is not None else "--"

        print(f"{sname:<6} {nfev:>12} {hits_str:>7} {rate_str:>6} {T_str:>8} "
              f"{tau_min:>8} {tau_med:>8} {tau_max:>8} "
              f"{p_str:>10} {a_str:>10} {ratio_str:>8}")

    print("-" * 115)

    # ======================================================
    # SUMMARY
    # ======================================================
    print("\n" + "=" * 80)
    print("SUMMARY BY SELECTED BUDGET")
    print("=" * 80)
    
    budget_counts = {b: 0 for b in budgets}
    for fname, (b, r) in selected.items():
        if b:
            budget_counts[b] += 1
    
    for b in budgets:
        print(f"  {b}: {budget_counts[b]} functions")

    n_above_50 = sum(1 for f, (b, r) in selected.items() if r and r["hit_rate"] > 0.5)
    n_zero = sum(1 for f, (b, r) in selected.items() if r and r["hits"] == 0)
    print(f"\nFunctions with rate > 50%: {n_above_50}/30")
    print(f"Functions with 0 hits: {n_zero}/30")

    # ======================================================
    # FIGURES - one per budget, only functions selected from that budget
    # ======================================================
    for label in budgets:
        funcs_this_budget = [f for f in funcs 
                            if selected[f][0] == label 
                            and selected[f][1] is not None
                            and selected[f][1]["hits"] >= 5]

        if not funcs_this_budget:
            print(f"\nNo functions with ≥5 hits selected from {label}, skipping plot.")
            continue

        data = all_data[label]
        n_cols = 4
        n_rows = (len(funcs_this_budget) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1 and n_cols > 1:
            axes = axes.reshape(1, -1)
        axes = np.atleast_2d(axes).flatten()

        for idx, fname in enumerate(funcs_this_budget):
            ax = axes[idx]
            f_star = CEC2017_OPTIMA[fname]
            curves = [run["curve"] for run in data[fname]]
            ev = compute_event_data(curves, f_star, EPSILON)
            km = kaplan_meier(ev.y, ev.delta)
            r = selected[fname][1]

            ax.plot(km["t_vals"], km["S_hat"], lw=1.5, color="steelblue")

            if r["T"] is not None:
                ax.axvline(r["T"], color="red", ls="--", alpha=0.6)

            title = f"{short_name(fname)} | {r['hits']}/{r['n_runs']}"
            if r["ratio"] is not None:
                title += f" | a/p={r['ratio']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Generation")
            ax.set_ylabel("P(τ > n)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)

        for j in range(len(funcs_this_budget), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Survival Curves - Budget {label}", fontsize=14, y=1.02)
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"survival_curves_{label}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
