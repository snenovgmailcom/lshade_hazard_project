"""
Calibrate the tightest *uniform exponential envelope* for Corollary 1.

Goal:
  Find the largest constant a such that for all n >= T,

      S_hat(n) <= S_hat(T-1) * (1-a)^(n-T+1),

where S_hat is the Kaplan–Meier (KM) estimate of the survival curve S(n)=P(tau > n)
for a first-hit time tau.

Important:
  - a_valid is NOT a verified lower bound on the random algorithmic hazard
        p_t(omega) = P(E_t | F_{t-1})(omega).
    It is the tightest exponential envelope parameter consistent with the *empirical* S_hat.
  - For a "constant hazard after T" comparison, the correct MLE under right censoring is

        p_cens = (# events after T) / (total exposure after T),

    i.e. events / person-time. This differs from 1/mean(tau-T+1) when censoring exists.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

CEC2017_OPTIMA = {f"cec2017_f{i}": 100.0 * i for i in range(1, 31)}

# -----------------------------
# Configuration (edit as needed)
# -----------------------------
PKL_PATH = "/home/svety/lshade_project/experiments/r_lshade_D10/raw_results_lshade.pkl"
EPSILON = 1e-2

# Choose how to set T:
#   - "first_hit": T = min observed hit time among runs (data-dependent)
#   - "fixed":     T = T_FIXED (deterministic)
T_MODE = "first_hit"
T_FIXED = 200  # used only if T_MODE == "fixed"

MIN_HITS_TO_REPORT = 10
OUTPUT_DIR = Path("results")


def load_data(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


@dataclass
class EventData:
    y: np.ndarray       # observed time (hit time if event, else censor time)
    delta: np.ndarray   # 1 if event observed (hit), 0 if right-censored
    c: np.ndarray       # censor time per run (last available index)


def compute_event_data(curves: List[np.ndarray], f_star: float, epsilon: float) -> EventData:
    """
    Returns:
      y_i     = time of hit if hit occurs, else censor time (last index in curve)
      delta_i = 1 if hit occurs, else 0
      c_i     = censor time (last index in curve)
    """
    target = f_star + epsilon
    y, delta, c = [], [], []

    for curve in curves:
        # Assume curve is indexed by generation/evaluation step: 0,1,...,len(curve)-1
        c_i = len(curve) - 1
        hits = np.where(curve <= target)[0]

        if len(hits) > 0:
            t_hit = int(hits[0])
            y.append(t_hit)
            delta.append(1)
        else:
            y.append(c_i)
            delta.append(0)

        c.append(c_i)

    return EventData(y=np.array(y, dtype=int),
                     delta=np.array(delta, dtype=int),
                     c=np.array(c, dtype=int))


def kaplan_meier_discrete(y: np.ndarray, delta: np.ndarray, n_max: Optional[int] = None):
    """
    Discrete-time Kaplan–Meier for S(n) = P(tau > n).

    At time t:
      n_t = # at risk (still under observation at t): y_i >= t
      d_t = # events at t: delta_i=1 and y_i=t
      hazard estimate: h_hat(t) = d_t / n_t

    Survival:
      S_hat(n) = Π_{t=0..n} (1 - h_hat(t))  (with the convention that empty product is 1)
    """
    if n_max is None:
        n_max = int(np.max(y))

    t_vals = np.arange(0, n_max + 1)

    n_at_risk = np.zeros_like(t_vals, dtype=int)
    d_events = np.zeros_like(t_vals, dtype=int)
    h_hat = np.zeros_like(t_vals, dtype=float)

    for idx, t in enumerate(t_vals):
        n_t = int(np.sum(y >= t))
        d_t = int(np.sum((delta == 1) & (y == t)))
        n_at_risk[idx] = n_t
        d_events[idx] = d_t
        h_hat[idx] = (d_t / n_t) if n_t > 0 else 0.0

    S_hat = np.cumprod(1.0 - h_hat)

    return {
        "t_vals": t_vals,
        "S_hat": S_hat,          # S_hat[n] = P(tau > n)
        "h_hat": h_hat,          # marginal hazard estimate
        "n_at_risk": n_at_risk,
        "d_events": d_events,
        "n_max": n_max,
    }


def choose_T(y: np.ndarray, delta: np.ndarray) -> Optional[int]:
    if T_MODE == "fixed":
        return int(T_FIXED)
    # first_hit
    hits = y[delta == 1]
    if len(hits) == 0:
        return None
    return int(np.min(hits))


def constant_hazard_mle_censored(y: np.ndarray, delta: np.ndarray, T: int) -> Optional[float]:
    """
    Constant hazard MLE under right censoring, for the "shifted" model starting at T.

    Consider only runs that are still under observation at time T (y_i >= T).
    Exposure for run i after T is:
        Y_i = y_i - T + 1
    Event indicator after T is delta_i (since if y_i >= T and delta_i=1, the event is after T).

    MLE:
        p_cens = (sum delta_i) / (sum Y_i)   over y_i >= T
    """
    mask = (y >= T)
    if not np.any(mask):
        return None

    y_shift = (y[mask] - T + 1).astype(float)
    d = delta[mask].astype(float)

    total_exposure = float(np.sum(y_shift))
    total_events = float(np.sum(d))
    if total_exposure <= 0:
        return None

    return total_events / total_exposure


def find_max_valid_a(S_hat: np.ndarray, T: int) -> Tuple[Optional[float], List[Tuple[int, int, float, float]]]:
    """
    Find largest a such that for all n >= T,

        S_hat(n) <= S_hat(T-1) * (1-a)^(n-T+1).

    Rearranged:
        (S_hat(n)/S_hat(T-1)) <= (1-a)^k,  k = n-T+1
        => a <= 1 - (S_hat(n)/S_hat(T-1))^(1/k)

    Return:
      a_valid and a list of constraints (n, k, S_cond(n), a_max(n)).
    """
    if T < 0 or T >= len(S_hat):
        return None, []

    S0 = S_hat[T - 1] if T >= 1 else 1.0
    if S0 <= 0:
        return None, []

    constraints = []
    for n in range(T, len(S_hat)):
        k = n - T + 1
        S_n = float(S_hat[n])
        S_cond = S_n / float(S0)

        if 0.0 < S_cond < 1.0 and k >= 1:
            a_max = 1.0 - (S_cond ** (1.0 / k))
            constraints.append((n, k, S_cond, a_max))

    if not constraints:
        return None, []

    a_valid = min(x[3] for x in constraints)
    return float(a_valid), constraints


def make_bound_curve(n_vals: np.ndarray, T: int, S0: float, a: float) -> np.ndarray:
    """
    Build the envelope curve:
      bound(n) = S0 for n < T
      bound(n) = S0 * (1-a)^(n-T+1) for n >= T
    """
    bound = np.full_like(n_vals, fill_value=np.nan, dtype=float)
    bound[n_vals < T] = S0
    idx = (n_vals >= T)
    bound[idx] = S0 * (1.0 - a) ** (n_vals[idx] - T + 1)
    return bound


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_data(PKL_PATH)

    print("=" * 80)
    print("TIGHTEST VALID EXPONENTIAL ENVELOPE FOR COROLLARY 1 (EMPIRICAL)")
    print("=" * 80)
    print(f"Epsilon threshold = {EPSILON}")
    print(f"T_MODE = {T_MODE}" + (f", T_FIXED = {T_FIXED}" if T_MODE == "fixed" else ""))
    print()

    # -----------------------------
    # Detailed analysis on f1
    # -----------------------------
    func_name = "cec2017_f1"
    f_star = CEC2017_OPTIMA[func_name]
    curves = [run["curve"] for run in data[func_name]]
    ev = compute_event_data(curves, f_star, EPSILON)

    km = kaplan_meier_discrete(ev.y, ev.delta)
    T = choose_T(ev.y, ev.delta)
    if T is None:
        print(f"No hits for {func_name}; cannot analyze.")
        return

    hits = int(np.sum(ev.delta == 1))
    cens = int(np.sum(ev.delta == 0))
    S0 = km["S_hat"][T - 1] if T >= 1 else 1.0

    a_valid, constraints = find_max_valid_a(km["S_hat"], T)
    p_cens = constant_hazard_mle_censored(ev.y, ev.delta, T)

    print("=" * 60)
    print(f"DETAILED ANALYSIS: {func_name}")
    print("=" * 60)
    print(f"Runs: {len(ev.y)} | Hits: {hits} | Censored: {cens}")
    print(f"T (start) = {T}")
    print(f"S_hat(T-1) = {S0:.6f}")
    print(f"Max observed time (KM n_max) = {km['n_max']}")
    print()

    if a_valid is None:
        print("Could not compute a_valid (insufficient variation in S_hat after T).")
        return

    print(f"Maximum valid envelope rate a_valid = {a_valid:.6f}")
    if p_cens is not None:
        print(f"Constant-hazard MLE under censoring p_cens = {p_cens:.6f}")
        print(f"Ratio: a_valid / p_cens = {a_valid / p_cens:.4f}")
    else:
        print("p_cens could not be computed (no exposure after T).")

    print("\nBinding constraints (first few + last few):")
    print(f"{'n':>6} {'k':>6} {'S_cond(n)':>12} {'a_max':>10}")
    print("-" * 42)
    for row in constraints[:5]:
        n, k, S_cond, a_max = row
        mark = " <-- binding" if abs(a_max - a_valid) < 1e-12 else ""
        print(f"{n:>6} {k:>6} {S_cond:>12.6f} {a_max:>10.6f}{mark}")
    if len(constraints) > 10:
        print("  ...")
    for row in constraints[-5:]:
        n, k, S_cond, a_max = row
        mark = " <-- binding" if abs(a_max - a_valid) < 1e-12 else ""
        print(f"{n:>6} {k:>6} {S_cond:>12.6f} {a_max:>10.6f}{mark}")

    print("\n" + "=" * 60)
    print("INTERPRETATION (corrected)")
    print("=" * 60)
    print(f"""
a_valid is the tightest *uniform exponential envelope* consistent with the entire KM survival curve
after T:

    S_hat(n) <= S_hat(T-1) * (1-a_valid)^(n-T+1),   for all n >= T.

If a_valid << p_cens, that typically means:

1) The survival curve stays high (often flat) for a long time after T (few/no hits),
   so any *uniform* exponential envelope must decay very slowly to remain valid for all n.
2) Later, hits may arrive in a burst (hazard increases), which affects p_cens but does not
   increase a_valid because a_valid is constrained by the slowest-decay segment.
3) Therefore a_valid measures a conservative worst-case rate, while p_cens is a best-fit
   constant-hazard parameter (not guaranteed to upper-bound survival everywhere).

Bottom line: time-varying/clustered hazards make uniform-hazard bounds conservative, exactly as expected.
""")

    # -----------------------------
    # Plot: empirical vs envelopes
    # -----------------------------
    n_vals = km["t_vals"]  # same indexing (0..n_max)

    bound_valid = make_bound_curve(n_vals, T, S0, a_valid)
    bound_pcens = None
    if p_cens is not None and p_cens > 0:
        bound_pcens = make_bound_curve(n_vals, T, S0, p_cens)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(n_vals, km["S_hat"], "b-", lw=2, label="KM $\\hat S(n)$")
    axes[0].plot(n_vals, bound_valid, "g-", lw=2, label=f"Valid envelope: $a={a_valid:.6g}$")
    if bound_pcens is not None:
        axes[0].plot(n_vals, bound_pcens, "r--", lw=2, label=f"Const-hazard fit: $p={p_cens:.4g}$")
    axes[0].axvline(T, color="gray", ls="--", alpha=0.5)
    axes[0].set_xlim([max(0, T - 10), min(km["n_max"], T + 80)])
    axes[0].set_xlabel("Generation / index n")
    axes[0].set_ylabel("$\\hat S(n)=\\Pr(\\tau>n)$")
    axes[0].set_title(f"{func_name}: Survival (linear)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(n_vals, km["S_hat"] + 1e-12, "b-", lw=2, label="KM $\\hat S(n)$")
    axes[1].semilogy(n_vals, bound_valid + 1e-12, "g-", lw=2, label=f"Valid envelope: $a={a_valid:.6g}$")
    if bound_pcens is not None:
        axes[1].semilogy(n_vals, bound_pcens + 1e-12, "r--", lw=2, label=f"Const-hazard fit: $p={p_cens:.4g}$")
    axes[1].axvline(T, color="gray", ls="--", alpha=0.5)
    axes[1].set_xlim([max(0, T - 10), min(km["n_max"], T + 80)])
    axes[1].set_xlabel("Generation / index n")
    axes[1].set_ylabel("$\\hat S(n)$ (log scale)")
    axes[1].set_title("Survival (log)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    out_path = OUTPUT_DIR / "valid_envelope_f1.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")

    # -----------------------------
    # Summary table for all functions
    # -----------------------------
    print("\n" + "=" * 80)
    print("SUMMARY (all functions)")
    print("=" * 80)
    print(f"{'Function':<12} {'Hits':>5} {'Cens':>5} {'T':>6} {'p_cens':>10} {'a_valid':>10} {'a/p':>8}")
    print("-" * 70)

    for i in range(1, 31):
        fname = f"cec2017_f{i}"
        f_star = CEC2017_OPTIMA[fname]
        curves = [run["curve"] for run in data[fname]]
        ev = compute_event_data(curves, f_star, EPSILON)

        hits = int(np.sum(ev.delta == 1))
        cens = int(np.sum(ev.delta == 0))
        if hits < MIN_HITS_TO_REPORT:
            print(f"{fname:<12} {hits:>5} {cens:>5} {'--':>6} {'--':>10} {'--':>10} {'--':>8}")
            continue

        km = kaplan_meier_discrete(ev.y, ev.delta)
        T = choose_T(ev.y, ev.delta)
        if T is None or T >= len(km["S_hat"]):
            print(f"{fname:<12} {hits:>5} {cens:>5} {'--':>6} {'--':>10} {'--':>10} {'--':>8}")
            continue

        a_valid, _ = find_max_valid_a(km["S_hat"], T)
        p_cens = constant_hazard_mle_censored(ev.y, ev.delta, T)

        if a_valid is None or p_cens is None or p_cens <= 0:
            print(f"{fname:<12} {hits:>5} {cens:>5} {T:>6} {'--':>10} {'--':>10} {'--':>8}")
            continue

        print(f"{fname:<12} {hits:>5} {cens:>5} {T:>6} {p_cens:>10.6f} {a_valid:>10.6f} {(a_valid/p_cens):>8.4f}")


if __name__ == "__main__":
    main()
