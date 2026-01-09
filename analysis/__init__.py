"""
Analysis utilities for the L-SHADE hazard / witness-regime paper.

This package is designed so that the individual scripts in this directory can be
run directly (e.g. `python analysis/combined_analysis.py ...`), but you can also
import the modules from Python.

Main entry points:
- analysis/km_analysis.py
- analysis/witness_frequency.py
- analysis/cluster_analysis.py
- analysis/combined_analysis.py
"""

from .km_analysis import (
    compute_hitting_times,
    kaplan_meier_discrete,
    compute_km_statistics,
    pooled_tail_hazard,
    pooled_tail_hazard_lcb,
    geometric_envelope_rate,
)

from .witness_frequency import (
    GAMMA_THRESHOLDS,
    check_L2,
    check_L3,
    compute_witness_indicators,
    estimate_gamma,
    estimate_gamma_with_ci,
    eta_r,
    compute_morse_a_t,
    compute_generic_a_t,
)

__all__ = [
    "compute_hitting_times",
    "kaplan_meier_discrete",
    "compute_km_statistics",
    "pooled_tail_hazard",
    "pooled_tail_hazard_lcb",
    "geometric_envelope_rate",
    "GAMMA_THRESHOLDS",
    "check_L2",
    "check_L3",
    "compute_witness_indicators",
    "estimate_gamma",
    "estimate_gamma_with_ci",
    "eta_r",
    "compute_morse_a_t",
    "compute_generic_a_t",
]
