"""
Analysis module for L-SHADE hazard paper.

Provides:
- Kaplan-Meier survival analysis
- Witness frequency (gamma_t) estimation
- Gap analysis (theory vs practice)
- Table and figure generation
"""

from .km_analysis import (
    compute_hitting_times,
    kaplan_meier,
    compute_km_statistics,
)

from .witness_frequency import (
    check_L2,
    check_L3,
    compute_witness_indicators,
    estimate_gamma,
)

__all__ = [
    'compute_hitting_times',
    'kaplan_meier', 
    'compute_km_statistics',
    'check_L2',
    'check_L3',
    'compute_witness_indicators',
    'estimate_gamma',
]
