"""
Coverage rate: the empirical fraction of actual outcomes falling inside a
predicted interval. Previously computed inline, identically, in both
Phase 7's and Phase 13's scripts (`((actual >= lower) & (actual <= upper)).mean()`)
-- duplicated and never unit-tested on its own. Extracted here as the
single source of truth per Phase 0's "do not duplicate functionality"
rule, found during Phase 19's test-coverage audit.
"""
from __future__ import annotations

import numpy as np


def coverage_rate(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if not (len(actual) == len(lower) == len(upper)):
        raise ValueError("actual, lower, upper must be the same length.")
    covered = (actual >= lower) & (actual <= upper)
    return float(covered.mean())
