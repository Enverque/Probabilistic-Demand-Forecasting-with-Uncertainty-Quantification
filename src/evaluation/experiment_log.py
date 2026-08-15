"""
Lightweight experiment log. NOT MLflow -- Phase 0 explicitly said not to
add MLflow "simply for resume keywords," and nothing about this project's
scale (single-machine, ~20 experiments total across all phases) justifies
the operational overhead of a tracking server. A single append-only CSV
is proportionate and sufficient.

Two uses:
1. `log_experiment()` -- available for any future phase script to call
   directly after a real run, appending a row with real measured values.
2. `data/interim/experiment_log.csv` (built by scripts/phase21_experiment_log.py)
   -- a RETROSPECTIVE consolidation of the real numbers already reported in
   Phases 5-17's own script output. This is transcription of real,
   already-computed values with their source phase cited, not new
   experimentation -- Phase 0's "do not invent numbers" rule applies here
   exactly as it did when those numbers were first reported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

EXPERIMENT_LOG_COLUMNS = [
    "phase", "model", "features", "forecast_horizon", "backtest_windows",
    "series_scope", "metric_name", "metric_value", "runtime_notes", "notes",
]


def log_experiment(record: dict, path: str) -> None:
    """Append one experiment record. Creates the file with a header if it
    doesn't exist yet. Validates the record has exactly the expected
    columns -- an experiment log with silently-inconsistent columns across
    rows would be worse than no log at all."""
    missing = set(EXPERIMENT_LOG_COLUMNS) - set(record.keys())
    extra = set(record.keys()) - set(EXPERIMENT_LOG_COLUMNS)
    if missing or extra:
        raise ValueError(f"Record must have exactly {EXPERIMENT_LOG_COLUMNS}. "
                          f"Missing: {missing}, unexpected: {extra}")
    path = Path(path)
    df = pd.DataFrame([record])[EXPERIMENT_LOG_COLUMNS]
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, mode="w", header=True, index=False)


def get_experiment_log(path: str) -> Optional[pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)
