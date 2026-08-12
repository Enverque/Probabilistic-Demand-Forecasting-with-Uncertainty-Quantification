"""
Phase 2: Hierarchy.

Builds the 12 official M5 aggregation levels, validates their series
counts against the WRMSSE weights file, and demonstrates — with a real
naive forecast, not a toy example — that independently forecasting Store
and State level does not produce coherent totals.

Run: python scripts/phase2_hierarchy.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.data.hierarchy import build_hierarchy, validate_against_weights, demonstrate_incoherence  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def seasonal_naive_forecast(history: np.ndarray, season: int = 7) -> float:
    """
    Forecast = MEDIAN of the same weekday over the last 4 occurrences.

    Median, not mean, is the point: mean is a linear operator, and summing
    stores into a state is also linear, so "sum of per-store means" and
    "mean of the state's summed history" are mathematically identical —
    a mean-based naive forecast is coherent by construction and would
    make this demonstration vacuous (confirmed empirically first: see
    docs/phase_notes/phase2.md for the zero-gap mean run). Median breaks
    that identity, which is representative of what happens with any
    realistic model (tree-based, neural, or anything nonlinear/clipped) —
    exactly the kind of models used from Phase 6 onward.
    """
    return float(np.median(history[-season::-season][:4]))


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    hierarchy = build_hierarchy(data.sales, data.day_cols)

    print("--- Level series counts vs. official weights file ---")
    results = validate_against_weights(hierarchy, data.weights)
    for name, r in results.items():
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {name}: {r['detail']}")

    print("\n--- Coherence demonstration: independent Store vs. State naive forecasts ---")
    comparison = demonstrate_incoherence(data.sales, data.day_cols, seasonal_naive_forecast)
    print(comparison.round(2).to_string())
    print(f"\nMean |gap| across states: {comparison['gap'].abs().mean():.1f} units")
    print(f"Mean |gap %|:            {comparison['gap_pct'].abs().mean():.2f}%")
    print(f"Max |gap %|:             {comparison['gap_pct'].abs().max():.2f}% "
          f"({comparison['gap_pct'].abs().idxmax()})")


if __name__ == "__main__":
    main()
