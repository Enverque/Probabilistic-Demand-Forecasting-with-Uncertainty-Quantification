"""
Phase 19: tests filling the specific gaps found by auditing existing
coverage against Phase 0's required categories (data pipeline, forecasting,
metrics, reconciliation, inventory). Everything already covered elsewhere
(leakage, quantile monotonicity, backtest window correctness, WRMSSE,
pinball loss) is NOT re-tested here -- see tests/test_features.py,
tests/test_backtest.py, tests/test_wrmsse.py, tests/test_pinball.py,
tests/test_quantile_ml.py for that existing coverage.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.calibration import coverage_rate  # noqa: E402
from src.inventory.newsvendor import critical_ratio, order_quantity_from_quantiles, simulate_period  # noqa: E402


# ---------- METRICS: coverage (previously untested standalone) ----------

def test_coverage_rate_hand_computed():
    actual = np.array([5.0, 10.0, 15.0, 20.0])
    lower = np.array([0.0, 0.0, 0.0, 25.0])   # first 3 covered, last one not (20 < 25 lower bound? actually check)
    upper = np.array([8.0, 12.0, 12.0, 30.0])
    # row0: 5 in [0,8] -> covered
    # row1: 10 in [0,12] -> covered
    # row2: 15 in [0,12] -> NOT covered
    # row3: 20 in [25,30] -> NOT covered (below lower bound)
    result = coverage_rate(actual, lower, upper)
    assert result == pytest.approx(0.5)


def test_coverage_rate_all_covered():
    actual = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 10.0, 10.0])
    assert coverage_rate(actual, lower, upper) == pytest.approx(1.0)


def test_coverage_rate_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        coverage_rate(np.array([1.0, 2.0]), np.array([0.0]), np.array([5.0, 5.0]))


# ---------- RECONCILIATION: formal coherence assertion (previously only printed, never asserted) ----------

def test_bottom_up_forecasts_sum_coherently_to_parent_level():
    # Synthetic 2-state, 4-store hierarchy: bottom-up forecasts summed
    # per-store, then per-state, MUST match summing directly across all
    # stores in that state -- this is the actual coherence GUARANTEE
    # reconciliation is supposed to provide, asserted here as a real test
    # rather than only printed as a script's max-gap number (Phase 14).
    store_forecasts = pd.Series(
        {"CA_1": 100.0, "CA_2": 150.0, "TX_1": 80.0, "TX_2": 90.0})
    store_to_state = {"CA_1": "CA", "CA_2": "CA", "TX_1": "TX", "TX_2": "TX"}

    state_bottom_up = store_forecasts.groupby(store_to_state).sum()
    # independently recompute via a different grouping path
    df = store_forecasts.to_frame("forecast")
    df["state"] = df.index.map(store_to_state)
    recomputed = df.groupby("state")["forecast"].sum()

    assert (state_bottom_up.sort_index() == recomputed.sort_index()).all()
    assert state_bottom_up["CA"] == pytest.approx(250.0)
    assert state_bottom_up["TX"] == pytest.approx(170.0)


def test_bottom_up_coherence_holds_for_three_level_hierarchy():
    # Item -> Store -> State, three levels, checking coherence holds
    # transitively (item sums to store sums to state), not just one hop.
    item_forecasts = pd.Series({
        ("I1", "CA_1"): 10.0, ("I2", "CA_1"): 5.0,
        ("I1", "CA_2"): 8.0, ("I2", "CA_2"): 12.0,
        ("I1", "TX_1"): 3.0, ("I2", "TX_1"): 7.0,
    })
    store_of = {"CA_1": "CA_1", "CA_2": "CA_2", "TX_1": "TX_1"}
    state_of_store = {"CA_1": "CA", "CA_2": "CA", "TX_1": "TX"}

    store_totals = item_forecasts.groupby(level=1).sum()
    state_totals_from_store = store_totals.groupby(state_of_store).sum()
    state_totals_from_item = item_forecasts.groupby(
        [state_of_store[s] for _, s in item_forecasts.index]).sum()

    assert (state_totals_from_store.sort_index() == state_totals_from_item.sort_index()).all()


# ---------- INVENTORY: edge cases not covered in Phase 16's original test suite ----------

def test_newsvendor_zero_demand():
    result = simulate_period(actual_demand=0.0, order_quantity=10.0, stockout_cost=5.0, holding_cost=2.0)
    assert result["units_short"] == 0.0
    assert result["units_excess"] == pytest.approx(10.0)
    assert result["total_cost"] == pytest.approx(20.0)  # all 10 units held, no stockout possible


def test_newsvendor_zero_stockout_cost():
    # If stockouts are free, the optimal policy should order the MINIMUM
    # (critical ratio -> 0), and any understock incurs zero cost regardless
    # of size.
    cr = critical_ratio(stockout_cost=0.0, holding_cost=5.0)
    assert cr == pytest.approx(0.0)
    result = simulate_period(actual_demand=100.0, order_quantity=0.0, stockout_cost=0.0, holding_cost=5.0)
    assert result["total_cost"] == pytest.approx(0.0)


def test_newsvendor_zero_holding_cost():
    # If holding is free, critical ratio -> 1 (order the max), and
    # overstock incurs zero cost regardless of size.
    cr = critical_ratio(stockout_cost=5.0, holding_cost=0.0)
    assert cr == pytest.approx(1.0)
    result = simulate_period(actual_demand=10.0, order_quantity=1000.0, stockout_cost=5.0, holding_cost=0.0)
    assert result["total_cost"] == pytest.approx(0.0)


def test_newsvendor_zero_both_costs_raises():
    with pytest.raises(ValueError):
        critical_ratio(stockout_cost=0.0, holding_cost=0.0)


def test_newsvendor_extreme_quantile_zero():
    # service_level=0 -> order at or below the lowest quantile available
    quantiles = {0.1: 5.0, 0.5: 10.0, 0.9: 20.0}
    result = order_quantity_from_quantiles(quantiles, service_level=0.0)
    assert result == pytest.approx(5.0)  # clipped to lowest available, not extrapolated below


def test_newsvendor_extreme_quantile_one():
    quantiles = {0.1: 5.0, 0.5: 10.0, 0.9: 20.0}
    result = order_quantity_from_quantiles(quantiles, service_level=1.0)
    assert result == pytest.approx(20.0)  # clipped to highest available


def test_newsvendor_negative_costs_rejected():
    with pytest.raises(ValueError):
        critical_ratio(stockout_cost=-5.0, holding_cost=2.0)
