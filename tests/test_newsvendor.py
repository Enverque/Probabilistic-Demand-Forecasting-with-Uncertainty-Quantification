import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inventory.newsvendor import critical_ratio, order_quantity_from_quantiles, simulate_period  # noqa: E402


def test_critical_ratio_hand_computed():
    # stockout cost 3x holding cost -> critical ratio 3/(3+1) = 0.75
    assert critical_ratio(stockout_cost=3, holding_cost=1) == pytest.approx(0.75)


def test_critical_ratio_symmetric_costs_gives_median():
    assert critical_ratio(stockout_cost=1, holding_cost=1) == pytest.approx(0.5)


def test_critical_ratio_raises_on_negative_costs():
    with pytest.raises(ValueError):
        critical_ratio(stockout_cost=-1, holding_cost=1)


def test_order_quantity_exact_quantile_match():
    quantiles = {0.1: 2.0, 0.5: 5.0, 0.75: 8.0, 0.9: 12.0}
    assert order_quantity_from_quantiles(quantiles, service_level=0.75) == pytest.approx(8.0)


def test_order_quantity_interpolates_between_quantiles():
    # service_level=0.8 is exactly 1/3 of the way from 0.75 (val=8) to 0.9 (val=12)
    # -> 8 + (12-8)*((0.8-0.75)/(0.9-0.75)) = 8 + 4*(0.05/0.15) = 8 + 1.333... = 9.333...
    quantiles = {0.1: 2.0, 0.5: 5.0, 0.75: 8.0, 0.9: 12.0}
    result = order_quantity_from_quantiles(quantiles, service_level=0.8)
    assert result == pytest.approx(8 + 4 * (0.05 / 0.15))


def test_order_quantity_clips_rather_than_extrapolates():
    quantiles = {0.1: 2.0, 0.9: 12.0}
    assert order_quantity_from_quantiles(quantiles, service_level=0.99) == pytest.approx(12.0)
    assert order_quantity_from_quantiles(quantiles, service_level=0.01) == pytest.approx(2.0)


def test_simulate_period_understock_hand_computed():
    # demand=10, order=6 -> short by 4, stockout_cost=5 -> cost = 20
    result = simulate_period(actual_demand=10, order_quantity=6, stockout_cost=5, holding_cost=2)
    assert result["units_short"] == pytest.approx(4)
    assert result["units_excess"] == pytest.approx(0)
    assert result["total_cost"] == pytest.approx(20)
    assert result["stockout_occurred"] is True


def test_simulate_period_overstock_hand_computed():
    # demand=6, order=10 -> excess of 4, holding_cost=2 -> cost = 8
    result = simulate_period(actual_demand=6, order_quantity=10, stockout_cost=5, holding_cost=2)
    assert result["units_excess"] == pytest.approx(4)
    assert result["units_short"] == pytest.approx(0)
    assert result["total_cost"] == pytest.approx(8)
    assert result["stockout_occurred"] is False


def test_simulate_period_perfect_match_zero_cost():
    result = simulate_period(actual_demand=7, order_quantity=7, stockout_cost=5, holding_cost=2)
    assert result["total_cost"] == pytest.approx(0)
    assert result["demand_fully_met"] is True


def test_newsvendor_optimal_quantity_minimizes_expected_cost_empirically():
    # Simulate many demand draws from a KNOWN distribution, then check
    # that ordering at the theoretically-optimal critical-ratio quantile
    # achieves lower (or equal) average cost than ordering at a few
    # deliberately-wrong quantities -- an end-to-end sanity check that
    # the whole newsvendor logic (not just its pieces) behaves correctly.
    import numpy as np
    rng = np.random.default_rng(0)
    demand_samples = rng.poisson(20, 5000).astype(float)
    stockout_cost, holding_cost = 4.0, 1.0
    cr = critical_ratio(stockout_cost, holding_cost)
    optimal_q = float(np.quantile(demand_samples, cr))

    def avg_cost(order_q):
        costs = [simulate_period(d, order_q, stockout_cost, holding_cost)["total_cost"] for d in demand_samples]
        return np.mean(costs)

    cost_at_optimal = avg_cost(optimal_q)
    cost_too_low = avg_cost(optimal_q - 8)
    cost_too_high = avg_cost(optimal_q + 8)
    assert cost_at_optimal <= cost_too_low
    assert cost_at_optimal <= cost_too_high
