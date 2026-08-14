"""
Newsvendor inventory model.

The classical result: for a single-period ordering decision with
stockout cost c_u (cost per unit of unmet demand) and holding cost c_o
(cost per unit of unsold inventory), the cost-minimizing order quantity
is the quantile of the demand distribution at the CRITICAL RATIO:

    critical_ratio = c_u / (c_u + c_o)
    Q* = F^{-1}(critical_ratio)

Intuition: if stockouts are much more expensive than holding excess
stock (c_u >> c_o), the critical ratio approaches 1 and you should order
near the top of the demand distribution; if holding is expensive relative
to stockouts, order near the bottom.

This module deliberately keeps the point-forecast policy and the
probabilistic policy on equal footing: POINT policy orders exactly the
point forecast regardless of the cost asymmetry (the naive thing an
inventory planner using only a point forecast would do), PROBABILISTIC
policy orders the quantile matching the true critical ratio -- the
comparison in scripts/phase16_inventory.py isolates the value of having a
predictive distribution at all, not a difference in cost assumptions.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def critical_ratio(stockout_cost: float, holding_cost: float) -> float:
    if stockout_cost < 0 or holding_cost < 0:
        raise ValueError("Costs must be non-negative.")
    if stockout_cost + holding_cost == 0:
        raise ValueError("At least one of stockout_cost/holding_cost must be positive.")
    return stockout_cost / (stockout_cost + holding_cost)


def order_quantity_from_quantiles(quantile_preds: Dict[float, float], service_level: float) -> float:
    """
    Given a discrete set of quantile predictions (e.g. {0.1: 2.0, 0.5:
    5.0, 0.9: 12.0}), returns the order quantity at `service_level` via
    linear interpolation between the two nearest available quantiles.
    Clips to the min/max quantile available rather than extrapolating
    beyond it -- extrapolating a tail we never actually estimated would
    overstate confidence in a region the model was never evaluated on.
    """
    qs = sorted(quantile_preds.keys())
    vals = [quantile_preds[q] for q in qs]
    if service_level <= qs[0]:
        return float(vals[0])
    if service_level >= qs[-1]:
        return float(vals[-1])
    return float(np.interp(service_level, qs, vals))


def simulate_period(actual_demand: float, order_quantity: float,
                     stockout_cost: float, holding_cost: float) -> Dict[str, float]:
    shortfall = max(0.0, actual_demand - order_quantity)
    surplus = max(0.0, order_quantity - actual_demand)
    return {
        "order_quantity": order_quantity,
        "actual_demand": actual_demand,
        "units_short": shortfall,
        "units_excess": surplus,
        "stockout_cost_incurred": shortfall * stockout_cost,
        "holding_cost_incurred": surplus * holding_cost,
        "total_cost": shortfall * stockout_cost + surplus * holding_cost,
        "stockout_occurred": shortfall > 0,
        "demand_fully_met": shortfall == 0,
    }
