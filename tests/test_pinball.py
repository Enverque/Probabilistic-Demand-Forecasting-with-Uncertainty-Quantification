import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.pinball import pinball_loss, mean_pinball_across_quantiles, scaled_pinball_loss  # noqa: E402


def test_pinball_loss_hand_computed_underprediction():
    # y=10, yhat=8 (underprediction, diff=y-yhat=2>=0), tau=0.9
    # loss = tau * diff = 0.9 * 2 = 1.8
    result = pinball_loss(np.array([10.0]), np.array([8.0]), tau=0.9)
    assert result == pytest.approx(1.8)


def test_pinball_loss_hand_computed_overprediction():
    # y=10, yhat=15 (overprediction, diff=y-yhat=-5<0), tau=0.9
    # loss = (tau-1) * diff = (0.9-1)*(-5) = (-0.1)*(-5) = 0.5
    result = pinball_loss(np.array([10.0]), np.array([15.0]), tau=0.9)
    assert result == pytest.approx(0.5)


def test_pinball_loss_asymmetric_penalty_direction():
    # At tau=0.9 (P90), UNDER-prediction should be penalized MORE heavily
    # than an equal-sized OVER-prediction -- that's the entire point of an
    # asymmetric quantile loss, and this test checks the asymmetry has the
    # correct sign, not just that it's asymmetric.
    under_loss = pinball_loss(np.array([10.0]), np.array([8.0]), tau=0.9)   # under by 2
    over_loss = pinball_loss(np.array([10.0]), np.array([12.0]), tau=0.9)   # over by 2
    assert under_loss > over_loss


def test_pinball_loss_at_median_equals_half_mae():
    # A well-known identity: pinball loss at tau=0.5 equals half the MAE,
    # regardless of over/under direction -- a strong, independent check.
    y = np.array([10.0, 5.0, 20.0, 3.0])
    yhat = np.array([8.0, 9.0, 15.0, 3.0])
    result = pinball_loss(y, yhat, tau=0.5)
    mae = np.mean(np.abs(y - yhat))
    assert result == pytest.approx(mae / 2)


def test_pinball_loss_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y.copy(), tau=0.1) == pytest.approx(0.0)
    assert pinball_loss(y, y.copy(), tau=0.9) == pytest.approx(0.0)


def test_lower_tau_penalizes_overprediction_more():
    # Mirror check: at tau=0.1 (P10), OVER-prediction should be penalized
    # more than under-prediction -- the opposite asymmetry from tau=0.9.
    under_loss = pinball_loss(np.array([10.0]), np.array([8.0]), tau=0.1)
    over_loss = pinball_loss(np.array([10.0]), np.array([12.0]), tau=0.1)
    assert over_loss > under_loss


def test_mean_pinball_across_quantiles_is_simple_average():
    y = np.array([10.0, 10.0])
    preds = {0.1: np.array([8.0, 8.0]), 0.9: np.array([12.0, 12.0])}
    l1 = pinball_loss(y, preds[0.1], 0.1)
    l9 = pinball_loss(y, preds[0.9], 0.9)
    result = mean_pinball_across_quantiles(y, preds)
    assert result == pytest.approx((l1 + l9) / 2)


def test_scaled_pinball_none_when_scale_undefined():
    train = np.zeros(30)
    result = scaled_pinball_loss(np.array([1.0]), np.array([1.0]), 0.5, train)
    assert result is None


def test_scaled_pinball_hand_computed():
    # train = [0,3,4,5] -> scale = 1.0 (same hand example as Phase 11's
    # RMSSE test) -> sqrt(scale) = 1.0, so scaled pinball == raw pinball here.
    train = np.array([0, 3, 4, 5], dtype=float)
    raw = pinball_loss(np.array([10.0]), np.array([8.0]), tau=0.9)
    scaled = scaled_pinball_loss(np.array([10.0]), np.array([8.0]), 0.9, train)
    assert scaled == pytest.approx(raw)
