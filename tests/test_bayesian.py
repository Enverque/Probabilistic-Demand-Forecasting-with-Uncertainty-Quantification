import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.bayesian_structural import BayesianStructuralModel  # noqa: E402


@pytest.fixture(scope="module")
def small_fitted_model():
    # Deliberately tiny: this test exists to check the model BUILDS,
    # SAMPLES, and FORECASTS correctly end-to-end -- not to check
    # forecast accuracy (which needs the real scale demonstrated in
    # scripts/phase9_bayesian.py). Keeping n/draws/tune small keeps this
    # test fast enough to run in the normal test suite.
    rng = np.random.default_rng(0)
    n = 60
    y = rng.poisson(10, n).astype(float)
    model = BayesianStructuralModel()
    model.build_model(y, use_weekly=True, use_monthly=False)
    model.fit(draws=50, tune=50, chains=1, target_accept=0.8)
    return model


def test_model_builds_with_expected_variables(small_fitted_model):
    var_names = small_fitted_model.trace.posterior.data_vars.keys()
    assert "trend" in var_names
    assert "beta_cos_weekly" in var_names
    assert "alpha" in var_names


def test_forecast_produces_correct_shape(small_fitted_model):
    result = small_fitted_model.forecast(steps=10, num_samples=50)
    assert result["forecast_samples"].shape == (50, 10)
    assert len(result["forecast_mean"]) == 10
    assert len(result["lower_95"]) == 10


def test_forecast_intervals_are_ordered(small_fitted_model):
    result = small_fitted_model.forecast(steps=10, num_samples=50)
    assert (result["lower_95"] <= result["lower_50"]).all()
    assert (result["lower_50"] <= result["upper_50"]).all()
    assert (result["upper_50"] <= result["upper_95"]).all()


def test_forecast_samples_are_non_negative_counts(small_fitted_model):
    result = small_fitted_model.forecast(steps=10, num_samples=50)
    assert (result["forecast_samples"] >= 0).all()
    assert np.allclose(result["forecast_samples"], np.round(result["forecast_samples"]))


def test_raises_if_forecast_called_before_fit():
    model = BayesianStructuralModel()
    rng = np.random.default_rng(1)
    model.build_model(rng.poisson(5, 30).astype(float))
    with pytest.raises(ValueError):
        model.forecast(steps=5)
