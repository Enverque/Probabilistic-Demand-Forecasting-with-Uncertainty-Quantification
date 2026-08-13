"""
Bayesian structural time series model with Negative Binomial likelihood.

Reused from `ImprovedBayesianTimeSeries` in
`sales-forecast-probabilistic-eda-ml.ipynb` -- the model structure itself
(random-walk trend + Fourier weekly/monthly seasonality + NB likelihood,
data standardized then affine-transformed back for the likelihood mean) is
kept faithful to the original, since Phase 0 asked me to evaluate this as
a candidate, not redesign it. One design choice worth flagging rather than
silently inheriting: the linear predictor is combined ADDITIVELY on the
standardized scale and then affine-transformed to the NB mean (clipped at
0.1), rather than passed through a log link the way count-data GLMs
usually are. That's a real modeling choice with a real cost (gradients
near the 0.1 clipping boundary can be poorly behaved, likely contributing
to the sampling difficulty noted in Phase 9's report) -- reused as-is here
rather than silently "fixed," since changing it would no longer be
evaluating the prototype's actual model.

Sampling defaults are reduced from the prototype's (draws=2000, tune=1000,
chains=4) to (draws=500, tune=500, chains=2) -- justified empirically in
scripts/phase9_bayesian.py's feasibility timing test, not picked
arbitrarily: this container has exactly 1 CPU core (chains run
sequentially regardless of the `chains` setting), and the full default
settings extrapolated to ~10 minutes for a SINGLE series with a
simplified likelihood in the feasibility test -- workable for one
demonstration fit, not for a systematic multi-series/multi-window
backtest, which is the central finding of this phase.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt


class BayesianStructuralModel:
    def __init__(self, name: str = "BSTS"):
        self.name = name
        self.model = None
        self.trace = None
        self.y_mean = None
        self.y_std = None
        self.n_obs = None

    def build_model(self, y: np.ndarray, use_weekly: bool = True, use_monthly: bool = True) -> pm.Model:
        self.n_obs = len(y)
        t = np.arange(self.n_obs)
        self.y_mean, self.y_std = float(np.mean(y)), float(np.std(y))

        with pm.Model() as model:
            sigma_trend = pm.HalfNormal("sigma_trend", sigma=0.1)
            level_init = pm.Normal("level_init", mu=0, sigma=1)
            trend_innovations = pm.Normal("trend_innovations", mu=0, sigma=sigma_trend, shape=self.n_obs)
            trend = pm.Deterministic("trend", level_init + pt.cumsum(trend_innovations))

            weekly = 0
            if use_weekly:
                n_f, period = 3, 7.0
                bc = pm.Normal("beta_cos_weekly", mu=0, sigma=0.5, shape=n_f)
                bs = pm.Normal("beta_sin_weekly", mu=0, sigma=0.5, shape=n_f)
                cos_feat = np.column_stack([np.cos(2 * np.pi * (i + 1) * t / period) for i in range(n_f)])
                sin_feat = np.column_stack([np.sin(2 * np.pi * (i + 1) * t / period) for i in range(n_f)])
                weekly = pm.Deterministic("weekly", pt.dot(cos_feat, bc) + pt.dot(sin_feat, bs))

            monthly = 0
            if use_monthly:
                n_f, period = 4, 30.5
                bc = pm.Normal("beta_cos_monthly", mu=0, sigma=0.5, shape=n_f)
                bs = pm.Normal("beta_sin_monthly", mu=0, sigma=0.5, shape=n_f)
                cos_feat = np.column_stack([np.cos(2 * np.pi * (i + 1) * t / period) for i in range(n_f)])
                sin_feat = np.column_stack([np.sin(2 * np.pi * (i + 1) * t / period) for i in range(n_f)])
                monthly = pm.Deterministic("monthly", pt.dot(cos_feat, bc) + pt.dot(sin_feat, bs))

            mu_scaled = pm.Deterministic("mu_scaled", trend + weekly + monthly)
            mu_original = pm.Deterministic("mu", mu_scaled * self.y_std + self.y_mean)

            alpha = pm.Gamma("alpha", alpha=2, beta=1)
            pm.NegativeBinomial("obs", mu=pm.math.maximum(mu_original, 0.1), alpha=alpha, observed=y)

        self.model = model
        return model

    def fit(self, draws: int = 500, tune: int = 500, chains: int = 2,
            target_accept: float = 0.9, random_seed: int = 42):
        if self.model is None:
            raise ValueError("Call build_model() first.")
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                                    target_accept=target_accept, random_seed=random_seed,
                                    progressbar=False, return_inferencedata=True)
        return self.trace

    def forecast(self, steps: int, num_samples: int = 300, random_seed: int = 42) -> Dict[str, np.ndarray]:
        if self.trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        rng = np.random.default_rng(random_seed)
        posterior = self.trace.posterior
        n_chains, n_draws = posterior.chain.size, posterior.draw.size
        chain_idx = rng.integers(0, n_chains, num_samples)
        draw_idx = rng.integers(0, n_draws, num_samples)
        t_future = np.arange(self.n_obs, self.n_obs + steps)

        has_weekly = "beta_cos_weekly" in posterior
        has_monthly = "beta_cos_monthly" in posterior

        samples = np.zeros((num_samples, steps))
        for i in range(num_samples):
            c, d = int(chain_idx[i]), int(draw_idx[i])
            last_trend = float(posterior["trend"].isel(chain=c, draw=d).values[-1])
            sigma_trend = float(posterior["sigma_trend"].isel(chain=c, draw=d).values)
            trend_future = last_trend + np.cumsum(rng.normal(0, sigma_trend, steps))

            weekly_future = np.zeros(steps)
            if has_weekly:
                bc = posterior["beta_cos_weekly"].isel(chain=c, draw=d).values
                bs = posterior["beta_sin_weekly"].isel(chain=c, draw=d).values
                for k in range(len(bc)):
                    weekly_future += (bc[k] * np.cos(2 * np.pi * (k + 1) * t_future / 7.0)
                                       + bs[k] * np.sin(2 * np.pi * (k + 1) * t_future / 7.0))

            monthly_future = np.zeros(steps)
            if has_monthly:
                bc = posterior["beta_cos_monthly"].isel(chain=c, draw=d).values
                bs = posterior["beta_sin_monthly"].isel(chain=c, draw=d).values
                for k in range(len(bc)):
                    monthly_future += (bc[k] * np.cos(2 * np.pi * (k + 1) * t_future / 30.5)
                                        + bs[k] * np.sin(2 * np.pi * (k + 1) * t_future / 30.5))

            mu_scaled_future = trend_future + weekly_future + monthly_future
            mu_future = np.maximum(mu_scaled_future * self.y_std + self.y_mean, 0.1)
            alpha = float(posterior["alpha"].isel(chain=c, draw=d).values)
            p = alpha / (alpha + mu_future)
            samples[i] = rng.negative_binomial(alpha, p)

        return {
            "forecast_samples": samples,
            "forecast_mean": samples.mean(axis=0),
            "forecast_median": np.median(samples, axis=0),
            "lower_95": np.percentile(samples, 2.5, axis=0),
            "upper_95": np.percentile(samples, 97.5, axis=0),
            "lower_50": np.percentile(samples, 25, axis=0),
            "upper_50": np.percentile(samples, 75, axis=0),
        }
