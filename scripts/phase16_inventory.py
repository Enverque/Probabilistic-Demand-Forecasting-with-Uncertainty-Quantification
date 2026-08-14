"""
Phase 16: Inventory Decision Simulation.

Uses the same quantile predictions machinery as Phase 8/12/13 (600-series
sample, single window matching TRAIN_END/TEST_END used throughout).
Compares two ordering policies, day by day, series by series:

  POINT policy: order = P50 (what a planner using only a point forecast
    would naturally do, regardless of the true cost asymmetry).
  PROBABILISTIC policy: order = the quantile matching the TRUE critical
    ratio for the given cost scenario, interpolated from the full
    quantile set.

Run under two cost scenarios (stockout expensive vs. holding expensive)
so the comparison isn't cherry-picked to one asymmetry.

Run: python scripts/phase16_inventory.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.features.engineering import build_long_format, add_features, feature_columns  # noqa: E402
from src.forecasting.probabilistic import generate_probabilistic_forecast, QUANTILE_LEVELS  # noqa: E402
from src.inventory.newsvendor import critical_ratio, order_quantity_from_quantiles, simulate_period  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RANDOM_STATE = 42
ITEM_SAMPLE_PER_CAT = 200
HORIZON = 28
TRAIN_END = pd.Timestamp("2016-04-24")
TEST_END = pd.Timestamp("2016-05-22")

SCENARIOS = {
    "stockout_expensive (4:1)": {"stockout_cost": 4.0, "holding_cost": 1.0},
    "holding_expensive (1:3)": {"stockout_cost": 1.0, "holding_cost": 3.0},
}


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    sample = (data.sales.groupby("cat_id", group_keys=False)[data.sales.columns]
              .apply(lambda g: g.sample(min(len(g), ITEM_SAMPLE_PER_CAT), random_state=RANDOM_STATE)))
    long_df = build_long_format(sample, data.day_cols, data.calendar, data.prices)
    featured = add_features(long_df, horizon=HORIZON)
    numeric_cols = featured.select_dtypes(include=["float64"]).columns
    featured[numeric_cols] = featured[numeric_cols].astype("float32")
    cols = feature_columns(HORIZON)

    train = featured[featured["date"] <= TRAIN_END].dropna(subset=cols)
    test = featured[(featured["date"] > TRAIN_END) & (featured["date"] <= TEST_END)].dropna(subset=cols)

    print("Generating quantile predictions (reusing Phase 8's exact setup)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = generate_probabilistic_forecast(train, test, cols, "sales", random_state=RANDOM_STATE)
    test = test.reset_index(drop=True)
    preds = preds.reset_index(drop=True)

    for scenario_name, costs in SCENARIOS.items():
        cr = critical_ratio(**costs)
        print(f"\n=== Scenario: {scenario_name}  (critical ratio = {cr:.3f}) ===")

        point_results, prob_results = [], []
        for i in range(len(test)):
            actual = test.loc[i, "sales"]
            point_order = preds.loc[i, 0.5]
            quantile_dict = {q: preds.loc[i, q] for q in QUANTILE_LEVELS}
            prob_order = order_quantity_from_quantiles(quantile_dict, cr)

            point_results.append(simulate_period(actual, point_order, **costs))
            prob_results.append(simulate_period(actual, prob_order, **costs))

        point_df = pd.DataFrame(point_results)
        prob_df = pd.DataFrame(prob_results)

        print(f"{'metric':<30}{'POINT policy':>15}{'PROBABILISTIC policy':>22}")
        print(f"{'Total cost':<30}{point_df['total_cost'].sum():>15.1f}{prob_df['total_cost'].sum():>22.1f}")
        print(f"{'Mean cost/period':<30}{point_df['total_cost'].mean():>15.3f}{prob_df['total_cost'].mean():>22.3f}")
        print(f"{'Stockout rate':<30}{point_df['stockout_occurred'].mean()*100:>14.1f}%"
              f"{prob_df['stockout_occurred'].mean()*100:>21.1f}%")
        print(f"{'Service level (demand met)':<30}{point_df['demand_fully_met'].mean()*100:>14.1f}%"
              f"{prob_df['demand_fully_met'].mean()*100:>21.1f}%")
        print(f"{'Mean units short (stockouts)':<30}{point_df['units_short'].mean():>15.3f}"
              f"{prob_df['units_short'].mean():>22.3f}")
        print(f"{'Mean units excess':<30}{point_df['units_excess'].mean():>15.3f}"
              f"{prob_df['units_excess'].mean():>22.3f}")

        cost_reduction = (point_df["total_cost"].sum() - prob_df["total_cost"].sum()) / point_df["total_cost"].sum()
        print(f"\nCost reduction, probabilistic vs. point policy: {cost_reduction*100:+.1f}%")


if __name__ == "__main__":
    main()
