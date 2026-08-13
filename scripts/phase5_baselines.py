"""
Phase 5: Simple Baselines.

Runs naive_last_value, seasonal_naive, and moving_average through the
Phase 4 backtest harness on a representative sample of real series (reusing
the same category-stratified 2,100-series sample from Phase 3, for
consistency across phases and to keep melt/groupby cost reasonable), plus
the Total-level series.

Run: python scripts/phase5_baselines.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.data.hierarchy import build_hierarchy  # noqa: E402
from src.forecasting.backtest import RollingOriginBacktester  # noqa: E402
from src.models.naive import naive_last_value, seasonal_naive, moving_average  # noqa: E402
from src.evaluation.metrics import wmape, bias, fva  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RANDOM_STATE = 42
ITEM_SAMPLE_PER_CAT = 700


def to_long(sales_wide: pd.DataFrame, day_cols, calendar: pd.DataFrame) -> pd.DataFrame:
    long_df = sales_wide.melt(id_vars=["id"], value_vars=day_cols, var_name="d", value_name="sales")
    date_map = calendar.set_index("d")["date"]
    long_df["date"] = long_df["d"].map(date_map)
    return long_df[["id", "date", "sales"]]


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    hierarchy = build_hierarchy(data.sales, data.day_cols)
    cal = data.calendar.iloc[:len(data.day_cols)].reset_index(drop=True)

    sample = (data.sales.groupby("cat_id", group_keys=False)[data.sales.columns]
              .apply(lambda g: g.sample(min(len(g), ITEM_SAMPLE_PER_CAT), random_state=RANDOM_STATE)))
    print(f"Sample: {len(sample)} item-store series (same stratified sample as Phase 3)")

    long_df = to_long(sample[["id"] + data.day_cols], data.day_cols, cal)
    print(f"Long-format rows: {len(long_df)}")

    bt = RollingOriginBacktester(horizon_days=28, n_windows=8, min_train_days=730)

    metric_fns = {"wmape": wmape, "bias": bias}
    all_results = {}
    for name, fn in [("naive_last_value", naive_last_value),
                      ("seasonal_naive", seasonal_naive),
                      ("moving_average_7d", moving_average)]:
        results = bt.run(long_df, date_col="date", feature_cols=["id"], target_col="sales",
                          fit_predict_fn=fn, metric_fns=metric_fns)
        all_results[name] = results
        print(f"\n--- {name} ---")
        print(results[["window_id", "test_start", "test_end", "wmape", "bias"]].to_string(index=False))
        print(f"Mean WMAPE: {results['wmape'].mean()*100:.2f}%  "
              f"std: {results['wmape'].std()*100:.2f}pp  "
              f"Mean bias: {results['bias'].mean()*100:+.2f}%")

    print("\n--- Forecast Value Added vs. naive_last_value (the weakest baseline) ---")
    base_wmape = all_results["naive_last_value"]["wmape"].mean()
    for name in ["seasonal_naive", "moving_average_7d"]:
        model_wmape = all_results[name]["wmape"].mean()
        print(f"{name}: FVA = {fva(base_wmape, model_wmape)*100:+.1f}%")


if __name__ == "__main__":
    main()
