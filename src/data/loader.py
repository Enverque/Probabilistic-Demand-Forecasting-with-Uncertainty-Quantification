"""
M5 data loading and schema validation.

Refactored from the `M5DataLoader` prototype in
`sales-forecast-probabilistic-eda-ml.ipynb`. Changes from that prototype
are documented in `docs/phase_notes/phase1.md` — summary:

- Hardcoded Kaggle paths (`/kaggle/input/...`) replaced with a configurable
  `data_dir`, since this project runs outside Kaggle.
- Loads `sales_train_evaluation.csv` (full history through d_1941) rather
  than `sales_train_validation.csv` (through d_1913). This is a deliberate
  choice, not an oversight: outside the live competition, the evaluation
  file's later days are not a hidden leaderboard answer, they are simply
  the most recent 28 days of real, released history. Our own rolling-origin
  backtest (Phase 4) carves *its own* train/test cutoffs out of this full
  history, so using the longer file gives us more legitimate backtest
  windows, not extra information at prediction time. Anyone using the
  shorter file, or blending in `sales_test_evaluation.csv`, would be doing
  so incorrectly for this project's purpose.
- Added `sample_submission.csv` handling removed (not used — we are not
  submitting to Kaggle).
- Added explicit schema validation (`validate_schema`) — the prototype
  assumed column names/shapes; this version checks them and fails loudly.
- Removed unused Bayesian-analysis-oriented methods (`get_sample_series`
  stratified sampling, `save_processed_data` to parquet-per-hierarchy) from
  this module; sampling and hierarchy construction belong to Phase 2/3 and
  will be built there deliberately rather than inherited.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Expected schema, established in Phase 0 and checked against the real
# files during Phase 1 exploration (see docs/phase_notes/phase1.md).
EXPECTED_N_SERIES = 30_490
EXPECTED_N_ITEMS = 3_049
EXPECTED_N_STORES = 10
EXPECTED_N_STATES = 3
EXPECTED_N_DEPTS = 7
EXPECTED_N_CATS = 3
EXPECTED_N_DAYS_EVALUATION = 1_941  # d_1 .. d_1941 in sales_train_evaluation.csv
HIERARCHY_ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


@dataclass
class M5Data:
    """Container for the three core M5 tables plus the WRMSSE weights."""
    sales: pd.DataFrame          # wide: one row per series, one col per d_
    calendar: pd.DataFrame       # one row per date, d_ <-> date mapping
    prices: pd.DataFrame         # long: store_id, item_id, wm_yr_wk, sell_price
    weights: pd.DataFrame        # WRMSSE weights per hierarchy aggregation level
    day_cols: List[str] = field(default_factory=list)


class M5DataLoader:
    """Loads and schema-validates the M5 competition files from local disk."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)

    def load(self, verbose: bool = True) -> M5Data:
        sales_path = self.data_dir / "sales_train_evaluation.csv"
        calendar_path = self.data_dir / "calendar.csv"
        prices_path = self.data_dir / "sell_prices.csv"
        weights_path = self.data_dir / "weights_evaluation.csv"

        for p in (sales_path, calendar_path, prices_path, weights_path):
            if not p.exists():
                raise FileNotFoundError(f"Expected M5 file not found: {p}")

        sales = pd.read_csv(sales_path)
        calendar = pd.read_csv(calendar_path, parse_dates=["date"])
        prices = pd.read_csv(prices_path)
        weights = pd.read_csv(weights_path)

        # This mirror's sales file has no `id` column and this calendar
        # file has no `d` column — both present in the raw Kaggle download
        # but apparently dropped by this GitHub mirror. Verified (not
        # assumed) by inspecting `sales.columns`/`calendar.columns` before
        # writing this: see docs/phase_notes/phase1.md. Reconstructed here
        # rather than silently working around their absence, because
        # `id` and `d_` are referenced throughout the rest of this project
        # and every other public M5 notebook.
        if "id" not in sales.columns:
            sales.insert(0, "id", sales["item_id"] + "_" + sales["store_id"])
        if "d" not in calendar.columns:
            if not calendar["date"].is_monotonic_increasing:
                raise ValueError(
                    "calendar.csv has no 'd' column and is not already in "
                    "chronological order — cannot safely reconstruct d_1..d_n "
                    "by row position."
                )
            calendar["d"] = ["d_" + str(i) for i in range(1, len(calendar) + 1)]

        day_cols = [c for c in sales.columns if c.startswith("d_")]
        # d_ columns are not guaranteed to already be in day order after
        # a plain string sort (d_10 < d_2 lexicographically), so sort by
        # the integer suffix explicitly. Getting this wrong would silently
        # scramble the time axis for every downstream feature.
        day_cols = sorted(day_cols, key=lambda c: int(c.split("_")[1]))

        if verbose:
            print(f"sales:    {sales.shape}")
            print(f"calendar: {calendar.shape}")
            print(f"prices:   {prices.shape}")
            print(f"weights:  {weights.shape}")
            print(f"day columns: {day_cols[0]} .. {day_cols[-1]} ({len(day_cols)} days)")

        return M5Data(sales=sales, calendar=calendar, prices=prices,
                      weights=weights, day_cols=day_cols)

    @staticmethod
    def validate_schema(data: M5Data) -> Dict[str, dict]:
        """
        Check the loaded data against the schema assumed in Phase 0.

        Returns a dict of check_name -> {"passed": bool, "detail": ...}
        rather than raising, so a caller (or test) can report every
        mismatch at once instead of stopping at the first one.
        """
        results: Dict[str, dict] = {}

        def check(name, passed, detail):
            results[name] = {"passed": bool(passed), "detail": detail}

        sales = data.sales

        check(
            "hierarchy_columns_present",
            all(c in sales.columns for c in HIERARCHY_ID_COLS),
            HIERARCHY_ID_COLS,
        )
        check("n_series", sales.shape[0] == EXPECTED_N_SERIES,
              f"got {sales.shape[0]}, expected {EXPECTED_N_SERIES}")
        check("n_items", sales["item_id"].nunique() == EXPECTED_N_ITEMS,
              f"got {sales['item_id'].nunique()}, expected {EXPECTED_N_ITEMS}")
        check("n_stores", sales["store_id"].nunique() == EXPECTED_N_STORES,
              f"got {sales['store_id'].nunique()}, expected {EXPECTED_N_STORES}")
        check("n_states", sales["state_id"].nunique() == EXPECTED_N_STATES,
              f"got {sales['state_id'].nunique()}, expected {EXPECTED_N_STATES}")
        check("n_departments", sales["dept_id"].nunique() == EXPECTED_N_DEPTS,
              f"got {sales['dept_id'].nunique()}, expected {EXPECTED_N_DEPTS}")
        check("n_categories", sales["cat_id"].nunique() == EXPECTED_N_CATS,
              f"got {sales['cat_id'].nunique()}, expected {EXPECTED_N_CATS}")
        check("n_days", len(data.day_cols) == EXPECTED_N_DAYS_EVALUATION,
              f"got {len(data.day_cols)}, expected {EXPECTED_N_DAYS_EVALUATION}")

        # Every item-store combination should appear exactly once.
        dup_ids = sales["id"].duplicated().sum()
        check("no_duplicate_series_ids", dup_ids == 0, f"{dup_ids} duplicate ids")

        # Calendar must cover at least as many days as the sales history,
        # contiguously, with no gaps — a gap here would silently misalign
        # every date-based feature (day-of-week, event flags, prices) from
        # the sales columns they're supposed to describe.
        cal_dates = data.calendar["date"].sort_values()
        expected_range = pd.date_range(cal_dates.iloc[0], cal_dates.iloc[-1], freq="D")
        check("calendar_no_date_gaps", len(cal_dates) == len(expected_range)
              and (cal_dates.values == expected_range.values).all(),
              f"{len(cal_dates)} calendar rows, {len(expected_range)} expected for a gap-free range")
        check("calendar_covers_sales_history", len(data.calendar) >= len(data.day_cols),
              f"{len(data.calendar)} calendar rows vs {len(data.day_cols)} day columns")

        # Sales values should be non-negative integers (unit counts) —
        # negative sales or non-integer values would indicate a parsing
        # problem or, if real, returns netted into sales that we'd want to
        # know about before modeling.
        vals = sales[data.day_cols].to_numpy()
        check("sales_non_negative", (vals >= 0).all(), f"min value = {vals.min()}")
        check("sales_no_nulls", not sales[data.day_cols].isnull().any().any(),
              "null check across all day columns")

        # Prices should be positive where present (missing price = item not
        # yet sold at that store/week, which is legitimate, not an error).
        check("prices_positive_where_present",
              (data.prices["sell_price"].dropna() > 0).all(),
              "min positive-check on non-null sell_price")

        # WRMSSE weights are defined per aggregation level and should each
        # sum to 1.0 within a level (they're a within-level weighting
        # scheme, not a global one) — this is exactly the kind of thing
        # Phase 11 says "validate a library/derived metric against a hand
        # example" and this is the cheapest place to catch a
        # misunderstanding of the weights file.
        per_level_sums = data.weights.groupby("Level_id")["weight"].sum()
        bad_levels = per_level_sums[(per_level_sums - 1.0).abs() > 1e-6]
        check("weights_sum_to_one_per_level", len(bad_levels) == 0,
              bad_levels.to_dict() if len(bad_levels) else "all 12 levels sum to 1.0")

        return results
