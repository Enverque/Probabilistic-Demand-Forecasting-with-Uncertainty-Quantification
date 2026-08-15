# Phase 1 — Data Understanding: decisions and changes

## Source of the data
Kaggle and Google Drive (where M5 is officially hosted) are not reachable
from this project's network environment. Traced the loader path used by
the `datasetsforecast` PyPI package and found it pulls the real M5 files
from a public GitHub mirror (`Nixtla/m5-forecasts`), which is reachable.
Downloaded from there and independently verified row counts, hierarchy
cardinalities, and WRMSSE weight-file structure against the well-documented
public M5 statistics before trusting the source.

## sales_train_evaluation.csv vs. sales_train_validation.csv
Uses `sales_train_evaluation.csv` (full history through d_1941), not
`sales_train_validation.csv` (through d_1913). This is deliberate: outside
the live Kaggle competition, the evaluation file's later days aren't a
hidden leaderboard answer — they're simply the most recent 28 days of real,
released history. This project's own rolling-origin backtest (Phase 4)
carves its own train/test cutoffs out of this full history, so using the
longer file gives more legitimate backtest windows, not extra information
at prediction time.

## Two real schema deviations found in the GitHub mirror (vs. raw Kaggle files)
1. `sales_train_evaluation.csv` has no `id` column. Reconstructed as
   `item_id + "_" + store_id`.
2. `calendar.csv` has no `d` column. Reconstructed from row position, only
   after verifying the calendar rows are already in chronological order
   (`d_1` = first row).

Both fixes are in `M5DataLoader.load()`, not silently worked around.
