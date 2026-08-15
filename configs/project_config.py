"""
Canonical project constants.

Every phase script from Phase 3 onward defines its own local copies of
these same values (RANDOM_STATE=42, HORIZON=28, TRAIN_END/TEST_END, etc.)
rather than importing them from here. That's a deliberate, documented
tradeoff, not an oversight discovered too late to fix: each phase script
was built to be runnable and readable standalone (Phase 0's "each phase
should be independently understandable"), and retroactively rewiring 11
already-tested, already-committed scripts to import from a shared config
this late risks introducing a bug in working code for a purely cosmetic
DRY improvement. This file exists as the single documented REFERENCE for
what those values are and should be, verified (Phase 20) to be identical
across every script that defines them locally -- not as the module those
scripts actually import from today.

If this project continued past Phase 23, migrating scripts to import from
here would be a reasonable next refactor -- but it should happen with a
full re-run and diff of every script's output, not as a last-minute
Phase 20 change with unverified downstream effects.
"""

RANDOM_STATE = 42
HORIZON_DAYS = 28

# NOT uniform across the project -- verified by grep across scripts/, not
# assumed. Phases 3 and 5 use 700/category (2,100-series sample, before
# the OOM issue surfaced). Phase 7 onward uses 200/category (600-series
# sample), after Phase 7's first run at 700/category OOM-killed the
# container (documented in scripts/phase7_quantile_ml.py). An earlier
# version of this file claimed a single canonical value of 200, which was
# wrong for Phases 3/5 -- caught by actually grepping every script's
# constant rather than asserting the value from memory.
ITEM_SAMPLE_PER_CAT_PHASE_3_5 = 700
ITEM_SAMPLE_PER_CAT_PHASE_7_PLUS = 200

TRAIN_END = "2016-04-24"
TEST_END = "2016-05-22"

BACKTEST_N_WINDOWS_FULL = 8      # Phases 4, 5
BACKTEST_N_WINDOWS_ML = 3        # Phases 7, 12, 13 -- reduced for LightGBM training cost, documented per-phase
BACKTEST_MIN_TRAIN_DAYS = 730    # 2 years, justified in Phase 4 against Phase 3's yearly-seasonality finding

QUANTILE_LEVELS = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]  # Phase 8's full probabilistic output set
