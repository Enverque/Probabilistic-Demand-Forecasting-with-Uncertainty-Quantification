# Reproducibility (Phase 20)

## Setup
```bash
pip install -r requirements.txt --break-system-packages
# Only if rerunning Phase 9's Bayesian model investigation:
pip install -r requirements-optional.txt --break-system-packages
```

Data: this repo does not commit the raw M5 CSVs (`data/raw/` is gitignored,
~330MB). See `docs/phase_notes/phase1.md` for where they come from and why
(Kaggle/Google Drive aren't reachable from this project's environment;
sourced from a public GitHub mirror instead, and verified against known M5
statistics before trusting it).

## Running a phase
Every phase's deliverable is `scripts/phaseN_<name>.py`, runnable standalone:
```bash
python3 scripts/phase7_quantile_ml.py
```
Each script is self-contained (loads its own data, defines its own local
constants) by design -- see `configs/project_config.py` for why constants
are duplicated per-script rather than imported from a shared config, and
for the actual verified values (including the one real inconsistency found
during this audit: `ITEM_SAMPLE_PER_CAT` differs between Phases 3/5 and
Phase 7+, for a documented reason).

## Tests
```bash
python3 -m pytest tests/ -q
```
108 tests as of Phase 20, all passing. See `docs/phase_notes/` for the
handful of cases where a test initially failed and what that revealed
(most notably a cross-series data-leakage bug found while building Phase
10, fixed in `src/features/engineering.py`).

## Dashboard
```bash
streamlit run app/dashboard.py
```
Uses a fast approximate forecaster (`src/forecasting/dashboard_forecaster.py`),
NOT the validated LightGBM model from Phases 7-13 -- stated in the app
itself, not just here.

## Actual repo structure (as it stands, not the Phase 0 plan)
```
data/
  raw/            # gitignored -- 8 M5 CSVs, sourced per docs/phase_notes/phase1.md
  interim/        # gitignored -- generated summary reports
src/
  data/           # loader.py, hierarchy.py
  eda/            # distribution_analysis.py, temporal_patterns.py
  features/       # engineering.py (horizon-safe lag/rolling features)
  models/         # naive.py, sarima.py, quantile_ml.py, local_global.py, bayesian_structural.py
  forecasting/    # backtest.py, probabilistic.py, dashboard_forecaster.py
  evaluation/     # metrics.py, wrmsse.py, pinball.py, calibration.py
  inventory/      # newsvendor.py
scripts/          # phase1_...py through phase19_...py, one per phase deliverable
tests/            # one test file per major module, 108 tests total
app/              # dashboard.py (Streamlit)
configs/          # project_config.py (documented canonical constants)
docs/phase_notes/ # decision records for Phases 1, 2, 9 (referenced directly from code comments)
requirements.txt, requirements-optional.txt
```

Notably absent from the Phase 0 plan but never built, and why: a
`src/reconciliation/` module was planned but Phase 14's bottom-up logic
lives directly in `scripts/phase14_reconciliation.py` since it was a
one-off, full-scale analysis rather than reusable machinery another phase
needed to call.

## Known limitations carried forward honestly (not hidden)
- Phases 7-13, 15-18's ML results are on a 600-series sample, not the full
  30,490-series hierarchy (compute-constrained; documented per-phase).
- The rolling-feature cross-series leakage bug found in Phase 10 affected
  roughly 1.8% of Phases 7-9's training rows before the fix; those phases'
  qualitative conclusions are very unlikely to change but weren't
  recomputed after the fix (a real project would rerun them before
  finalizing Phase 22's comparison table).
- Phase 9's Bayesian model is documented but not adopted (non-convergent +
  computationally infeasible at this project's scale -- see
  `docs/phase_notes/phase9.md`).
