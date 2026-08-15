# Phase 2 — Hierarchy: the mean-based coherence null result

## What happened
The first attempt to demonstrate hierarchical incoherence (Store vs. State
forecasts not summing correctly) used a MEAN-based seasonal-naive forecast
(average of the last 4 same-weekday values). The result was a gap of
EXACTLY 0.0% for every state.

## Why
Mean is a linear operator, and summing stores into a state is also linear,
so "sum of per-store means" and "mean of the state's summed history" are
mathematically identical. A mean-based naive forecast is coherent BY
CONSTRUCTION — this isn't a demonstration failing, it's a genuine finding
that a bad example was chosen to illustrate incoherence with.

## Fix
Switched to a MEDIAN-based forecast (doesn't distribute over sums), which
produced real, non-zero gaps (up to 0.81% of a state's total) — representative
of what happens with the nonlinear/tree-based models used from Phase 6 onward.

## Where this matters again later
Phase 14 (Hierarchical Reconciliation) hit the exact same degenerate
mean-is-linear result on its first run (DIRECT and BOTTOM-UP scores were
identical at all 12 levels using mean-based seasonal-naive) and applied the
same median-based fix, for the same underlying reason.
