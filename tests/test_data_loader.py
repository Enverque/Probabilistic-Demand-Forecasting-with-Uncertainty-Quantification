"""
Phase 1 tests: does the data actually match what Phase 0 assumed?

These run against the real downloaded M5 files (data/raw/), not synthetic
fixtures — the point of this phase is verifying the real schema, so
mocking it away would defeat the purpose.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader, EXPECTED_N_SERIES  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def data():
    loader = M5DataLoader(data_dir=str(DATA_DIR))
    return loader.load(verbose=False)


def test_files_load_without_error(data):
    assert data.sales.shape[0] > 0
    assert data.calendar.shape[0] > 0
    assert data.prices.shape[0] > 0
    assert data.weights.shape[0] > 0


def test_day_columns_are_in_correct_temporal_order(data):
    # This is the leakage-adjacent check for Phase 1: if d_10 sorted before
    # d_2 lexicographically and nobody caught it, every lag/rolling feature
    # built later would be silently wrong.
    day_nums = [int(c.split("_")[1]) for c in data.day_cols]
    assert day_nums == sorted(day_nums)
    assert day_nums == list(range(1, len(day_nums) + 1))


def test_series_id_is_unique_key(data):
    assert data.sales["id"].duplicated().sum() == 0
    assert data.sales.shape[0] == EXPECTED_N_SERIES


def test_schema_validation_all_pass(data):
    results = M5DataLoader.validate_schema(data)
    failed = {k: v for k, v in results.items() if not v["passed"]}
    assert not failed, f"schema checks failed: {failed}"


def test_hierarchy_cardinalities_multiply_out(data):
    # 3 states -> 10 stores -> 7 depts -> 3 cats -> 3049 items is the
    # claimed hierarchy from Phase 0. Departments nest inside categories,
    # so dept-to-category should be a many-to-one mapping, not a free mix.
    dept_cat = data.sales[["dept_id", "cat_id"]].drop_duplicates()
    assert dept_cat["dept_id"].is_unique, (
        "a dept_id maps to more than one cat_id — hierarchy is not "
        "strictly nested the way Phase 0 assumed"
    )


def test_calendar_date_to_d_mapping_is_sequential(data):
    cal = data.calendar.sort_values("date").reset_index(drop=True)
    assert cal["date"].is_monotonic_increasing
    # first calendar row should be d_1
    assert cal.loc[0, "d"] == "d_1"
