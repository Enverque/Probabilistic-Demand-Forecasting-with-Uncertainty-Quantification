import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.data.hierarchy import build_hierarchy, validate_against_weights, HIERARCHY_LEVELS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def data():
    return M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)


@pytest.fixture(scope="module")
def hierarchy(data):
    return build_hierarchy(data.sales, data.day_cols)


def test_all_12_levels_built(hierarchy):
    assert len(hierarchy) == 12
    assert set(hierarchy.keys()) == set(HIERARCHY_LEVELS.keys())


def test_series_counts_match_official_weights_file(data, hierarchy):
    results = validate_against_weights(hierarchy, data.weights)
    failed = {k: v for k, v in results.items() if not v["passed"]}
    assert not failed, f"level series counts don't match weights file: {failed}"


def test_bottom_level_sums_to_total(data, hierarchy, ):
    total_from_hierarchy = hierarchy["Level1_total"][data.day_cols].sum(axis=1).iloc[0]
    total_from_raw = data.sales[data.day_cols].sum().sum()
    assert total_from_hierarchy == total_from_raw


def test_store_sums_equal_state_sums_when_grouped(data, hierarchy):
    # Coherence *should* hold for a simple sum-based aggregation itself
    # (this is arithmetic, not forecasting) — stores within a state must
    # sum to that state's total in the raw data. This test exists to
    # separate "the aggregation code is right" from "independently
    # *forecasting* each level stays coherent" (Phase 2 section 8 below;
    # the latter is expected to fail, that's the whole point of Phase 14).
    store_level = hierarchy["Level3_store"]
    state_level = hierarchy["Level2_state"]
    store_to_state = data.sales[["store_id", "state_id"]].drop_duplicates()
    merged = store_level.merge(store_to_state, on="store_id")
    bottom_up = merged.groupby("state_id")[data.day_cols].sum()
    top = state_level.set_index("state_id")[data.day_cols]
    assert (bottom_up.sort_index() == top.sort_index()).all().all()
