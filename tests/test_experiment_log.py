import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.experiment_log import log_experiment, get_experiment_log, EXPERIMENT_LOG_COLUMNS  # noqa: E402


def _valid_record(**overrides):
    base = {c: f"val_{c}" for c in EXPERIMENT_LOG_COLUMNS}
    base["metric_value"] = 0.5
    base.update(overrides)
    return base


def test_log_experiment_creates_file_with_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "log.csv")
        log_experiment(_valid_record(phase="Phase5"), path)
        df = get_experiment_log(path)
        assert list(df.columns) == EXPERIMENT_LOG_COLUMNS
        assert len(df) == 1
        assert df.iloc[0]["phase"] == "Phase5"


def test_log_experiment_appends_without_duplicating_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "log.csv")
        log_experiment(_valid_record(phase="Phase5"), path)
        log_experiment(_valid_record(phase="Phase6"), path)
        df = get_experiment_log(path)
        assert len(df) == 2
        assert list(df["phase"]) == ["Phase5", "Phase6"]


def test_log_experiment_rejects_missing_column():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "log.csv")
        bad_record = _valid_record()
        del bad_record["phase"]
        with pytest.raises(ValueError):
            log_experiment(bad_record, path)


def test_log_experiment_rejects_extra_column():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "log.csv")
        bad_record = _valid_record()
        bad_record["unexpected_col"] = "oops"
        with pytest.raises(ValueError):
            log_experiment(bad_record, path)


def test_get_experiment_log_returns_none_if_missing():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "does_not_exist.csv")
        assert get_experiment_log(path) is None
