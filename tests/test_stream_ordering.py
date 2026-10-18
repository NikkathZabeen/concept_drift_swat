"""Tests for timestamp parsing, stream order validation, and timestamp anomalies."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.inspect_dataset import inspect_csv, parse_timestamps
from src.data.preprocess import iter_swat_events


def test_timestamp_parsing_valid_and_whitespace():
    fmt = "%d/%m/%Y %I:%M:%S %p"
    series = pd.Series(["  28/12/2015 10:29:14 AM  ", "22/12/2015 05:18:22 PM"])
    parsed = parse_timestamps(series, fmt)
    assert parsed.iloc[0] == pd.Timestamp("2015-12-28 10:29:14")
    assert parsed.iloc[1] == pd.Timestamp("2015-12-22 17:18:22")
    assert parsed.isna().sum() == 0


def test_timestamp_parsing_corrupt():
    fmt = "%d/%m/%Y %I:%M:%S %p"
    series = pd.Series(["invalid_timestamp", "28/12/2015 10:29:14 AM"])
    parsed = parse_timestamps(series, fmt)
    assert pd.isna(parsed.iloc[0])
    assert pd.notna(parsed.iloc[1])


def test_stream_order_preserves_source_order(tmp_path: Path):
    """Verify stream events preserve chronological source file order and assign stream_step."""
    csv_file = tmp_path / "order_test.csv"
    data = {
        "Timestamp": ["28/12/2015 10:00:00 AM", "28/12/2015 10:00:01 AM", "28/12/2015 10:00:05 AM"],  # 4s gap
        "FIT101": [1.0, 2.0, 3.0],
        "MV101": [1, 1, 2],
        "Normal/Attack": ["Normal", "Normal", "Attack"],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    cfg = {
        "data": {
            "timestamp_col": "Timestamp",
            "label_col": "Normal/Attack",
            "label_map": {"Normal": 0, "Attack": 1},
            "timestamp_format": "%d/%m/%Y %I:%M:%S %p",
        }
    }
    events = list(iter_swat_events(csv_file, cfg))
    assert len(events) == 3
    assert [e["stream_step"] for e in events] == [0, 1, 2]
    assert [e["global_event_index"] for e in events] == [0, 1, 2]
    assert events[0]["label"] == 0
    assert events[2]["label"] == 1


def test_inspect_csv_detects_non_monotonic_and_gaps(tmp_path: Path):
    csv_file = tmp_path / "anomaly_test.csv"
    # Row 1 is 10:00:05 (gap > 2s from 10:00:00).
    data = {
        "Timestamp": [
            "28/12/2015 10:00:00 AM",
            "28/12/2015 10:00:05 AM",  # gap of 5s
            "28/12/2015 10:00:06 AM",
        ],
        "FIT101": [1.0, 2.0, 3.0],
        "Normal/Attack": ["Normal", "Normal", "Attack"],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    cfg = {
        "data": {
            "timestamp_col": "Timestamp",
            "label_col": "Normal/Attack",
            "label_map": {"Normal": 0, "Attack": 1},
            "timestamp_format": "%d/%m/%Y %I:%M:%S %p",
        }
    }
    report = inspect_csv(csv_file, cfg, sample_chunks=1)
    assert report["n_rows"] == 3
    assert report["gaps_gt_2s"] >= 1
    assert report["timestamp_parse_failures"] == 0
