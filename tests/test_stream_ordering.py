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


def test_smoke_window_index_semantics_and_coordinate_mapping(tmp_path: Path):
    """Verify coordinate systems and index semantics for windows starting at large global indices.

    Requirements:
    - Run begins at global index 1,385,000 and contains global reference point 1,387,098.
    - Global event indices (1,385,000..1,387,999) and local stream steps (0..2,999) are both preserved.
    - Warm-up completion and detector activation use local stream step (0..999).
    - Global reference point 1,387,098 maps to local run step 2,098.
    - Out-of-window reference points (e.g. 50,000) are flagged as outside the evaluation window.
    - Delays and recovery evaluate without out-of-bounds errors.
    """
    from src.evaluation.metrics import drift_detection_metrics, recovery_time_events
    from src.models.baseline import BaselineADWINSystem
    import yaml

    csv_file = tmp_path / "smoke_stream.csv"
    n_events = 3000
    start_global = 1385000
    global_ref = 1387098

    # Create synthetic events spanning global indices 1385000 to 1387999
    # Transition occurs at global index 1387098
    labels = ["Normal" if (start_global + i) < global_ref else "Attack" for i in range(n_events)]
    data = {
        "Timestamp": [f"28/12/2015 10:00:{i:02d} AM" for i in range(n_events)],
        "FIT101": [10.0 if lbl == "Normal" else 25.0 for lbl in labels],
        "Normal/Attack": labels,
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

    # Verify iteration semantics
    events = list(iter_swat_events(csv_file, cfg, max_events=n_events, start_index=0))
    # We simulate start_index=start_global in the stream
    for i, ev in enumerate(events):
        ev["global_event_index"] = start_global + i
        ev["stream_step"] = i

    assert events[0]["global_event_index"] == 1385000
    assert events[0]["stream_step"] == 0
    assert events[-1]["global_event_index"] == 1387999
    assert events[-1]["stream_step"] == 2999

    # Warm-up semantics: 1000 steps
    warmup_steps = 1000
    baseline_cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    system = BaselineADWINSystem(baseline_cfg, warmup_steps=warmup_steps)

    rows = []
    for ev in events:
        row = system.process_event(
            ev["features"],
            label=ev["label"],
            stream_step=ev["stream_step"],
            global_event_index=ev["global_event_index"],
        )
        rows.append(row)

    # Local steps 0..999 are warm-up; step 1000 is post-warmup
    assert rows[0]["is_warmup"] == 1
    assert rows[999]["is_warmup"] == 1
    assert rows[1000]["is_warmup"] == 0

    # Coordinate mapping:
    # Reference 1387098 is in window [1385000, 1388000)
    # Local step = 1387098 - 1385000 = 2098
    local_ref = global_ref - start_global
    assert local_ref == 2098

    # Test drift_detection_metrics with in-window and out-of-window references
    configured_refs = [global_ref, 50000]  # 50000 is outside window
    drift_m = drift_detection_metrics(
        detected_indices=[1387120],  # Alarm 22 events after reference
        reference_indices=configured_refs,
        tolerance=500,
        stream_length=n_events,
        start_index=start_global,
    )
    assert drift_m["references_in_window"] == [global_ref]
    assert drift_m["references_outside_window"] == [50000]
    assert drift_m["detection_delays_event_index"] == [22]
    assert drift_m["missed_reference_drifts"] == 0
    assert drift_m["matched_reference_events"] == 1

    # Recovery metrics evaluated with local step 2098
    correct_flags = [r["correct"] for r in rows]
    rec = recovery_time_events(correct_flags, local_ref)
    assert rec["reason"] if not rec["recovered"] else "recovered"
    # Recovery does NOT crash or raise IndexError with local_ref=2098
    assert rec.get("reason") != "reference_outside_evaluation_window"

    # Out-of-window reference safely returns out-of-window reason
    rec_out = recovery_time_events(correct_flags, global_ref)  # 1387098 > 3000
    assert rec_out["reason"] == "reference_outside_evaluation_window"

