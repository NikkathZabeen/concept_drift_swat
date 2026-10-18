"""Unit tests for supervised ADWIN, ADWIN-U-inspired detector, and drift metrics."""

from pathlib import Path

import yaml

from src.data.preprocess import PreprocessState
from src.drift.adwin_baseline import SupervisedADWINDetector
from src.drift.adwin_u import ADWINU
from src.evaluation.metrics import drift_detection_metrics, recovery_time_events
from src.models.baseline import BaselineADWINSystem
from src.models.enhanced import EnhancedAdaptiveSystem


def test_supervised_adwin_detects_error_shift():
    det = SupervisedADWINDetector(delta=0.01, min_instances_between_drifts=5)
    fired = False
    for i in range(200):
        y_true, y_pred = (1, 0) if i > 80 else (0, 0)
        if det.update(y_true, y_pred, i):
            fired = True
    assert det.n_drifts >= 1 or fired
    assert det.n_raw_alarms >= det.n_drifts


def test_adwin_u_unsupervised_no_labels():
    """Verify ADWIN-U operates strictly without labels and detects sudden distribution shift."""
    det = ADWINU(delta=0.01, buffer_size=64, min_instances_between_drifts=5)
    fired = False
    for i in range(300):
        # Feature mean shifts abruptly at i=120
        x = {f"f{j}": (15.0 if i > 120 else 0.0) + j * 0.01 for j in range(8)}
        if det.update(x, i):
            fired = True

    assert det.n_instances == 300
    assert fired or det.n_raw_alarms > 0


def test_detection_delay_metrics():
    m = drift_detection_metrics([1005], [1000], tolerance=50, stream_length=5000, raw_detected_indices=[1005, 1006])
    assert m["mean_detection_delay"] == 5
    assert m["missed_reference_drifts"] == 0
    assert m["matched_reference_events"] == 1
    assert m["unmatched_alarms"] == 0
    assert m["n_raw_alarms"] == 2


def test_detection_metrics_false_alarms():
    # Detection at 500 is far outside tolerance around reference at 1000
    m = drift_detection_metrics([500], [1000], tolerance=50, stream_length=10000)
    assert m["missed_reference_drifts"] == 1
    assert m["unmatched_alarms"] == 1
    assert m["false_alarms_per_10k_events"] == 1.0


def test_recovery_time():
    flags = [1] * 500 + [0] * 100 + [1] * 500
    r = recovery_time_events(flags, 500)
    assert r["recovered"] is True
    assert r["recovery_time_events"] is not None


def test_baseline_and_enhanced_model_resets():
    b_cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    e_cfg = yaml.safe_load(Path("configs/enhanced.yaml").read_text())

    b_sys = BaselineADWINSystem(b_cfg, warmup_steps=10)
    e_sys = EnhancedAdaptiveSystem(e_cfg, warmup_steps=10)

    # Initial state
    assert b_sys.n_resets == 0
    assert e_sys.n_resets == 0

    # Add items to buffer and trigger on_drift
    b_sys.buffer.add({"FIT101": 1.0}, 0)
    b_sys.on_drift()
    assert b_sys.n_resets == 1

    e_sys.buffer.add({"FIT101": 1.0}, 0)
    e_sys.on_drift()
    assert e_sys.n_resets == 1


def test_drift_metrics_signed_delays_and_operational_false_alarms():
    """Verify signed delays, raw alarm distinction, and operational false alarm definitions."""
    # 1. Negative delay (early / anticipatory detection before reference)
    m_early = drift_detection_metrics(
        detected_indices=[980],  # 20 events before reference at 1000
        reference_indices=[1000],
        tolerance=50,
        stream_length=2000,
        raw_detected_indices=[978, 980],
    )
    assert m_early["detection_delays_event_index"] == [-20]
    assert m_early["mean_detection_delay"] == -20.0
    assert m_early["matched_reference_events"] == 1
    assert m_early["missed_reference_drifts"] == 0
    assert m_early["false_alarms"] == 0
    assert m_early["n_raw_alarms"] == 2
    assert m_early["n_detected"] == 1

    # 2. Positive delay (late detection after reference)
    m_late = drift_detection_metrics(
        detected_indices=[1035],  # 35 events after reference at 1000
        reference_indices=[1000],
        tolerance=50,
        stream_length=2000,
    )
    assert m_late["detection_delays_event_index"] == [35]
    assert m_late["mean_detection_delay"] == 35.0

    # 3. Operational false alarms: detections outside tolerance of any in-window reference
    m_fa = drift_detection_metrics(
        detected_indices=[200, 1010],  # 200 is far outside tolerance 50
        reference_indices=[1000],
        tolerance=50,
        stream_length=2000,
    )
    assert m_fa["matched_reference_events"] == 1
    assert m_fa["unmatched_alarms"] == 1
    assert m_fa["false_alarms"] == 1
    assert "tolerance window" in m_fa["false_alarms_operational_definition"]


    # 4. Out-of-window references are reported as outside and not penalized as missed in-window
    m_out = drift_detection_metrics(
        detected_indices=[1500],
        reference_indices=[5000],  # 5000 is outside [1000, 2000)
        tolerance=50,
        stream_length=1000,
        start_index=1000,
    )
    assert m_out["n_reference_in_window"] == 0
    assert m_out["references_outside_window"] == [5000]
    assert m_out["missed_reference_drifts"] == 0

