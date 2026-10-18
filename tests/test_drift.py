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
