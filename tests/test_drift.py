from src.drift.adwin_baseline import SupervisedADWINDetector
from src.drift.adwin_u import ADWINU
from src.evaluation.metrics import drift_detection_metrics, recovery_time_events


def test_supervised_adwin_detects_error_shift():
    det = SupervisedADWINDetector(delta=0.01, min_instances_between_drifts=5)
    fired = False
    for i in range(200):
        y_true, y_pred = (1, 0) if i > 80 else (0, 0)
        if det.update(y_true, y_pred, i):
            fired = True
    assert det.n_drifts >= 1 or fired


def test_adwin_u_unsupervised_no_labels():
    det = ADWINU(delta=0.01, buffer_size=64, min_instances_between_drifts=5)
    for i in range(300):
        x = {f"f{j}": (10.0 if i > 120 else 0.0) + j * 0.01 for j in range(8)}
        det.update(x, i)
    assert det.n_instances == 300


def test_detection_delay_metrics():
    m = drift_detection_metrics([1005], [1000], tolerance=50, stream_length=5000)
    assert m["mean_detection_delay"] == 5
    assert m["missed_reference_drifts"] == 0


def test_recovery_time():
    flags = [1] * 500 + [0] * 100 + [1] * 500
    r = recovery_time_events(flags, 500)
    assert r["recovered"] is True
