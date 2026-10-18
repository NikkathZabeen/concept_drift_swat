"""Unit tests for classification, probability-based metrics, and performance."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import classification_metrics, prequential_curve, streaming_perf


def test_classification_metrics_continuous_scores():
    """Verify continuous probability scores produce exact scikit-learn ROC-AUC and PR-AUC."""
    y_true = [0, 0, 1, 1, 0, 1]
    y_score = [0.1, 0.4, 0.65, 0.8, 0.2, 0.9]
    y_pred = [int(s >= 0.5) for s in y_score]

    m = classification_metrics(y_true, y_pred, y_score)
    expected_roc = float(roc_auc_score(y_true, y_score))
    expected_pr = float(average_precision_score(y_true, y_score))

    assert m["roc_auc"] is not None
    assert m["pr_auc"] is not None
    assert pytest.approx(m["roc_auc"]) == expected_roc
    assert pytest.approx(m["pr_auc"]) == expected_pr
    assert m["accuracy"] == 1.0


def test_classification_metrics_without_scores():
    """Verify missing continuous scores results in roc_auc=None rather than fake binary AUC."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    m = classification_metrics(y_true, y_pred, y_score=None)

    assert m["accuracy"] == 1.0
    assert m["roc_auc"] is None
    assert m["pr_auc"] is None


def test_single_class_window_safety():
    """Verify single-class windows safely return None without raising ValueError."""
    # Window with only Normal (0)
    y_true_all_zeros = [0, 0, 0, 0, 0]
    y_score = [0.1, 0.2, 0.05, 0.3, 0.15]
    y_pred = [0, 0, 0, 0, 0]

    m_zero = classification_metrics(y_true_all_zeros, y_pred, y_score)
    assert m_zero["accuracy"] == 1.0
    assert m_zero["roc_auc"] is None
    assert m_zero["pr_auc"] is None

    # Window with only Attack (1)
    y_true_all_ones = [1, 1, 1, 1]
    y_pred_ones = [1, 1, 1, 1]
    y_score_ones = [0.9, 0.8, 0.75, 0.95]
    m_one = classification_metrics(y_true_all_ones, y_pred_ones, y_score_ones)
    assert m_one["roc_auc"] is None
    assert m_one["pr_auc"] is None


def test_prequential_curve():
    curve = prequential_curve([1, 0, 1, 1], window=2)
    assert len(curve) == 4
    assert curve[-1] == 1.0


def test_streaming_perf_percentiles():
    latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    perf = streaming_perf(
        latencies_total_ms=latencies,
        total_wall_s=0.5,
        n_events=len(latencies),
        latencies_pred_ms=[l * 0.5 for l in latencies],
        latencies_det_ms=[l * 0.2 for l in latencies],
        latencies_learn_ms=[l * 0.3 for l in latencies],
    )
    assert "latency_p50_ms" in perf
    assert "latency_p95_ms" in perf
    assert "latency_p99_ms" in perf
    assert perf["latency_p95_ms"] > perf["latency_p50_ms"]
    assert perf["throughput_events_per_s"] == 20.0
