"""Metrics for predictive quality, drift detection, and continual learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class RecoveryConfig:
    window: int = 500
    threshold_fraction_of_pre_drift: float = 0.9
    min_pre_window: int = 200


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics with continuous probability ranking metrics.

    Requirements:
    - ROC-AUC and Average Precision (PR-AUC) use continuous predicted probability
      scores for class 1 (Attack).
    - If continuous scores are not provided or if y_true contains only a single class,
      roc_auc and pr_auc are safely returned as None (not estimated via hard binary predictions).
    - Class 1 (Attack) precision, recall, and F1 are explicitly provided.
    """
    if not y_true:
        return {}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    out: dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "precision_attack": prec,
        "recall": rec,
        "recall_attack": rec,
        "f1": f1,
        "f1_attack": f1,
        "confusion_matrix": cm.tolist(),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "roc_auc": None,
        "pr_auc": None,
    }

    # Probability-based ranking metrics: requires continuous scores and both classes present
    if y_score is not None and len(y_score) == len(y_true):
        unique_classes = set(y_true)
        if len(unique_classes) > 1:
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            except ValueError:
                out["roc_auc"] = None

            try:
                out["pr_auc"] = float(average_precision_score(y_true, y_score))
            except ValueError:
                out["pr_auc"] = None

    return out


def prequential_curve(correct_flags: list[int], window: int = 1000) -> list[float]:
    if not correct_flags:
        return []
    arr = np.array(correct_flags, dtype=float)
    curve = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        curve.append(float(arr[start : i + 1].mean()))
    return curve


def drift_detection_metrics(
    detected_indices: list[int],
    reference_indices: list[int],
    tolerance: int = 500,
    stream_length: int | None = None,
    raw_detected_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Match detected drifts to reference points within tolerance (event-time delay).

    Honest reporting:
    - Distinguishes raw statistical alarms from cooldown-filtered model resets.
    - Matches candidate detections within tolerance window of reference events.
    - Computes detection delays and false alarms outside tolerance.
    """
    n_detected = len(detected_indices)
    n_raw = len(raw_detected_indices) if raw_detected_indices is not None else n_detected

    if not reference_indices:
        return {
            "n_detected": n_detected,
            "n_raw_alarms": n_raw,
            "n_reference": 0,
            "note": "No reference drift indices configured",
        }

    matched_delays: list[int] = []
    false_alarms = 0
    missed = 0
    used_detections: set[int] = set()

    for ref in reference_indices:
        candidates = [
            d for d in detected_indices if abs(d - ref) <= tolerance and d not in used_detections
        ]
        if candidates:
            best = min(candidates, key=lambda d: abs(d - ref))
            used_detections.add(best)
            matched_delays.append(best - ref)
        else:
            missed += 1

    for d in detected_indices:
        if d not in used_detections:
            if not any(abs(d - ref) <= tolerance for ref in reference_indices):
                false_alarms += 1

    out: dict[str, Any] = {
        "n_detected": n_detected,
        "n_raw_alarms": n_raw,
        "n_reference": len(reference_indices),
        "detection_delays_event_index": matched_delays,
        "mean_detection_delay": float(np.mean(matched_delays)) if matched_delays else None,
        "missed_reference_drifts": missed,
        "matched_reference_events": len(matched_delays),
        "unmatched_alarms": false_alarms,
        "false_alarms": false_alarms,
        "tolerance_events": tolerance,
    }
    if stream_length and stream_length > 0:
        out["false_alarms_per_10k_events"] = float(false_alarms / (stream_length / 10_000))
    return out


def recovery_time_events(
    correct_flags: list[int],
    drift_index: int,
    cfg: RecoveryConfig | None = None,
) -> dict[str, Any]:
    """Recovery = first index after drift where rolling accuracy >= threshold * pre-drift accuracy."""
    cfg = cfg or RecoveryConfig()
    if drift_index >= len(correct_flags):
        return {"recovered": False, "recovery_time_events": None}

    pre_start = max(0, drift_index - cfg.window)
    pre = correct_flags[pre_start:drift_index]
    if len(pre) < cfg.min_pre_window:
        return {"recovered": False, "recovery_time_events": None, "reason": "insufficient_pre_window"}

    pre_acc = float(np.mean(pre))
    threshold = cfg.threshold_fraction_of_pre_drift * pre_acc

    for i in range(drift_index, len(correct_flags)):
        start = max(drift_index, i - cfg.window + 1)
        rolling = float(np.mean(correct_flags[start : i + 1]))
        if rolling >= threshold:
            return {
                "recovered": True,
                "recovery_time_events": i - drift_index,
                "pre_drift_accuracy": pre_acc,
                "threshold": threshold,
            }
    return {"recovered": False, "recovery_time_events": None, "pre_drift_accuracy": pre_acc}


def streaming_perf(
    latencies_total_ms: list[float],
    total_wall_s: float,
    n_events: int,
    latencies_pred_ms: list[float] | None = None,
    latencies_det_ms: list[float] | None = None,
    latencies_learn_ms: list[float] | None = None,
) -> dict[str, float]:
    """Fair streaming performance instrumentation distinguishing stages and reporting percentiles."""
    if not latencies_total_ms:
        return {}

    arr_tot = np.array(latencies_total_ms)
    tot_processing_s = float(np.sum(arr_tot) / 1000.0)

    perf: dict[str, float] = {
        "latency_mean_ms": float(arr_tot.mean()),
        "latency_p50_ms": float(np.percentile(arr_tot, 50)),
        "latency_p95_ms": float(np.percentile(arr_tot, 95)),
        "latency_p99_ms": float(np.percentile(arr_tot, 99)),
        "throughput_events_per_s": float(n_events / total_wall_s) if total_wall_s > 0 else 0.0,
        "event_processing_throughput_events_per_s": float(n_events / tot_processing_s) if tot_processing_s > 0 else 0.0,
    }

    if latencies_pred_ms:
        arr_p = np.array(latencies_pred_ms)
        perf["latency_pred_mean_ms"] = float(arr_p.mean())
        perf["latency_pred_p50_ms"] = float(np.percentile(arr_p, 50))
        perf["latency_pred_p95_ms"] = float(np.percentile(arr_p, 95))

    if latencies_det_ms:
        arr_d = np.array(latencies_det_ms)
        perf["latency_detector_mean_ms"] = float(arr_d.mean())
        perf["latency_detector_p50_ms"] = float(np.percentile(arr_d, 50))
        perf["latency_detector_p95_ms"] = float(np.percentile(arr_d, 95))

    if latencies_learn_ms:
        arr_l = np.array(latencies_learn_ms)
        perf["latency_learn_mean_ms"] = float(arr_l.mean())
        perf["latency_learn_p50_ms"] = float(np.percentile(arr_l, 50))
        perf["latency_learn_p95_ms"] = float(np.percentile(arr_l, 95))

    return perf
