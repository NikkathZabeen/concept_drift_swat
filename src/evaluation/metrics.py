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


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if not y_true:
        return {}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    if len(set(y_true)) > 1:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_pred))
            out["pr_auc"] = float(average_precision_score(y_true, y_pred))
        except ValueError:
            pass
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
) -> dict[str, Any]:
    """Match detected drifts to reference points within tolerance (event-time delay)."""
    if not reference_indices:
        return {
            "n_detected": len(detected_indices),
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
        "n_detected": len(detected_indices),
        "n_reference": len(reference_indices),
        "detection_delays_event_index": matched_delays,
        "mean_detection_delay": float(np.mean(matched_delays)) if matched_delays else None,
        "missed_reference_drifts": missed,
        "false_alarms": false_alarms,
        "tolerance_events": tolerance,
    }
    if stream_length:
        out["false_alarms_per_10k_events"] = false_alarms / (stream_length / 10_000)
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


def streaming_perf(latencies_ms: list[float], total_wall_s: float, n_events: int) -> dict[str, float]:
    if not latencies_ms:
        return {}
    arr = np.array(latencies_ms)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "throughput_events_per_s": n_events / total_wall_s if total_wall_s > 0 else 0.0,
    }
