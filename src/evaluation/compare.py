"""Baseline vs enhanced comparison tables."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_comparison_table(baseline: dict[str, Any], enhanced: dict[str, Any]) -> pd.DataFrame:
    def g(summary: dict, *keys, default=None):
        cur = summary
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, default)
        return cur

    rows = [
        _row("Accuracy", g(baseline, "classification", "accuracy"), g(enhanced, "classification", "accuracy")),
        _row("Precision", g(baseline, "classification", "precision"), g(enhanced, "classification", "precision")),
        _row("Recall", g(baseline, "classification", "recall"), g(enhanced, "classification", "recall")),
        _row("F1", g(baseline, "classification", "f1"), g(enhanced, "classification", "f1")),
        _row("ROC-AUC", g(baseline, "classification", "roc_auc"), g(enhanced, "classification", "roc_auc")),
        _row(
            "Detected drifts (count)",
            g(baseline, "drift_detection", "n_detected"),
            g(enhanced, "drift_detection", "n_detected"),
        ),
        _row(
            "Mean detection delay (events)",
            g(baseline, "drift_detection", "mean_detection_delay"),
            g(enhanced, "drift_detection", "mean_detection_delay"),
        ),
        _row(
            "False alarms",
            g(baseline, "drift_detection", "false_alarms"),
            g(enhanced, "drift_detection", "false_alarms"),
        ),
        _row(
            "Recovery time (events)",
            g(baseline, "recovery", "recovery_time_events"),
            g(enhanced, "recovery", "recovery_time_events"),
        ),
        _row(
            "Throughput (events/s)",
            g(baseline, "streaming", "throughput_events_per_s"),
            g(enhanced, "streaming", "throughput_events_per_s"),
        ),
        _row(
            "Latency p95 (ms)",
            g(baseline, "streaming", "latency_p95_ms"),
            g(enhanced, "streaming", "latency_p95_ms"),
        ),
        _row(
            "Peak memory (MB approx)",
            g(baseline, "peak_memory_mb_approx"),
            g(enhanced, "peak_memory_mb_approx"),
        ),
        _row("Model resets", g(baseline, "n_model_resets"), g(enhanced, "n_model_resets")),
    ]
    return pd.DataFrame(rows)


def _row(metric: str, b, e) -> dict:
    diff = None
    if isinstance(b, (int, float)) and isinstance(e, (int, float)):
        diff = e - b
    return {
        "Metric": metric,
        "Baseline ADWIN": b,
        "Enhanced Method": e,
        "Difference (Enhanced - Baseline)": diff,
    }
