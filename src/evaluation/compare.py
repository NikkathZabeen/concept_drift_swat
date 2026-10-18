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
        _row("Scored events", g(baseline, "n_events_scored"), g(enhanced, "n_events_scored")),
        _row("Accuracy", g(baseline, "classification", "accuracy"), g(enhanced, "classification", "accuracy")),
        _row("Precision (Attack)", g(baseline, "classification", "precision_attack"), g(enhanced, "classification", "precision_attack")),
        _row("Recall (Attack)", g(baseline, "classification", "recall_attack"), g(enhanced, "classification", "recall_attack")),
        _row("F1 (Attack)", g(baseline, "classification", "f1_attack"), g(enhanced, "classification", "f1_attack")),
        _row("ROC-AUC", g(baseline, "classification", "roc_auc"), g(enhanced, "classification", "roc_auc")),
        _row("PR-AUC", g(baseline, "classification", "pr_auc"), g(enhanced, "classification", "pr_auc")),
        _row(
            "Detected drifts (count)",
            g(baseline, "drift_detection", "n_detected"),
            g(enhanced, "drift_detection", "n_detected"),
        ),
        _row(
            "Raw alarms (count)",
            g(baseline, "drift_detection", "n_raw_alarms"),
            g(enhanced, "drift_detection", "n_raw_alarms"),
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
            "False alarm rate (per 10k events)",
            g(baseline, "drift_detection", "false_alarms_per_10k_events"),
            g(enhanced, "drift_detection", "false_alarms_per_10k_events"),
        ),
        _row("Model resets", g(baseline, "n_model_resets"), g(enhanced, "n_model_resets")),
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
            "Event processing throughput (events/s)",
            g(baseline, "streaming", "event_processing_throughput_events_per_s"),
            g(enhanced, "streaming", "event_processing_throughput_events_per_s"),
        ),
        _row(
            "Latency p50 (ms)",
            g(baseline, "streaming", "latency_p50_ms"),
            g(enhanced, "streaming", "latency_p50_ms"),
        ),
        _row(
            "Latency p95 (ms)",
            g(baseline, "streaming", "latency_p95_ms"),
            g(enhanced, "streaming", "latency_p95_ms"),
        ),
        _row(
            "Latency p99 (ms)",
            g(baseline, "streaming", "latency_p99_ms"),
            g(enhanced, "streaming", "latency_p99_ms"),
        ),
        _row(
            "Peak memory (MB approx)",
            g(baseline, "peak_memory_mb_approx"),
            g(enhanced, "peak_memory_mb_approx"),
        ),
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
