"""ADWIN-U-Inspired Unsupervised Drift Detection.

Academic attribution & methodological disclosure:
- Inspired by Assis & Souza (2025) ("ADWIN-U: Adaptive Windowing for Unsupervised Drift Detection on Data Streams").
- Strictly unsupervised: Ground-truth labels are NEVER accessed during monitoring.
- Implementation specifics: Monitors streaming descriptive statistics (mean, std, skewness, kurtosis)
  over a sliding buffer of standardized feature vectors pooled across all sensor/actuator dimensions.
- Distinction from full published paper: Uses pooled descriptive statistics rather than
  class-conditional multi-detector banks keyed by predicted labels.
- Both raw alarm occurrences and cooldown-filtered drift events are recorded for honest accounting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from river import drift
from scipy.stats import kurtosis, skew


@dataclass
class ADWINU:
    """ADWIN-U-inspired unsupervised drift detector monitoring streaming feature statistics."""

    delta: float = 0.002
    buffer_size: int = 512
    monitor_stats: tuple[str, ...] = ("mean", "std", "skewness", "kurtosis")
    min_instances_between_drifts: int = 50

    _buffer: deque[np.ndarray] = field(init=False)
    _monitors: dict[str, drift.ADWIN] = field(init=False)
    n_instances: int = 0
    n_drifts: int = 0
    n_raw_alarms: int = 0
    last_drift_index: int | None = None
    drift_indices: list[int] = field(default_factory=list)
    raw_alarm_indices: list[int] = field(default_factory=list)
    last_stats: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._buffer = deque(maxlen=self.buffer_size)
        self._monitors = {name: drift.ADWIN(delta=self.delta) for name in self.monitor_stats}

    def _compute_stats(self) -> dict[str, float]:
        if len(self._buffer) < 8:
            return {}
        mat = np.vstack(list(self._buffer))
        flat = mat.reshape(-1)
        return {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "skewness": float(skew(flat, bias=False)),
            "kurtosis": float(kurtosis(flat, bias=False)),
        }

    def update(self, x: dict[str, float], event_index: int) -> bool:
        """Update monitors using feature vector x only. Ground truth is strictly NOT accessed."""
        vec = np.fromiter(x.values(), dtype=float)
        self._buffer.append(vec)
        self.n_instances += 1
        stats = self._compute_stats()
        if not stats:
            return False
        self.last_stats = stats

        fired = False
        for name, monitor in self._monitors.items():
            if name in stats:
                monitor.update(stats[name])
                if monitor.drift_detected:
                    fired = True

        if fired:
            self.n_raw_alarms += 1
            self.raw_alarm_indices.append(event_index)
            # Enforce cooldown / minimum instances between adaptation triggers
            if (
                self.last_drift_index is None
                or event_index - self.last_drift_index >= self.min_instances_between_drifts
            ):
                self.n_drifts += 1
                self.last_drift_index = event_index
                self.drift_indices.append(event_index)
                return True
        return False

    def monitor_widths(self) -> dict[str, float]:
        return {
            name: float(getattr(m, "width", 0.0)) for name, m in self._monitors.items()
        }

    def reset_detectors(self) -> None:
        """Reset internal ADWIN monitors upon model reset."""
        self._monitors = {name: drift.ADWIN(delta=self.delta) for name in self.monitor_stats}
        self._buffer.clear()
