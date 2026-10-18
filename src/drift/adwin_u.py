"""ADWIN-U: unsupervised drift detection via ADWIN on streaming descriptive statistics.

Faithful to Assis & Souza (2025) at the level implemented here:
- No labels are used in the monitoring stage.
- Per-event descriptive statistics (mean, std, skewness, kurtosis) are computed over
  a bounded recent sample of standardized feature vectors.
- Each statistic stream is monitored by an independent ADWIN instance; drift is
  declared when any monitor fires (OR combination).

Difference from the full paper: we use a single pooled statistic vector rather than
class-conditional multi-detector banks tied to predicted labels (Section 4.1 optional mode).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from river import drift
from scipy.stats import kurtosis, skew


@dataclass
class ADWINU:
    delta: float = 0.002
    buffer_size: int = 512
    monitor_stats: tuple[str, ...] = ("mean", "std", "skewness", "kurtosis")
    min_instances_between_drifts: int = 30

    _buffer: deque[np.ndarray] = field(init=False)
    _monitors: dict[str, drift.ADWIN] = field(init=False)
    n_instances: int = 0
    n_drifts: int = 0
    last_drift_index: int | None = None
    drift_indices: list[int] = field(default_factory=list)
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
        """Update monitors using feature vector x only (unsupervised)."""
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
