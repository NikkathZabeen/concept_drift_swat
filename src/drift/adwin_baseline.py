"""Supervised ADWIN drift detection on prequential classification error."""

from __future__ import annotations

from dataclasses import dataclass, field

from river import drift


@dataclass
class SupervisedADWINDetector:
    """Monitors 0/1 loss with River's ADWIN (supervised error monitoring)."""

    delta: float = 0.002
    min_instances_between_drifts: int = 30
    _detector: drift.ADWIN = field(init=False)
    n_instances: int = 0
    n_drifts: int = 0
    n_warnings: int = 0
    last_drift_index: int | None = None
    drift_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._detector = drift.ADWIN(delta=self.delta)

    def update(self, y_true: int, y_pred: int, event_index: int) -> bool:
        error = 0 if y_true == y_pred else 1
        self._detector.update(error)
        self.n_instances += 1
        drift_detected = self._detector.drift_detected
        if drift_detected:
            if (
                self.last_drift_index is None
                or event_index - self.last_drift_index >= self.min_instances_between_drifts
            ):
                self.n_drifts += 1
                self.last_drift_index = event_index
                self.drift_indices.append(event_index)
                return True
        return False

    @property
    def width(self) -> float:
        return float(getattr(self._detector, "width", 0.0))

    @property
    def estimation(self) -> float:
        return float(getattr(self._detector, "estimation", 0.0))
