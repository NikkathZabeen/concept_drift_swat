"""Baseline: supervised ADWIN on error + Hoeffding tree with full reset on drift."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.data.preprocess import ReplayBuffer, vectorize_event
from src.drift.adwin_baseline import SupervisedADWINDetector
from src.models.online_learning import build_classifier

logger = logging.getLogger(__name__)


@dataclass
class BaselineADWINSystem:
    cfg: dict
    seed: int = 42
    classifier: Any = field(init=False)
    detector: SupervisedADWINDetector = field(init=False)
    buffer: ReplayBuffer = field(init=False)
    n_resets: int = 0
    n_updates: int = 0
    warmup: int = 5000

    def __post_init__(self) -> None:
        clf_cfg = dict(self.cfg.get("classifier", {}))
        name = clf_cfg.pop("name", "hoeffding_tree")
        self.classifier = build_classifier(name, seed=self.seed, **clf_cfg)
        drift_cfg = self.cfg.get("drift", {})
        self.detector = SupervisedADWINDetector(
            delta=drift_cfg.get("delta", 0.002),
            min_instances_between_drifts=drift_cfg.get("min_instances_between_drifts", 30),
        )
        adapt = self.cfg.get("adaptation", {})
        self.buffer = ReplayBuffer(adapt.get("retrain_buffer_size", 2000))

    def predict(self, x: dict[str, float]) -> int:
        proba = self.classifier.predict_proba_one(x)
        if not proba:
            return 0
        return max(proba, key=proba.get)

    def _reset_classifier(self) -> None:
        clf_cfg = dict(self.cfg.get("classifier", {}))
        name = clf_cfg.pop("name", "hoeffding_tree")
        clf_cfg.pop("seed", None)
        self.classifier = build_classifier(name, seed=self.seed + self.n_resets, **clf_cfg)
        self.n_resets += 1
        for x, y in self.buffer.items():
            self.classifier.learn_one(x, y)
            self.n_updates += 1

    def on_drift(self) -> None:
        policy = self.cfg.get("adaptation", {}).get("policy", "full_reset")
        if policy == "full_reset":
            self._reset_classifier()

    def process_event(
        self,
        features_raw: dict[str, Any],
        label: int,
        event_index: int,
        state,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        x = vectorize_event(features_raw, state)
        y_pred = self.predict(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        drift = False
        if event_index >= self.warmup:
            drift = self.detector.update(label, y_pred, event_index)
            if drift:
                self.on_drift()

        # Prequential: learn after prediction
        self.classifier.learn_one(x, label)
        self.buffer.add(x, label)
        self.n_updates += 1

        return {
            "method": "baseline_adwin",
            "event_index": event_index,
            "y_true": label,
            "y_pred": y_pred,
            "correct": int(y_pred == label),
            "drift_detected": int(drift),
            "latency_ms": latency_ms,
            "detector_error_estimation": self.detector.estimation,
            "detector_width": self.detector.width,
        }
