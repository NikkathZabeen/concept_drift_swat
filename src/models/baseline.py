"""Baseline: supervised ADWIN on error + Hoeffding tree with full reset on drift."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.data.preprocess import OnlineStandardScaler, PreprocessState, ReplayBuffer, vectorize_event
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
    scaler: OnlineStandardScaler = field(default_factory=OnlineStandardScaler)
    n_resets: int = 0
    n_updates: int = 0
    warmup_steps: int = 5000

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
        if not hasattr(self, "scaler") or self.scaler is None:
            self.scaler = OnlineStandardScaler()

    def predict_proba(self, x: dict[str, float]) -> tuple[int, float]:
        """Return (predicted_class, positive_class_probability)."""
        proba = self.classifier.predict_proba_one(x)
        if not proba:
            return 0, 0.0
        y_score = float(proba.get(1, 0.0))
        y_pred = int(y_score >= 0.5) if (0 in proba and 1 in proba) else int(max(proba, key=proba.get))
        return y_pred, y_score

    def predict(self, x: dict[str, float]) -> int:
        y_pred, _ = self.predict_proba(x)
        return y_pred

    def _reset_classifier(self) -> None:
        clf_cfg = dict(self.cfg.get("classifier", {}))
        name = clf_cfg.pop("name", "hoeffding_tree")
        clf_cfg.pop("seed", None)
        self.classifier = build_classifier(name, seed=self.seed + self.n_resets + 1, **clf_cfg)
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
        stream_step: int,
        global_event_index: int,
        state: PreprocessState | OnlineStandardScaler | None = None,
    ) -> dict[str, Any]:
        """Prequential evaluation step:

        1. Transform features causally using statistics computed strictly prior to this observation.
        2. Predict continuous score and binary class before learning.
        3. Supervised drift monitor update (active only after warmup_steps).
        4. Trigger adaptation on drift alarm.
        5. Learn from true label and buffer for replay.
        6. Causally update scaler statistics after transform, prediction, and drift decision.
        """
        active_scaler = state if state is not None else self.scaler
        t0 = time.perf_counter()
        x = vectorize_event(features_raw, active_scaler)
        y_pred, y_score = self.predict_proba(x)
        t1 = time.perf_counter()
        t_pred_ms = (t1 - t0) * 1000.0

        is_warmup = stream_step < self.warmup_steps
        drift = False

        if not is_warmup:
            drift = self.detector.update(label, y_pred, global_event_index)

        t2 = time.perf_counter()
        t_det_ms = (t2 - t1) * 1000.0

        if drift:
            self.on_drift()

        t3 = time.perf_counter()
        t_drift_ms = (t3 - t2) * 1000.0

        # Prequential learning: strictly after prediction and drift decision
        self.classifier.learn_one(x, label)
        self.buffer.add(x, label)
        self.n_updates += 1

        # Causal scaler update: strictly after transform and prediction
        if isinstance(active_scaler, OnlineStandardScaler):
            active_scaler.update_one(features_raw)

        t4 = time.perf_counter()
        t_learn_ms = (t4 - t3) * 1000.0
        t_total_ms = (t4 - t0) * 1000.0

        return {
            "method": "baseline_adwin",
            "stream_step": stream_step,
            "event_index": global_event_index,
            "global_event_index": global_event_index,
            "is_warmup": int(is_warmup),
            "y_true": label,
            "y_pred": y_pred,
            "y_score": y_score,
            "correct": int(y_pred == label),
            "drift_detected": int(drift),
            "latency_ms": t_total_ms,
            "latency_pred_ms": t_pred_ms,
            "latency_detector_ms": t_det_ms,
            "latency_drift_ms": t_drift_ms,
            "latency_learn_ms": t_learn_ms,
            "latency_total_ms": t_total_ms,
            "detector_error_estimation": self.detector.estimation,
            "detector_width": self.detector.width,
        }
