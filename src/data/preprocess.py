"""Causal preprocessing for streaming SWaT features.

Guarantees:
- Strictly causal data flow: observations are transformed using only information
  available prior to the current observation.
- Supports both:
    1) OnlineStandardScaler: incremental running mean and variance via Welford's algorithm.
    2) Prefix PreprocessState: fitted strictly on a designated pre-evaluation warm-up prefix,
       with warm-up events explicitly excluded from scored evaluation.
- Categorical actuators and continuous sensors handled deterministically.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import yaml

from src.data.inspect_dataset import classify_columns, normalize_columns, parse_timestamps

logger = logging.getLogger(__name__)


@dataclass
class PreprocessState:
    numeric_means: dict[str, float] = field(default_factory=dict)
    numeric_stds: dict[str, float] = field(default_factory=dict)
    categorical_maps: dict[str, dict[Any, int]] = field(default_factory=dict)
    feature_order: list[str] = field(default_factory=list)
    fitted: bool = False
    warmup_n: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "categorical_maps": self.categorical_maps,
            "feature_order": self.feature_order,
            "fitted": self.fitted,
            "warmup_n": self.warmup_n,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessState:
        return cls(**data)


class OnlineStandardScaler:
    """Strictly causal online standard scaler using Welford's algorithm.

    Guarantees:
    - An observation x_t is scaled using statistics computed strictly from
      observations x_0, ..., x_{t-1}.
    - No future observations are ever accessed or leaked.
    - Continuous sensor features are standardized; categorical actuators are encoded.
    """

    def __init__(
        self,
        feature_order: list[str] | None = None,
        numeric_cols: list[str] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> None:
        self.feature_order: list[str] = list(feature_order or [])
        self.numeric_cols: set[str] = set(numeric_cols or [])
        self.categorical_cols: set[str] = set(categorical_cols or [])

        # Welford's accumulators: count, mean, M2
        self.counts: dict[str, int] = {}
        self.means: dict[str, float] = {}
        self.m2: dict[str, float] = {}
        self.categorical_maps: dict[str, dict[Any, int]] = {}
        self.n_samples: int = 0

    def _init_columns(self, features: dict[str, Any]) -> None:
        if not self.feature_order:
            num_cols, cat_cols = classify_columns(list(features.keys()))
            self.numeric_cols = set(num_cols)
            self.categorical_cols = set(cat_cols)
            self.feature_order = num_cols + cat_cols

    def transform_one(self, features: dict[str, Any]) -> dict[str, float]:
        """Causally transform features using current (prior-to-observation) statistics."""
        self._init_columns(features)
        out: dict[str, float] = {}
        for col in self.feature_order:
            val = features.get(col, 0.0)
            if col in self.numeric_cols:
                c = self.counts.get(col, 0)
                if c >= 2:
                    mean = self.means[col]
                    var = max(self.m2[col] / (c - 1), 1e-12)
                    std = np.sqrt(var)
                    out[col] = float((float(val) - mean) / std)
                else:
                    out[col] = float(val)
            else:
                mapping = self.categorical_maps.setdefault(col, {})
                out[col] = float(mapping.get(val, len(mapping)))
        return out

    def update_one(self, features: dict[str, Any]) -> None:
        """Causally update running statistics after transform_one has been called."""
        self._init_columns(features)
        self.n_samples += 1
        for col in self.feature_order:
            val = features.get(col, 0.0)
            if col in self.numeric_cols:
                fval = float(val)
                c = self.counts.get(col, 0) + 1
                self.counts[col] = c
                if c == 1:
                    self.means[col] = fval
                    self.m2[col] = 0.0
                else:
                    delta = fval - self.means[col]
                    self.means[col] += delta / c
                    delta2 = fval - self.means[col]
                    self.m2[col] += delta * delta2
            else:
                mapping = self.categorical_maps.setdefault(col, {})
                if val not in mapping:
                    mapping[val] = len(mapping)

    def to_state(self) -> PreprocessState:
        numeric_stds = {}
        for col in self.numeric_cols:
            c = self.counts.get(col, 0)
            if c >= 2:
                var = max(self.m2[col] / (c - 1), 1e-12)
                numeric_stds[col] = float(np.sqrt(var))
            else:
                numeric_stds[col] = 1.0

        return PreprocessState(
            numeric_means=dict(self.means),
            numeric_stds=numeric_stds,
            categorical_maps={k: dict(v) for k, v in self.categorical_maps.items()},
            feature_order=list(self.feature_order),
            fitted=self.n_samples > 0,
            warmup_n=self.n_samples,
        )

    def clone(self) -> OnlineStandardScaler:
        """Create an independent copy of this scaler's current Welford statistics."""
        scaler = OnlineStandardScaler(
            feature_order=list(self.feature_order),
            numeric_cols=list(self.numeric_cols),
            categorical_cols=list(self.categorical_cols),
        )
        scaler.counts = dict(self.counts)
        scaler.means = dict(self.means)
        scaler.m2 = dict(self.m2)
        scaler.categorical_maps = {k: dict(v) for k, v in self.categorical_maps.items()}
        scaler.n_samples = self.n_samples
        return scaler



def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def iter_swat_events(
    csv_path: Path,
    cfg: dict,
    max_events: int | None = None,
    start_index: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield chronologically ordered stream events from a SWaT CSV.

    Yields dictionary with:
    - event_id / global_event_index: canonical 0-indexed row position in the full dataset.
    - stream_step: 0-indexed step in the current experiment run.
    - timestamp_raw: original raw timestamp string.
    - timestamp: ISO 8601 formatted timestamp string.
    - features: raw sensor and actuator values.
    - label: binary integer label (Normal=0, Attack=1).
    - label_raw: original label string.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"SWaT dataset file not found at: {csv_path.resolve()}.\n"
            f"Please download it following archive/README.md or generate a test fixture."
        )

    data_cfg = cfg["data"]
    ts_col = data_cfg["timestamp_col"]
    label_col = data_cfg["label_col"]
    label_map = data_cfg["label_map"]
    ts_fmt = data_cfg["timestamp_format"]

    global_idx = 0
    stream_step = 0
    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        chunk = normalize_columns(chunk)
        for _, row in chunk.iterrows():
            if global_idx < start_index:
                global_idx += 1
                continue

            lbl_raw = str(row[label_col]).strip()
            label = label_map.get(lbl_raw, lbl_raw)
            ts_raw = str(row[ts_col]).strip()
            ts = parse_timestamps(pd.Series([ts_raw]), ts_fmt).iloc[0]
            features = {
                col: row[col]
                for col in chunk.columns
                if col not in {ts_col, label_col}
            }

            yield {
                "event_id": global_idx,
                "global_event_index": global_idx,
                "event_index": global_idx,
                "stream_step": stream_step,
                "timestamp_raw": ts_raw,
                "timestamp": ts.isoformat() if pd.notna(ts) else None,
                "features": features,
                "label": int(label) if str(label).isdigit() else label,
                "label_raw": lbl_raw,
            }

            global_idx += 1
            stream_step += 1
            if max_events is not None and stream_step >= max_events:
                return


def fit_preprocess_state(events: Iterator[dict[str, Any]], cfg: dict, warmup_n: int | None = None) -> PreprocessState:
    """Fit scaler/category maps using ONLY an explicit warm-up prefix (causal).

    Events consumed here must be strictly separated from scored evaluation events.
    """
    if warmup_n is None:
        warmup_n = cfg["data"].get("warmup_min_labeled", cfg["data"].get("initial_train_size", 5000))

    scaler = OnlineStandardScaler()
    count = 0
    for ev in events:
        scaler.update_one(ev["features"])
        count += 1
        if count >= warmup_n:
            break

    state = scaler.to_state()
    state.warmup_n = count
    return state


def vectorize_event(features: dict[str, Any], state: PreprocessState | OnlineStandardScaler) -> dict[str, float]:
    """Vectorize raw features using either a fitted PreprocessState or OnlineStandardScaler."""
    if isinstance(state, OnlineStandardScaler):
        return state.transform_one(features)

    if not state.fitted:
        raise RuntimeError("PreprocessState must be fitted before vectorization")

    out: dict[str, float] = {}
    for col in state.feature_order:
        raw_val = features.get(col, 0.0)
        if col in state.numeric_means:
            mean = state.numeric_means[col]
            std = state.numeric_stds.get(col, 1.0)
            val = float(raw_val)
            out[col] = (val - mean) / std if std > 1e-12 else 0.0
        else:
            mapping = state.categorical_maps.get(col, {})
            out[col] = float(mapping.get(raw_val, len(mapping)))
    return out


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._buf: deque[tuple[dict[str, float], int]] = deque(maxlen=capacity)

    def add(self, x: dict[str, float], y: int) -> None:
        self._buf.append((x, y))

    def items(self) -> list[tuple[dict[str, float], int]]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


def save_preprocess_state(state: PreprocessState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(state.to_dict(), f, indent=2)


def load_preprocess_state(path: Path) -> PreprocessState:
    with path.open() as f:
        return PreprocessState.from_dict(json.load(f))
