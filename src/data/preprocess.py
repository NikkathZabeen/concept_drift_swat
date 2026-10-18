"""Causal preprocessing for streaming SWaT features."""

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

from src.data.inspect_dataset import normalize_columns, parse_timestamps

logger = logging.getLogger(__name__)


@dataclass
class PreprocessState:
    numeric_means: dict[str, float] = field(default_factory=dict)
    numeric_stds: dict[str, float] = field(default_factory=dict)
    categorical_maps: dict[str, dict[Any, int]] = field(default_factory=dict)
    feature_order: list[str] = field(default_factory=list)
    fitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "categorical_maps": self.categorical_maps,
            "feature_order": self.feature_order,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessState:
        return cls(**data)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def iter_swat_events(
    csv_path: Path,
    cfg: dict,
    max_events: int | None = None,
    start_index: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield chronologically ordered stream events from a SWaT CSV."""
    data_cfg = cfg["data"]
    ts_col = data_cfg["timestamp_col"]
    label_col = data_cfg["label_col"]
    label_map = data_cfg["label_map"]
    ts_fmt = data_cfg["timestamp_format"]

    n = 0
    emitted = 0
    for chunk in pd.read_csv(csv_path, chunksize=50_000):
        chunk = normalize_columns(chunk)
        for _, row in chunk.iterrows():
            if n < start_index:
                n += 1
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
                "event_id": n,
                "event_index": n,
                "timestamp_raw": ts_raw,
                "timestamp": ts.isoformat() if pd.notna(ts) else None,
                "features": features,
                "label": int(label) if str(label).isdigit() else label,
                "label_raw": lbl_raw,
            }
            n += 1
            emitted += 1
            if max_events is not None and emitted >= max_events:
                return


def fit_preprocess_state(events: Iterator[dict[str, Any]], cfg: dict) -> PreprocessState:
    """Fit scaler/category maps using only the initial training window (causal)."""
    data_cfg = cfg["data"]
    init_n = cfg["data"].get("warmup_min_labeled", cfg["data"].get("initial_train_size", 5000))
    numeric_cols = None
    cat_cols = None
    sums: dict[str, float] = {}
    sq_sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    cat_values: dict[str, set[Any]] = {}

    state = PreprocessState()
    for i, ev in enumerate(events):
        feats = ev["features"]
        if numeric_cols is None:
            from src.data.inspect_dataset import classify_columns

            numeric_cols, cat_cols = classify_columns(list(feats.keys()))
            state.feature_order = numeric_cols + cat_cols

        for col in numeric_cols:
            val = float(feats[col])
            sums[col] = sums.get(col, 0.0) + val
            sq_sums[col] = sq_sums.get(col, 0.0) + val * val
            counts[col] = counts.get(col, 0) + 1
        for col in cat_cols:
            cat_values.setdefault(col, set()).add(feats[col])

        if i + 1 >= init_n:
            break

    for col in numeric_cols or []:
        c = counts[col]
        mean = sums[col] / c
        var = max(sq_sums[col] / c - mean * mean, 1e-12)
        state.numeric_means[col] = mean
        state.numeric_stds[col] = float(np.sqrt(var))

    for col, values in cat_values.items():
        uniq = sorted(values, key=lambda x: str(x))
        state.categorical_maps[col] = {v: i for i, v in enumerate(uniq)}

    state.fitted = True
    return state


def vectorize_event(features: dict[str, Any], state: PreprocessState) -> dict[str, float]:
    if not state.fitted:
        raise RuntimeError("PreprocessState must be fitted before vectorization")
    out: dict[str, float] = {}
    for col in state.feature_order:
        if col in state.numeric_means:
            val = float(features[col])
            mean = state.numeric_means[col]
            std = state.numeric_stds[col]
            out[col] = (val - mean) / std if std > 0 else 0.0
        else:
            mapping = state.categorical_maps[col]
            raw = features[col]
            out[col] = float(mapping.get(raw, len(mapping)))
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
