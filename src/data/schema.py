from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetSchema:
    feature_columns: list[str]
    timestamp_col: str
    label_col: str
    label_map: dict[str, int]
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    n_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "timestamp_col": self.timestamp_col,
            "label_col": self.label_col,
            "label_map": self.label_map,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "n_rows": self.n_rows,
        }
