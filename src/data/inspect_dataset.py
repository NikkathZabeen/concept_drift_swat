"""Dataset inspection utilities for SWaT CSV files."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.schema import DatasetSchema

logger = logging.getLogger(__name__)

SWAT_SENSOR_PREFIXES = ("FIT", "LIT", "AIT", "PIT", "DPIT")
SWAT_ACTUATOR_PREFIXES = ("MV", "P", "UV")


def load_project_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def parse_timestamps(series: pd.Series, fmt: str) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    return pd.to_datetime(cleaned, format=fmt, errors="coerce")


def classify_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in columns:
        upper = col.upper()
        if any(upper.startswith(p) for p in SWAT_SENSOR_PREFIXES):
            numeric.append(col)
        elif any(upper.startswith(p) for p in SWAT_ACTUATOR_PREFIXES):
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical


def inspect_csv(path: Path, cfg: dict, sample_chunks: int = 3) -> dict:
    data_cfg = cfg["data"]
    ts_col = data_cfg["timestamp_col"]
    label_col = data_cfg["label_col"]
    ts_fmt = data_cfg["timestamp_format"]

    label_counts: Counter = Counter()
    n_rows = 0
    missing_total = 0
    dup_total = 0
    inf_total = 0
    ts_fail = 0
    ts_min = ts_max = None
    prev_ts = None
    non_monotonic = 0
    gaps_gt_2s = 0
    columns: list[str] | None = None

    for i, chunk in enumerate(pd.read_csv(path, chunksize=100_000)):
        chunk = normalize_columns(chunk)
        if columns is None:
            columns = list(chunk.columns)
        n_rows += len(chunk)
        missing_total += int(chunk.isna().sum().sum())
        dup_total += int(chunk.duplicated().sum())
        num = chunk.select_dtypes(include=[np.number])
        if len(num.columns):
            inf_total += int(np.isinf(num.to_numpy()).sum())

        if label_col in chunk.columns:
            label_counts.update(
                chunk[label_col].astype(str).str.strip().value_counts().to_dict()
            )

        if ts_col in chunk.columns:
            ts = parse_timestamps(chunk[ts_col], ts_fmt)
            ts_fail += int(ts.isna().sum())
            valid = ts.dropna()
            if len(valid):
                cmin, cmax = valid.iloc[0], valid.iloc[-1]
                ts_min = cmin if ts_min is None else min(ts_min, cmin)
                ts_max = cmax if ts_max is None else max(ts_max, cmax)
                if prev_ts is not None and valid.iloc[0] < prev_ts:
                    non_monotonic += 1
                diffs = valid.diff().dropna()
                gaps_gt_2s += int((diffs > pd.Timedelta(seconds=2)).sum())
                prev_ts = valid.iloc[-1]

        if i + 1 >= sample_chunks and sample_chunks > 0:
            # continue full scan for counts — sample_chunks only limits early dtype peek
            pass

    assert columns is not None
    feature_cols = [
        c
        for c in columns
        if c not in {ts_col, label_col}
    ]
    numeric_cols, categorical_cols = classify_columns(feature_cols)

    return {
        "path": str(path),
        "n_rows": n_rows,
        "n_columns": len(columns),
        "columns": columns,
        "label_counts": dict(label_counts),
        "missing_values": missing_total,
        "duplicate_rows_in_chunks": dup_total,
        "inf_values_in_numeric": inf_total,
        "timestamp_parse_failures": ts_fail,
        "timestamp_min": ts_min.isoformat() if ts_min is not None else None,
        "timestamp_max": ts_max.isoformat() if ts_max is not None else None,
        "non_monotonic_chunk_boundaries": non_monotonic,
        "gaps_gt_2s": gaps_gt_2s,
        "feature_columns": feature_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def find_label_transitions(path: Path, cfg: dict) -> list[dict]:
    label_col = cfg["data"]["label_col"]
    ts_col = cfg["data"]["timestamp_col"]
    transitions: list[dict] = []
    idx = 0
    prev: str | None = None
    for chunk in pd.read_csv(path, chunksize=200_000):
        chunk = normalize_columns(chunk)
        for _, row in chunk.iterrows():
            lbl = str(row[label_col]).strip()
            if prev is not None and lbl != prev:
                transitions.append(
                    {
                        "stream_index": idx,
                        "from": prev,
                        "to": lbl,
                        "timestamp_raw": str(row[ts_col]).strip(),
                    }
                )
            prev = lbl
            idx += 1
    return transitions


def build_schema(report: dict, cfg: dict) -> DatasetSchema:
    return DatasetSchema(
        feature_columns=report["feature_columns"],
        timestamp_col=cfg["data"]["timestamp_col"],
        label_col=cfg["data"]["label_col"],
        label_map=cfg["data"]["label_map"],
        numeric_columns=report["numeric_columns"],
        categorical_columns=report["categorical_columns"],
        n_rows=report["n_rows"],
    )


def run_inspect(config: Path, output: Path) -> None:
    cfg = load_project_config(config)
    archive = Path(cfg["data"]["archive_dir"])
    files = ["normal.csv", "attack.csv", "merged.csv"]
    reports = {}
    for name in files:
        path = archive / name
        logger.info("Inspecting %s", path)
        reports[name] = inspect_csv(path, cfg, sample_chunks=0)

    merged_path = archive / cfg["data"]["primary_file"]
    transitions = find_label_transitions(merged_path, cfg)
    reports["merged_label_transitions"] = transitions

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(reports, f, indent=2, default=str)

    schema = build_schema(reports["merged.csv"], cfg)
    schema_path = output.parent / "dataset_schema.json"
    with schema_path.open("w") as f:
        json.dump(schema.to_dict(), f, indent=2)

    logger.info("Wrote %s and %s", output, schema_path)


def generate_synthetic_swat_csv(
    output_path: Path,
    n_normal: int = 2000,
    n_attack: int = 4000,
    seed: int = 42,
    schema_path: Path | None = None,
) -> Path:
    """Generate a synthetic SWaT-format CSV conforming to the exact 53-column schema.

    Used for reproducible unit tests, regression verification, and offline demos
    when the 407 MB archive/merged.csv is not present locally.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    cols = [
        "Timestamp",
        "FIT101", "LIT101", "MV101", "P101", "P102",
        "AIT201", "AIT202", "AIT203", "FIT201", "MV201", "P201", "P202", "P203", "P204", "P205", "P206",
        "DPIT301", "FIT301", "LIT301", "MV301", "MV302", "MV303", "MV304", "P301", "P302",
        "AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402", "P403", "P404", "UV401",
        "AIT501", "AIT502", "AIT503", "AIT504", "FIT501", "FIT502", "FIT503", "FIT504", "P501", "P502",
        "PIT501", "PIT502", "PIT503", "FIT601", "P601", "P602", "P603",
        "Normal/Attack",
    ]
    numeric_cols, cat_cols = classify_columns([c for c in cols if c not in {"Timestamp", "Normal/Attack"}])

    total_rows = n_normal + n_attack
    start_time = pd.Timestamp("2015-12-28 10:00:00")
    timestamps = [
        (start_time + pd.Timedelta(seconds=i)).strftime("%d/%m/%Y %I:%M:%S %p")
        for i in range(total_rows)
    ]
    labels = ["Normal"] * n_normal + ["Attack"] * n_attack

    data: dict[str, list] = {"Timestamp": timestamps}
    for col in numeric_cols:
        # Normal phase: mean 10.0, std 1.0. Attack phase: distribution shifts (concept drift)
        normal_vals = rng.normal(loc=10.0, scale=1.0, size=n_normal)
        attack_vals = rng.normal(loc=18.0, scale=3.5, size=n_attack)
        data[col] = list(np.concatenate([normal_vals, attack_vals]))

    for col in cat_cols:
        normal_vals = rng.choice([1, 2], size=n_normal, p=[0.8, 0.2])
        attack_vals = rng.choice([1, 2], size=n_attack, p=[0.3, 0.7])
        data[col] = list(np.concatenate([normal_vals, attack_vals]))

    data["Normal/Attack"] = labels
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(output_path, index=False)
    logger.info("Generated synthetic SWaT CSV at %s (%d rows)", output_path, total_rows)
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Inspect SWaT CSV datasets")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/project.yaml"),
    )
    parser.add_argument("--output", type=Path, default=Path("results/dataset_inspection.json"))
    args = parser.parse_args()
    run_inspect(args.config, args.output)


if __name__ == "__main__":
    main()

