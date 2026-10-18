"""Generate publication-style plots from experiment outputs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def plot_all(run_dir: Path, project_cfg: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(project_cfg.read_text())
    ref_idx = cfg["data"]["reference_drift_indices"][0]["stream_index"]

    b = pd.read_csv(run_dir / "baseline_events.csv")
    e = pd.read_csv(run_dir / "enhanced_events.csv")

    # Label distribution from event labels in run
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(b["y_true"]).value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_title("Label distribution in experiment stream")
    ax.set_xlabel("Label (0=Normal, 1=Attack)")
    ax.set_ylabel("Count")
    _save(fig, out_dir / "01_label_distribution")

    # Prequential F1
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(b["event_index"], _rolling(b, "correct", 1000), label="Baseline accuracy (1k window)", alpha=0.8)
    ax.plot(e["event_index"], _rolling(e, "correct", 1000), label="Enhanced accuracy (1k window)", alpha=0.8)
    ax.axvline(ref_idx, color="red", linestyle="--", label=f"Reference label boundary @ {ref_idx}")
    ax.set_title("Prequential accuracy over stream position")
    ax.set_xlabel("Event index")
    ax.legend()
    _save(fig, out_dir / "04_prequential_accuracy")

    # Drift timeline
    fig, ax = plt.subplots(figsize=(10, 3))
    b_drifts = b[b["drift_detected"] == 1]["event_index"]
    e_drifts = e[e["drift_detected"] == 1]["event_index"]
    ax.scatter(b_drifts, [1] * len(b_drifts), label="Baseline drifts", marker="|", s=200)
    ax.scatter(e_drifts, [2] * len(e_drifts), label="Enhanced drifts", marker="|", s=200)
    ax.axvline(ref_idx, color="red", linestyle="--", label="Reference boundary")
    ax.set_yticks([1, 2], ["Baseline", "Enhanced"])
    ax.set_xlabel("Event index")
    ax.set_title("Drift detection timeline")
    ax.legend(loc="upper right")
    _save(fig, out_dir / "06_drift_timeline")

    # Comparison bar chart
    comp = pd.read_csv(run_dir / "comparison_table.csv")
    numeric = comp.dropna(subset=["Difference (Enhanced - Baseline)"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(numeric["Metric"], numeric["Difference (Enhanced - Baseline)"], color="#55a868")
    ax.set_title("Enhanced − Baseline (selected metrics)")
    ax.set_xlabel("Difference")
    _save(fig, out_dir / "12_baseline_vs_enhancement_summary")

    # Sensor time series sample around drift
    _plot_sensor_context(run_dir, cfg, ref_idx, out_dir)

    logger.info("Plots saved to %s", out_dir)


def _plot_sensor_context(run_dir: Path, cfg: dict, ref_idx: int, out_dir: Path) -> None:
    """Optional: requires raw features not stored in event logs — use dataset slice."""
    archive = Path(cfg["data"]["archive_dir"]) / cfg["data"]["primary_file"]
    skip = max(1, ref_idx - 200)
    df = pd.read_csv(archive, skiprows=range(1, skip), nrows=400)
    df.columns = df.columns.str.strip()
    if "FIT101" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(df))
    colors = df["Normal/Attack"].astype(str).str.strip().map({"Normal": "#4c72b0", "Attack": "#c44e52"}).fillna("gray")
    ax.scatter(x, df["FIT101"], c=colors, s=8, alpha=0.7)
    ax.set_title("FIT101 around verified Normal→Attack boundary (context plot)")
    ax.set_xlabel("Relative index (window around boundary)")
    ax.set_ylabel("FIT101")
    _save(fig, out_dir / "02_sensor_timeseries_context")


def _rolling(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return df[col].rolling(window=window, min_periods=1).mean()


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("results/runs/latest"))
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--out", type=Path, default=Path("results/plots"))
    args = p.parse_args()
    plot_all(args.run_dir, args.config, args.out)


if __name__ == "__main__":
    main()
