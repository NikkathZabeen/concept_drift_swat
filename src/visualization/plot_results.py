"""Generate publication-style plots from experiment outputs."""

from __future__ import annotations

import argparse
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
    refs = cfg["data"].get("reference_drift_indices", [])
    ref_idx = refs[0]["stream_index"] if refs else None

    b_path = run_dir / "baseline_events.csv"
    e_path = run_dir / "enhanced_events.csv"

    if not b_path.exists() or not e_path.exists():
        logger.warning("Event logs not found in %s; cannot generate event-level plots.", run_dir)
        return

    b = pd.read_csv(b_path)
    e = pd.read_csv(e_path)

    # 1. Label distribution from event labels in run
    fig, ax = plt.subplots(figsize=(6, 4))
    b["y_true"].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4c72b0", "#dd8452"])
    ax.set_title("Label Distribution in Evaluated Stream")
    ax.set_xlabel("Label (0=Normal, 1=Attack)")
    ax.set_ylabel("Count")
    _save(fig, out_dir / "01_label_distribution")

    # 2. Prequential Accuracy Curve
    fig, ax = plt.subplots(figsize=(10, 4))
    window = min(1000, max(50, len(b) // 10))
    ax.plot(b["event_index"], _rolling(b, "correct", window), label=f"Baseline accuracy ({window} window)", alpha=0.8)
    ax.plot(e["event_index"], _rolling(e, "correct", window), label=f"Enhanced accuracy ({window} window)", alpha=0.8)
    if ref_idx is not None and b["event_index"].min() <= ref_idx <= b["event_index"].max():
        ax.axvline(ref_idx, color="red", linestyle="--", label=f"Reference label boundary @ {ref_idx}")
    ax.set_title("Prequential Accuracy Over Stream Position")
    ax.set_xlabel("Event Index")
    ax.set_ylabel("Rolling Accuracy")
    ax.legend()
    _save(fig, out_dir / "04_prequential_accuracy")

    # 3. Drift Timeline
    fig, ax = plt.subplots(figsize=(10, 3))
    b_drifts = b[b["drift_detected"] == 1]["event_index"]
    e_drifts = e[e["drift_detected"] == 1]["event_index"]
    if len(b_drifts):
        ax.scatter(b_drifts, [1] * len(b_drifts), label=f"Baseline drifts (n={len(b_drifts)})", marker="|", s=200, color="#4c72b0")
    if len(e_drifts):
        ax.scatter(e_drifts, [2] * len(e_drifts), label=f"Enhanced drifts (n={len(e_drifts)})", marker="|", s=200, color="#dd8452")
    if ref_idx is not None and b["event_index"].min() <= ref_idx <= b["event_index"].max():
        ax.axvline(ref_idx, color="red", linestyle="--", label="Reference boundary")
    ax.set_yticks([1, 2], ["Baseline", "Enhanced"])
    ax.set_xlabel("Event Index")
    ax.set_title("Drift Detection Timeline")
    ax.legend(loc="upper right")
    _save(fig, out_dir / "06_drift_timeline")

    # 4. Comparison Bar Chart
    comp_path = run_dir / "comparison_table.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        comp["Difference (Enhanced - Baseline)"] = pd.to_numeric(comp["Difference (Enhanced - Baseline)"], errors="coerce")
        numeric = comp.dropna(subset=["Difference (Enhanced - Baseline)"])
        # Filter to key presentation metrics
        key_metrics = [
            "Accuracy",
            "Precision (Attack)",
            "Recall (Attack)",
            "F1 (Attack)",
            "ROC-AUC",
            "Detected drifts (count)",
            "False alarms",
            "Model resets",
        ]
        plot_comp = numeric[numeric["Metric"].isin(key_metrics)]
        if not plot_comp.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#55a868" if v >= 0 else "#c44e52" for v in plot_comp["Difference (Enhanced - Baseline)"]]
            ax.barh(plot_comp["Metric"], plot_comp["Difference (Enhanced - Baseline)"], color=colors)
            ax.set_title("Enhanced − Baseline (Selected Metrics)")
            ax.set_xlabel("Difference (Enhanced minus Baseline)")
            _save(fig, out_dir / "12_baseline_vs_enhancement_summary")

    # 5. Sensor Context Plot
    if ref_idx is not None:
        _plot_sensor_context(cfg, ref_idx, out_dir)

    logger.info("Plots saved to %s", out_dir)


def _plot_sensor_context(cfg: dict, ref_idx: int, out_dir: Path) -> None:
    archive = Path(cfg["data"]["archive_dir"]) / cfg["data"]["primary_file"]
    if not archive.exists():
        return

    try:
        skip = max(1, ref_idx - 200)
        df = pd.read_csv(archive, skiprows=range(1, skip), nrows=400)
        df.columns = df.columns.str.strip()
        if "FIT101" not in df.columns or "Normal/Attack" not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(df))
        colors = df["Normal/Attack"].astype(str).str.strip().map({"Normal": "#4c72b0", "Attack": "#c44e52"}).fillna("gray")
        ax.scatter(x, df["FIT101"], c=colors, s=8, alpha=0.7)
        ax.set_title("FIT101 Sensor Stream Context Around Reference Boundary")
        ax.set_xlabel("Relative Observation Position in Window")
        ax.set_ylabel("FIT101 Sensor Value")
        _save(fig, out_dir / "02_sensor_timeseries_context")
    except Exception as exc:
        logger.warning("Could not generate sensor context plot: %s", exc)


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
