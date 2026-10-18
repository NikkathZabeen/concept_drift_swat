"""Run baseline and enhanced systems on the same chronological SWaT stream."""

from __future__ import annotations

import json
import logging
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import river
import yaml

from src.data.preprocess import fit_preprocess_state, iter_swat_events, save_preprocess_state
from src.evaluation.metrics import (
    classification_metrics,
    drift_detection_metrics,
    prequential_curve,
    recovery_time_events,
    streaming_perf,
)
from src.models.baseline import BaselineADWINSystem
from src.models.enhanced import EnhancedAdaptiveSystem

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def reference_indices(cfg: dict) -> list[int]:
    refs = cfg.get("data", {}).get("reference_drift_indices", [])
    return [int(r["stream_index"]) for r in refs]


def run_method(
    name: str,
    system,
    events,
    state,
    max_events: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    ru0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for ev in events:
        row = system.process_event(ev["features"], int(ev["label"]), ev["event_index"], state)
        rows.append(row)
        if max_events is not None and len(rows) >= max_events:
            break

    wall = time.perf_counter() - t0
    ru1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_kb = max(ru0, ru1)
    # macOS reports bytes; Linux KB — normalize approximately
    peak_mb = peak_rss_kb / (1024 * 1024) if peak_rss_kb > 10_000_000 else peak_rss_kb / 1024

    y_true = [r["y_true"] for r in rows]
    y_pred = [r["y_pred"] for r in rows]
    correct = [r["correct"] for r in rows]
    latencies = [r["latency_ms"] for r in rows]
    drift_idx = [r["event_index"] for r in rows if r.get("drift_detected")]

    summary: dict[str, Any] = {
        "method": name,
        "n_events": len(rows),
        "classification": classification_metrics(y_true, y_pred),
        "prequential_accuracy_curve": prequential_curve(correct, window=1000),
        "prequential_f1_curve": _rolling_f1(y_true, y_pred, window=1000),
        "drift_detection": drift_detection_metrics(
            drift_idx,
            reference_indices({}),  # filled by caller
            stream_length=len(rows),
        ),
        "streaming": streaming_perf(latencies, wall, len(rows)),
        "peak_memory_mb_approx": float(peak_mb),
        "n_model_resets": getattr(system, "n_resets", 0),
        "n_updates": getattr(system, "n_updates", 0),
        "detected_drift_indices": drift_idx,
        "wall_time_s": wall,
    }
    return rows, summary


def _rolling_f1(y_true: list[int], y_pred: list[int], window: int = 1000) -> list[float]:
    from sklearn.metrics import f1_score

    out = []
    for i in range(len(y_true)):
        s = max(0, i - window + 1)
        out.append(float(f1_score(y_true[s : i + 1], y_pred[s : i + 1], zero_division=0)))
    return out


def run_experiment(
    project_cfg_path: Path,
    baseline_cfg_path: Path,
    enhanced_cfg_path: Path,
    output_dir: Path,
    mode: str = "smoke",
) -> dict[str, Any]:
    project_cfg = load_yaml(project_cfg_path)
    baseline_cfg = load_yaml(baseline_cfg_path)
    enhanced_cfg = load_yaml(enhanced_cfg_path)

    seed = project_cfg["project"]["seed"]
    archive = Path(project_cfg["data"]["archive_dir"])
    csv_path = archive / project_cfg["data"]["primary_file"]

    start_index = 0
    if mode == "smoke":
        max_events = project_cfg["stream"].get("smoke_max_events", 80_000)
        start_index = int(project_cfg["stream"].get("smoke_start_index", 0))
    elif mode == "full":
        max_events = project_cfg["stream"].get("full_max_events")
    else:
        max_events = int(mode)

    logger.info("Running experiment mode=%s max_events=%s on %s", mode, max_events, csv_path)

    # Causal preprocess fit on initial segment
    fit_start = max(0, start_index - project_cfg["data"]["initial_train_size"])
    events_for_fit = iter_swat_events(
        csv_path,
        project_cfg,
        max_events=project_cfg["data"]["initial_train_size"],
        start_index=fit_start,
    )
    state = fit_preprocess_state(events_for_fit, project_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_preprocess_state(state, output_dir / "preprocess_state.json")

    # Baseline run
    baseline = BaselineADWINSystem(baseline_cfg, seed=seed, warmup=project_cfg["data"]["initial_train_size"])
    baseline_events = iter_swat_events(
        csv_path, project_cfg, max_events=max_events, start_index=start_index
    )
    b_rows, b_summary = run_method("baseline_adwin", baseline, baseline_events, state, max_events)

    # Enhanced run (same preprocess state, fresh stream)
    enhanced = EnhancedAdaptiveSystem(enhanced_cfg, seed=seed, warmup=project_cfg["data"]["initial_train_size"])
    e_events = iter_swat_events(
        csv_path, project_cfg, max_events=max_events, start_index=start_index
    )
    e_rows, e_summary = run_method("enhanced_adaptive", enhanced, e_events, state, max_events)

    refs = reference_indices(project_cfg)
    for summary in (b_summary, e_summary):
        summary["drift_detection"] = drift_detection_metrics(
            summary["detected_drift_indices"],
            refs,
            tolerance=500,
            stream_length=summary["n_events"],
        )
        if refs:
            summary["recovery"] = recovery_time_events(
                [r["correct"] for r in (b_rows if summary["method"] == "baseline_adwin" else e_rows)],
                refs[0],
            )

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "max_events": max_events,
        "start_index": start_index,
        "python": sys.version,
        "platform": platform.platform(),
        "river": river.__version__,
        "project_cfg": str(project_cfg_path),
        "reference_drift_indices": refs,
    }

    pd.DataFrame(b_rows).to_csv(output_dir / "baseline_events.csv", index=False)
    pd.DataFrame(e_rows).to_csv(output_dir / "enhanced_events.csv", index=False)

    with (output_dir / "baseline_summary.json").open("w") as f:
        json.dump(b_summary, f, indent=2)
    with (output_dir / "enhanced_summary.json").open("w") as f:
        json.dump(e_summary, f, indent=2)
    with (output_dir / "run_metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

    from src.evaluation.compare import build_comparison_table

    comparison = build_comparison_table(b_summary, e_summary)
    comparison.to_csv(output_dir / "comparison_table.csv", index=False)

    return {"baseline": b_summary, "enhanced": e_summary, "meta": meta, "output_dir": str(output_dir)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="smoke", help="smoke, full, or integer max_events")
    parser.add_argument("--output", type=Path, default=Path("results/runs/latest"))
    args = parser.parse_args()

    run_experiment(
        Path("configs/project.yaml"),
        Path("configs/baseline.yaml"),
        Path("configs/enhanced.yaml"),
        args.output,
        mode=args.mode,
    )
    logger.info("Experiment complete -> %s", args.output)


if __name__ == "__main__":
    main()
