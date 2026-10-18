"""Run baseline and enhanced systems on the same chronological SWaT stream."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import river
import sklearn
import yaml

from src.data.inspect_dataset import generate_synthetic_swat_csv
from src.data.preprocess import (
    OnlineStandardScaler,
    PreprocessState,
    fit_preprocess_state,
    iter_swat_events,
    save_preprocess_state,
)
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


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def run_method(
    name: str,
    system: BaselineADWINSystem | EnhancedAdaptiveSystem,
    events,
    state: PreprocessState | OnlineStandardScaler,
    max_events: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    proc = psutil.Process()
    mem_before_mb = proc.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()

    for ev in events:
        step = ev["stream_step"]
        global_idx = ev["global_event_index"]
        label = int(ev["label"])
        row = system.process_event(ev["features"], label, step, global_idx, state)
        rows.append(row)
        if max_events is not None and len(rows) >= max_events:
            break

    wall = time.perf_counter() - t0
    mem_after_mb = proc.memory_info().rss / (1024 * 1024)
    peak_mb = max(mem_before_mb, mem_after_mb)

    # Separate scored events from warm-up prefix events
    scored_rows = [r for r in rows if r["is_warmup"] == 0]
    warmup_rows = [r for r in rows if r["is_warmup"] == 1]

    # Evaluate classification metrics strictly on scored observations
    eval_target_rows = scored_rows if scored_rows else rows
    y_true = [r["y_true"] for r in eval_target_rows]
    y_pred = [r["y_pred"] for r in eval_target_rows]
    y_score = [r["y_score"] for r in eval_target_rows]
    correct = [r["correct"] for r in eval_target_rows]

    lat_total = [r["latency_total_ms"] for r in rows]
    lat_pred = [r["latency_pred_ms"] for r in rows]
    lat_det = [r["latency_detector_ms"] for r in rows]
    lat_learn = [r["latency_learn_ms"] for r in rows]

    drift_idx = [r["event_index"] for r in rows if r.get("drift_detected")]
    raw_drift_idx = getattr(system.detector, "raw_alarm_indices", drift_idx)

    summary: dict[str, Any] = {
        "method": name,
        "n_events_total": len(rows),
        "n_events_warmup": len(warmup_rows),
        "n_events_scored": len(scored_rows),
        "n_events": len(rows),
        "classification": classification_metrics(y_true, y_pred, y_score),
        "prequential_accuracy_curve": prequential_curve(correct, window=1000),
        "prequential_f1_curve": _rolling_f1(y_true, y_pred, window=1000),
        "drift_detection": drift_detection_metrics(
            drift_idx,
            reference_indices({}),  # filled by caller with cfg refs
            stream_length=len(rows),
            raw_detected_indices=raw_drift_idx,
        ),
        "streaming": streaming_perf(
            lat_total,
            wall,
            len(rows),
            latencies_pred_ms=lat_pred,
            latencies_det_ms=lat_det,
            latencies_learn_ms=lat_learn,
        ),
        "peak_memory_mb_approx": float(peak_mb),
        "n_model_resets": getattr(system, "n_resets", 0),
        "n_updates": getattr(system, "n_updates", 0),
        "detected_drift_indices": drift_idx,
        "raw_drift_indices": raw_drift_idx,
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

    # Verify dataset file or fallback to synthetic test fixture with explicit provenance
    dataset_provenance = "canonical_swat_archive"
    if not csv_path.exists():
        logger.warning(
            "Dataset %s not found. Generating synthetic SWaT stream for verification.",
            csv_path,
        )
        generate_synthetic_swat_csv(
            csv_path,
            n_normal=project_cfg["data"].get("initial_train_size", 5000) + 5000,
            n_attack=3000,
            seed=seed,
        )
        dataset_provenance = "synthetic_fixture_verification"

    start_index = 0
    configured_warmup = int(project_cfg["data"].get("warmup_min_labeled", 5000))

    if dataset_provenance == "synthetic_fixture_verification":
        start_index = 0
        max_events = 4500 if mode == "smoke" else 8000
        warmup_steps = 1000
        refs = [3000]
    elif mode == "smoke":
        max_events = project_cfg["stream"].get("smoke_max_events", 25000)
        start_index = int(project_cfg["stream"].get("smoke_start_index", 0))
        warmup_steps = min(configured_warmup, 2000)
        refs = reference_indices(project_cfg)
    elif mode == "full":
        max_events = project_cfg["stream"].get("full_max_events")
        start_index = 0
        warmup_steps = configured_warmup
        refs = reference_indices(project_cfg)
    else:
        max_events = int(mode)
        start_index = 0
        warmup_steps = min(configured_warmup, max(100, max_events // 5))
        refs = reference_indices(project_cfg)

    logger.info(
        "Running experiment mode=%s max_events=%s warmup_steps=%s start_index=%s on %s (provenance=%s)",
        mode,
        max_events,
        warmup_steps,
        start_index,
        csv_path,
        dataset_provenance,
    )

    # Causal preprocessing: fit strictly on the initial warm-up prefix
    events_for_fit = iter_swat_events(
        csv_path,
        project_cfg,
        max_events=warmup_steps,
        start_index=start_index,
    )
    state = fit_preprocess_state(events_for_fit, project_cfg, warmup_n=warmup_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_preprocess_state(state, output_dir / "preprocess_state.json")

    # Baseline run
    baseline = BaselineADWINSystem(
        baseline_cfg,
        seed=seed,
        warmup_steps=warmup_steps,
    )
    baseline_events = iter_swat_events(
        csv_path, project_cfg, max_events=max_events, start_index=start_index
    )
    b_rows, b_summary = run_method("baseline_adwin", baseline, baseline_events, state, max_events)

    # Enhanced run (same preprocess state, fresh stream)
    enhanced = EnhancedAdaptiveSystem(
        enhanced_cfg,
        seed=seed,
        warmup_steps=warmup_steps,
    )
    e_events = iter_swat_events(
        csv_path, project_cfg, max_events=max_events, start_index=start_index
    )
    e_rows, e_summary = run_method("enhanced_adaptive", enhanced, e_events, state, max_events)

    refs = reference_indices(project_cfg)
    for summary in (b_summary, e_summary):
        method_rows = b_rows if summary["method"] == "baseline_adwin" else e_rows
        summary["drift_detection"] = drift_detection_metrics(
            summary["detected_drift_indices"],
            refs,
            tolerance=500,
            stream_length=summary["n_events_total"],
            raw_detected_indices=summary.get("raw_drift_indices"),
        )
        if refs:
            # Recovery measured around first reference drift
            summary["recovery"] = recovery_time_events(
                [r["correct"] for r in method_rows],
                refs[0] if refs[0] < len(method_rows) else max(0, len(method_rows) // 2),
            )

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "mode": mode,
        "dataset_provenance": dataset_provenance,
        "max_events": max_events,
        "start_index": start_index,
        "warmup_steps": warmup_steps,
        "n_events_scored": b_summary["n_events_scored"],
        "python": sys.version,
        "platform": platform.platform(),
        "river": river.__version__,
        "sklearn": sklearn.__version__,
        "project_cfg": str(project_cfg_path),
        "reference_drift_indices": refs,
        "status": "COMPLETED",
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
