#!/usr/bin/env python3
"""Entry point for dataset inspection, experiments, plots, and optional infra."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="SWaT continual learning project runner")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="Inspect dataset and write schema JSON")
    exp = sub.add_parser("experiment", help="Run baseline vs enhanced experiment")
    exp.add_argument("--mode", default="smoke", help="smoke | full | <int max events>")
    exp.add_argument("--output", type=Path, default=None)
    sub.add_parser("plot", help="Generate plots from latest run")
    infra = sub.add_parser("infra", help="Start docker compose stack")
    infra.add_argument("action", choices=["up", "down"])
    demo = sub.add_parser("demo", help="Quick end-to-end local demo (no Kafka required)")
    demo.add_argument("--mode", default="smoke")

    args = parser.parse_args()

    if args.command == "inspect":
        from src.data.inspect_dataset import run_inspect

        run_inspect(Path("configs/project.yaml"), Path("results/dataset_inspection.json"))
        return

    if args.command == "experiment":
        from src.evaluation.experiment_runner import run_experiment

        out = args.output or Path(
            f"results/runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.mode}"
        )
        run_experiment(
            Path("configs/project.yaml"),
            Path("configs/baseline.yaml"),
            Path("configs/enhanced.yaml"),
            out,
            mode=args.mode,
        )
        latest = Path("results/runs/latest")
        latest.parent.mkdir(parents=True, exist_ok=True)
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out.resolve())
        print(f"Experiment output: {out}")
        return

    if args.command == "plot":
        from src.visualization.plot_results import plot_all

        run_dir = Path("results/runs/latest")
        plot_all(run_dir, Path("configs/project.yaml"), Path("results/plots"))
        return

    if args.command == "infra":
        cmd = ["docker", "compose", args.action, "-d"] if args.action == "up" else ["docker", "compose", args.action]
        subprocess.run(cmd, cwd=ROOT, check=False)
        return

    if args.command == "demo":
        from src.evaluation.experiment_runner import run_experiment
        from src.visualization.plot_results import plot_all

        out = Path(f"results/runs/demo_{args.mode}")
        run_experiment(
            Path("configs/project.yaml"),
            Path("configs/baseline.yaml"),
            Path("configs/enhanced.yaml"),
            out,
            mode=args.mode,
        )
        plot_all(out, Path("configs/project.yaml"), Path("results/plots"))
        print(f"Demo complete: {out}")


if __name__ == "__main__":
    main()
