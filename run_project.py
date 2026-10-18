#!/usr/bin/env python3
"""Entry point for dataset inspection, experiments, plots, comparison, and infra."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def update_latest_link(run_dir: Path, latest_link: Path = Path("results/runs/latest")) -> None:
    """Create or update pointer to latest run directory portably across OSes."""
    latest_link.parent.mkdir(parents=True, exist_ok=True)
    if latest_link.exists() or latest_link.is_symlink():
        try:
            if latest_link.is_symlink() or latest_link.is_file():
                latest_link.unlink()
            elif latest_link.is_dir():
                # On Windows directory junctions may appear as dirs
                try:
                    os.rmdir(latest_link)
                except OSError:
                    shutil.rmtree(latest_link)
        except Exception as exc:
            logger.warning("Could not unlink existing %s: %s", latest_link, exc)

    # 1. Try standard symlink
    try:
        latest_link.symlink_to(run_dir.resolve(), target_is_directory=True)
        return
    except OSError:
        pass

    # 2. On Windows, try Directory Junction
    if sys.platform == "win32":
        try:
            import _winapi

            _winapi.CreateJunction(str(run_dir.resolve()), str(latest_link.resolve()))
            return
        except Exception:
            pass

    # 3. Fallback: Write pointer file and directory copy of summaries
    pointer = latest_link.parent / "latest_run_path.txt"
    pointer.write_text(str(run_dir.resolve()), encoding="utf-8")
    try:
        shutil.copytree(run_dir, latest_link, dirs_exist_ok=True)
    except Exception as exc:
        logger.warning("Could not mirror directory to latest: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="SWaT continual learning project runner")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inspect", help="Inspect dataset and write schema JSON")

    exp = sub.add_parser("experiment", help="Run baseline vs enhanced experiment")
    exp.add_argument("--mode", default="smoke", help="smoke | full | <int max events>")
    exp.add_argument("--output", type=Path, default=None)

    comp = sub.add_parser("compare", help="Compare baseline vs enhanced from saved summary files")
    comp.add_argument("--run-dir", type=Path, default=Path("results/runs/latest"))
    comp.add_argument("--baseline", type=Path, default=None)
    comp.add_argument("--enhanced", type=Path, default=None)
    comp.add_argument("--output", type=Path, default=None)

    plt_cmd = sub.add_parser("plot", help="Generate plots from latest or specified run")
    plt_cmd.add_argument("--run-dir", type=Path, default=Path("results/runs/latest"))
    plt_cmd.add_argument("--output", type=Path, default=Path("results/plots"))

    infra = sub.add_parser("infra", help="Manage docker compose stack")
    infra.add_argument("action", choices=["up", "down", "status"])

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
        res = run_experiment(
            Path("configs/project.yaml"),
            Path("configs/baseline.yaml"),
            Path("configs/enhanced.yaml"),
            out,
            mode=args.mode,
        )
        update_latest_link(out)
        print(f"Experiment output saved to: {out}")
        return

    if args.command == "compare":
        from src.evaluation.compare import build_comparison_table

        run_dir = args.run_dir
        b_path = args.baseline or (run_dir / "baseline_summary.json")
        e_path = args.enhanced or (run_dir / "enhanced_summary.json")

        if not b_path.exists() or not e_path.exists():
            print(f"Error: Summary files not found at {b_path} or {e_path}")
            sys.exit(1)

        b_summary = json.loads(b_path.read_text(encoding="utf-8"))
        e_summary = json.loads(e_path.read_text(encoding="utf-8"))
        tbl = build_comparison_table(b_summary, e_summary)

        out_path = args.output or (run_dir / "comparison_table.csv")
        tbl.to_csv(out_path, index=False)
        print(tbl.to_string(index=False))
        print(f"\nWrote comparison table to {out_path}")
        return

    if args.command == "plot":
        from src.visualization.plot_results import plot_all

        plot_all(args.run_dir, Path("configs/project.yaml"), args.output)
        return

    if args.command == "infra":
        if args.action == "status":
            subprocess.run(["docker", "compose", "ps"], cwd=ROOT, check=False)
        else:
            cmd = (
                ["docker", "compose", args.action, "-d"]
                if args.action == "up"
                else ["docker", "compose", args.action]
            )
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
        update_latest_link(out)
        plot_all(out, Path("configs/project.yaml"), Path("results/plots"))
        print(f"Demo complete: {out}")


if __name__ == "__main__":
    main()
