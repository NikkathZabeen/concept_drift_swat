"""Spark Structured Streaming processor (requires Java + PySpark).

This module is optional. When Java is unavailable, use local simulation via
experiment_runner.py or kafka_consumer.py.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def java_available() -> bool:
    return shutil.which("java") is not None


def check_spark_ready() -> tuple[bool, str]:
    if not java_available():
        return False, "Java Runtime not found; Spark Structured Streaming cannot start."
    try:
        subprocess.run(["java", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return False, f"Java check failed: {exc}"
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False, "PySpark not installed."
    return True, "Spark prerequisites satisfied."


def run_spark_job(cfg: dict) -> None:
    ok, msg = check_spark_ready()
    if not ok:
        raise RuntimeError(msg)
    logger.info("Spark job placeholder — configure PySpark session and Kafka source in deployment.")
    # Full Spark pipeline is environment-dependent; Docker Compose service spark-worker can host this.
