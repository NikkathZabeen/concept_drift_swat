"""Write experiment metrics to InfluxDB."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def write_event_metrics(run_dir: Path, cfg_path: Path) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    if not cfg.get("influx", {}).get("enabled", False):
        logger.info("InfluxDB disabled in config; skipping write.")
        return

    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS

    influx = cfg["influx"]
    client = InfluxDBClient(
        url=influx["url"],
        token=influx["token"],
        org=influx["org"],
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for method_file, method in [
        ("baseline_events.csv", "baseline"),
        ("enhanced_events.csv", "enhanced"),
    ]:
        path = run_dir / method_file
        if not path.exists():
            continue
        df = pd.read_csv(path)
        points = []
        # Anchor series in the recent past so Grafana "Last 6 hours" includes all points (1 Hz).
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=max(len(df) - 1, 0))
        for i, (_, row) in enumerate(df.iterrows()):
            idx = int(row.get("event_index", 0))
            ts = start_time + timedelta(seconds=i)
            points.append(
                Point("swat_stream")
                .tag("method", method)
                .field("correct", int(row.get("correct", 0)))
                .field("y_true", int(row.get("y_true", 0)))
                .field("y_pred", int(row.get("y_pred", 0)))
                .field("y_score", float(row.get("y_score", 0.0)))
                .field("drift_detected", int(row.get("drift_detected", 0)))
                .field("latency_ms", float(row.get("latency_ms", 0.0)))
                .field("latency_total_ms", float(row.get("latency_total_ms", row.get("latency_ms", 0.0))))
                .field("event_index", idx)
                .time(ts, WritePrecision.NS)
            )
        write_api.write(bucket=influx["bucket"], org=influx["org"], record=points)
    client.close()
    logger.info("InfluxDB write complete for %s", run_dir)
