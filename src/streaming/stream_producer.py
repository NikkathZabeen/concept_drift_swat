"""Replay SWaT CSV rows to Kafka (or stdout in local mode)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import yaml

from src.data.preprocess import iter_swat_events

logger = logging.getLogger(__name__)


def produce(
    cfg_path: Path,
    max_events: int | None,
    use_kafka: bool,
    start_index: int | None = None,
) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    csv_path = Path(cfg["data"]["archive_dir"]) / cfg["data"]["primary_file"]
    rate = cfg["stream"].get("replay_rate_hz", 0)
    if start_index is None:
        start_index = int(cfg["stream"].get("smoke_start_index", 0))

    producer = None
    topic = cfg["kafka"]["topic"]
    if use_kafka:
        from kafka import KafkaProducer

        producer = KafkaProducer(
            bootstrap_servers=cfg["kafka"]["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

    n = 0
    for ev in iter_swat_events(
        csv_path, cfg, max_events=max_events, start_index=start_index
    ):
        payload = {
            "event_id": ev["event_id"],
            "event_index": ev["event_index"],
            "timestamp": ev["timestamp"],
            "timestamp_raw": ev["timestamp_raw"],
            "features": ev["features"],
            "label": ev["label"],
            "label_raw": ev["label_raw"],
        }
        if producer:
            producer.send(topic, payload)
        else:
            print(json.dumps({"topic": topic, "event": payload})[:200])
        n += 1
        if rate and rate > 0:
            time.sleep(1.0 / rate)

    if producer:
        producer.flush()
        producer.close()
    logger.info("Produced %s events (kafka=%s)", n, use_kafka)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--max-events", type=int, default=1000)
    p.add_argument("--kafka", action="store_true")
    p.add_argument("--start-index", type=int, default=None)
    args = p.parse_args()
    produce(args.config, args.max_events, args.kafka, start_index=args.start_index)


if __name__ == "__main__":
    main()
