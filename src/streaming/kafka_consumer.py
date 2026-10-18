"""Consume SWaT events from Kafka and run online learner (local integration path)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml
from kafka import KafkaConsumer

from src.data.preprocess import load_preprocess_state
from src.models.enhanced import EnhancedAdaptiveSystem

logger = logging.getLogger(__name__)


def consume(cfg_path: Path, preprocess_path: Path, max_events: int) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    enhanced_cfg = yaml.safe_load(Path("configs/enhanced.yaml").read_text())
    state = load_preprocess_state(preprocess_path)
    system = EnhancedAdaptiveSystem(enhanced_cfg, warmup_steps=cfg["data"]["initial_train_size"])

    consumer = KafkaConsumer(
        cfg["kafka"]["topic"],
        bootstrap_servers=cfg["kafka"]["bootstrap_servers"],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        consumer_timeout_ms=5000,
    )

    n = 0
    for msg in consumer:
        ev = msg.value
        system.process_event(
            ev["features"],
            int(ev["label"]),
            stream_step=n,
            global_event_index=int(ev.get("event_index", n)),
            state=state,
        )
        n += 1
        if n >= max_events:
            break
    logger.info("Consumed %s events", n)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--preprocess", type=Path, default=Path("results/runs/latest/preprocess_state.json"))
    p.add_argument("--max-events", type=int, default=5000)
    args = p.parse_args()
    consume(args.config, args.preprocess, args.max_events)


if __name__ == "__main__":
    main()
