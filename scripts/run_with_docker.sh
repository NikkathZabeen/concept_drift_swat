#!/usr/bin/env bash
# End-to-end run with Docker infra (Kafka, InfluxDB, Grafana).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
mkdir -p "$MPLCONFIGDIR" results/plots

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

echo "=== 1) docker compose up -d ==="
docker compose up -d

echo "=== 2) Wait for Kafka (9092) and Influx (8086) ==="
echo "Waiting for InfluxDB..."
for i in $(seq 1 60); do
  curl -sf http://localhost:8086/health >/dev/null 2>&1 && break
  sleep 2
done
echo "Waiting for Kafka..."
for i in $(seq 1 60); do
  if docker compose exec -T kafka kafka-broker-api-versions --bootstrap-server localhost:9092 >/dev/null 2>&1; then
    echo "Kafka ready"
    break
  fi
  sleep 3
done

RUN_DIR="results/runs/docker_$(date -u +%Y%m%dT%H%M%SZ)"
echo "=== 3) Prequential experiment (smoke) -> $RUN_DIR ==="
"$PY" run_project.py experiment --mode smoke --output "$RUN_DIR"

mkdir -p results/runs
( cd results/runs && rm -f latest && ln -s "$(basename "$RUN_DIR")" latest )

echo "=== 4) Kafka producer replay (25000 events, drift window) ==="
"$PY" -m src.streaming.stream_producer --config configs/project.yaml --max-events 25000 --kafka

echo "=== 5) InfluxDB metrics write ==="
"$PY" -c "
from pathlib import Path
from src.storage.influx_writer import write_event_metrics
write_event_metrics(Path('$RUN_DIR'), Path('configs/project.yaml'))
"

echo "=== 6) Plots ==="
MPLCONFIGDIR="$MPLCONFIGDIR" "$PY" run_project.py plot

echo "=== Done ==="
echo "Grafana: http://localhost:3000"
echo "Run dir: $RUN_DIR"
echo "Comparison: $RUN_DIR/comparison_table.csv"
