# PROJECT_STATUS — Continual Learning for Concept-Drifting SWaT Streams

**Last updated:** 2026-10-18 (local inspection + smoke/full experiments)

## Environment

| Component | Status |
|-----------|--------|
| Python | 3.14.6 (`.venv` created in repo) |
| Java | **Not installed** — Spark Structured Streaming not executed locally |
| Docker CLI | 29.6.1 installed; **Docker daemon not running** during implementation |
| Kafka / Influx / Grafana | Configured in `docker-compose.yml`; not verified live |
| River | 0.22.0 (online learning + supervised ADWIN) |
| PySpark | Not installed (optional; blocked without Java) |

## Workspace tree (excluding `.venv`, `.git`, large artifacts)

```
concept_drift_swat/
├── archive/                 # Original CSVs (unchanged)
├── sda_project/             # PDFs / proposal (unchanged)
├── configs/
├── dashboard/
├── reports/
├── results/
├── src/
├── tests/
├── docker-compose.yml
├── run_project.py
├── requirements.txt
├── README.md
├── DEMO_GUIDE.md
└── PROJECT_STATUS.md
```

## Dataset files

| File | Size | Rows (data) |
|------|------|-------------|
| `archive/normal.csv` | 384 MB | 1,387,098 |
| `archive/attack.csv` | 14 MB | 54,621 |
| `archive/merged.csv` | 407 MB | 1,441,719 |

Row counts match `wc -l` minus header (±0).

## Schema (53 columns)

- **Timestamp** — string, format `DD/MM/YYYY HH:MM:SS AM/PM` (strip whitespace before parsing).
- **Normal/Attack** — label column.
- **51 feature columns** — SWaT sensors (`FIT*`, `LIT*`, `AIT*`, `PIT*`, `DPIT*`) and actuators (`MV*`, `P*`, `UV*`).

### Label semantics (verified)

| File | Distinct labels |
|------|-----------------|
| `normal.csv` | `Normal` only |
| `attack.csv` | `Attack` only |
| `merged.csv` | `Normal`: 1,387,098; `Attack`: 54,621 |

Binary mapping used in code: `Normal→0`, `Attack→1`.

### Temporal structure of `merged.csv`

- **All Normal rows precede all Attack rows** (single transition at stream index **1,387,098**).
- Transition timestamp (raw): `28/12/2015 10:29:14 AM` (prior row: `9:59:59 AM` — **~29 minute gap**, not 1 Hz continuity).
- Timestamps parse correctly after `.str.strip()`; without strip, parse failures were observed.
- Minor non-monotonicity at chunk boundaries and small gap counts (>2s) exist in long Normal segments.

### Overlap across CSVs

- `merged` label counts equal `normal` + `attack` counts (1,387,098 + 54,621 = 1,441,719).
- **Not** a naive concatenation of file row order: `normal.csv` time range differs from the start of `merged.csv`.
- **Use `merged.csv` as the canonical chronological stream** for experiments.

## Research alignment (from uploaded PDFs)

### Paper A — ADWIN-U (Assis & Souza, 2025)

- **Question:** Unsupervised drift detection without label monitoring.
- **Method:** ADWIN on descriptive statistics (e.g., skewness, kurtosis) of incoming data.
- **Our use:** `src/drift/adwin_u.py` — bounded buffer statistics + River ADWIN monitors (implementation notes in module docstring).

### Paper B — SGBR (Gunasekara et al., 2025)

- **Scope:** Streaming **regression** with gradient-boosted bagging.
- **Our use:** Related work only; **not** used as the binary attack classifier.

### Paper C — CapyMOA (Gomes et al., 2025)

- **Scope:** Python stream ML framework (MOA integration).
- **Our use:** Informed choice of **River** for pure-Python prequential pipeline without Java.

## Implementation plan (executed)

1. ✅ Dataset inspection → `results/dataset_inspection.json`
2. ✅ Causal preprocessing + chronological iterator
3. ✅ Baseline: Hoeffding tree + supervised ADWIN on error (`baseline_adwin`)
4. ✅ Enhanced: ADWIN-U + ARF + buffer reset (`enhanced_adaptive`)
5. ✅ Experiment runner + metrics + comparison tables
6. ✅ Plots → `results/plots/`
7. ✅ pytest suite (8 tests)
8. ⚠️ Kafka/Spark/Influx/Grafana — code present; infra not running locally
9. 🔄 Full-stream experiment (`--mode full`) started in background

## Smoke experiment configuration

- Window: `smoke_start_index=1,385,000`, `smoke_max_events=25,000` (includes verified label boundary at 1,387,098).
- Results: `results/runs/latest/` (symlink to timestamped run).

## Known limitations

- Label boundary ≠ guaranteed **real concept drift** (attack period is ground truth for detection delay proxy).
- Enhanced ADWIN-U shows **higher false-alarm rate** than supervised ADWIN in smoke tests (honest trade-off).
- Full dataset run is compute-intensive (~hours on laptop).
