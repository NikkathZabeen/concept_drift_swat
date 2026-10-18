"""Tests for causal preprocessing, leakage prevention, and warm-up semantics."""

from pathlib import Path

import pytest
import yaml

from src.data.inspect_dataset import generate_synthetic_swat_csv
from src.data.preprocess import (
    OnlineStandardScaler,
    PreprocessState,
    fit_preprocess_state,
    iter_swat_events,
    vectorize_event,
)
from src.models.baseline import BaselineADWINSystem
from src.models.enhanced import EnhancedAdaptiveSystem


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "sample_swat.csv"
    generate_synthetic_swat_csv(csv_file, n_normal=200, n_attack=100, seed=42)
    return csv_file


@pytest.fixture
def project_cfg() -> dict:
    return yaml.safe_load(Path("configs/project.yaml").read_text())


def test_no_label_in_features(sample_csv: Path, project_cfg: dict):
    events = iter_swat_events(sample_csv, project_cfg, max_events=10)
    for ev in events:
        assert "Normal/Attack" not in ev["features"]
        assert "Timestamp" not in ev["features"]
        assert "global_event_index" in ev
        assert "stream_step" in ev


def test_preprocess_fit_causal(sample_csv: Path, project_cfg: dict):
    n_fit = 50
    events = iter_swat_events(sample_csv, project_cfg, max_events=n_fit)
    state = fit_preprocess_state(events, project_cfg, warmup_n=n_fit)
    assert state.fitted
    assert state.warmup_n == n_fit

    ev = next(iter_swat_events(sample_csv, project_cfg, max_events=1))
    vec = vectorize_event(ev["features"], state)
    assert len(vec) == len(state.feature_order)


def test_online_scaler_no_future_leakage():
    """Prove that OnlineStandardScaler transforms observation t using only information

    available prior to t, and future observations t+1..N cannot alter x_t.
    """
    scaler = OnlineStandardScaler()

    # Observation 0
    raw_0 = {"FIT101": 10.0, "LIT101": 20.0, "MV101": 1}
    vec_0 = scaler.transform_one(raw_0)
    # Before any updates, count < 2 -> no division by future std
    scaler.update_one(raw_0)

    # Observation 1
    raw_1 = {"FIT101": 12.0, "LIT101": 24.0, "MV101": 1}
    vec_1_initial = scaler.transform_one(raw_1)
    scaler.update_one(raw_1)

    # Observation 2
    raw_2 = {"FIT101": 14.0, "LIT101": 28.0, "MV101": 2}
    vec_2_before_future = scaler.transform_one(raw_2)

    # Re-evaluate transformation of observation 2 in isolation
    scaler_isolated = OnlineStandardScaler()
    scaler_isolated.update_one(raw_0)
    scaler_isolated.update_one(raw_1)
    vec_2_recomputed = scaler_isolated.transform_one(raw_2)

    # Must be exactly identical: future observations do not affect vec_2
    assert pytest.approx(vec_2_before_future["FIT101"]) == vec_2_recomputed["FIT101"]

    # Now add an extreme future outlier at observation 3
    raw_3_future = {"FIT101": 1000.0, "LIT101": 5000.0, "MV101": 1}
    scaler.update_one(raw_3_future)

    # Transforming raw_2 retrospectively with future knowledge WOULD change,
    # but the causal scaler recorded vec_2_before_future BEFORE raw_3 was ever observed!
    assert vec_2_before_future["FIT101"] != scaler.transform_one(raw_2)["FIT101"]


def test_warmup_semantics_smoke_mode_bug_fix(project_cfg: dict):
    """Prove the smoke-run warm-up bug is fixed:

    In smoke mode, starting at a large global index (e.g. 1,385,000),
    the detector must remain INACTIVE for the designated warm-up period.
    """
    baseline_cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    warmup_steps = 50

    system = BaselineADWINSystem(baseline_cfg, warmup_steps=warmup_steps)
    state = PreprocessState(
        numeric_means={"FIT101": 10.0},
        numeric_stds={"FIT101": 1.0},
        categorical_maps={},
        feature_order=["FIT101"],
        fitted=True,
    )

    smoke_start_global = 1385000

    # Stream steps 0 to 49 (global index 1385000 to 1385049) are within warm-up
    for step in range(warmup_steps):
        global_idx = smoke_start_global + step
        res = system.process_event(
            features_raw={"FIT101": 100.0},  # sudden jump
            label=1,
            stream_step=step,
            global_event_index=global_idx,
            state=state,
        )
        assert res["is_warmup"] == 1
        assert res["drift_detected"] == 0
        assert system.detector.n_drifts == 0

    # Stream step 50 is the first post-warmup step; detector is now ACTIVE
    res_active = system.process_event(
        features_raw={"FIT101": 100.0},
        label=0,  # prediction error
        stream_step=warmup_steps,
        global_event_index=smoke_start_global + warmup_steps,
        state=state,
    )
    assert res_active["is_warmup"] == 0


def test_warmup_semantics_enhanced_system():
    """Verify EnhancedAdaptiveSystem respects stream_step warm-up guard without labels."""
    enhanced_cfg = yaml.safe_load(Path("configs/enhanced.yaml").read_text())
    warmup_steps = 30
    system = EnhancedAdaptiveSystem(enhanced_cfg, warmup_steps=warmup_steps)
    state = PreprocessState(
        numeric_means={"FIT101": 10.0},
        numeric_stds={"FIT101": 1.0},
        categorical_maps={},
        feature_order=["FIT101"],
        fitted=True,
    )

    for step in range(warmup_steps):
        res = system.process_event(
            features_raw={"FIT101": 500.0},  # big feature shift
            label=0,
            stream_step=step,
            global_event_index=step + 10000,
            state=state,
        )
        assert res["is_warmup"] == 1
        assert res["drift_detected"] == 0
        assert system.detector.n_drifts == 0
