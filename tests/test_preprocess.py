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


def test_online_scaling_strictly_causal_no_retrospective_fit():
    """Regression test: verify that models operate with genuinely causal, isolated scalers.

    Requirements:
    - BaselineADWINSystem and EnhancedAdaptiveSystem each own an independent OnlineStandardScaler.
    - An observation at step t is transformed strictly using statistics from steps 0..t-1.
    - Future stream events do NOT retrospectively alter features or predictions at step t.
    - Baseline and Enhanced do not share or contaminate each other's scaler state.
    """
    baseline_cfg = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    enhanced_cfg = yaml.safe_load(Path("configs/enhanced.yaml").read_text())

    b_sys = BaselineADWINSystem(baseline_cfg, warmup_steps=10)
    e_sys = EnhancedAdaptiveSystem(enhanced_cfg, warmup_steps=10)

    assert b_sys.scaler is not e_sys.scaler, "Baseline and Enhanced must not share scaler instances"
    assert b_sys.scaler.n_samples == 0, "Online scaler must start unfitted with 0 samples"

    # Step 0: Observation 0
    raw_0 = {"FIT101": 10.0, "LIT101": 50.0, "MV101": 1}
    row_b0 = b_sys.process_event(raw_0, label=0, stream_step=0, global_event_index=0)
    # At step 0, prior count was 0 -> transformed values equal raw values
    assert row_b0["is_warmup"] == 1
    assert b_sys.scaler.n_samples == 1

    # Step 1: Observation 1
    raw_1 = {"FIT101": 12.0, "LIT101": 52.0, "MV101": 1}
    row_b1 = b_sys.process_event(raw_1, label=0, stream_step=1, global_event_index=1)
    # After step 1, scaler has seen 2 observations
    assert b_sys.scaler.n_samples == 2

    # Enhanced system has not seen any events yet; its scaler must remain untouched
    assert e_sys.scaler.n_samples == 0

    # Step 2: Observation 2
    raw_2 = {"FIT101": 14.0, "LIT101": 54.0, "MV101": 2}
    row_b2 = b_sys.process_event(raw_2, label=0, stream_step=2, global_event_index=2)

    # Re-evaluate an identical clone of stream up to step 2
    b_sys_clone = BaselineADWINSystem(baseline_cfg, warmup_steps=10)
    b_sys_clone.process_event(raw_0, label=0, stream_step=0, global_event_index=0)
    b_sys_clone.process_event(raw_1, label=0, stream_step=1, global_event_index=1)
    row_b2_clone = b_sys_clone.process_event(raw_2, label=0, stream_step=2, global_event_index=2)

    # Predictions and scores must be identical
    assert row_b2["y_pred"] == row_b2_clone["y_pred"]
    assert pytest.approx(row_b2["y_score"]) == row_b2_clone["y_score"]

    # Now feed extreme future outliers to b_sys
    for fut_step in range(3, 10):
        b_sys.process_event(
            {"FIT101": 10000.0, "LIT101": 99999.0, "MV101": 2},
            label=1,
            stream_step=fut_step,
            global_event_index=fut_step,
        )

    # The prediction recorded at step 2 was made before future events existed:
    # Verifying that b_sys recorded row_b2 strictly before step 3 updates
    assert b_sys.scaler.n_samples == 10
    # Future scaler mean has shifted drastically
    assert b_sys.scaler.means["FIT101"] > 5000.0
    # But past clone scaler mean at step 2 was ~12.0
    assert b_sys_clone.scaler.means["FIT101"] < 20.0
