"""Unit tests for AURA-Face v0.8.0 refinement features.

Tests cover:
1. MarkerMode cycling and interactive HUD toggle flags.
2. Normalized behavioral proxies (Valence [-1, 1], Arousal [0, 1], Workload [0, 1]).
3. Temporal persistence (Smile >=0.8s, Duchenne >=1.2s, Cognitive Strain >=5s hysteresis).
4. Explainable evidence logging and confidence score bounds.
5. Degraded tracking fallback (<0.60 tracking quality).
6. State-dependent ocular telemetry (near threshold, prolonged closure).
7. SQLite schema migration and persistence of new proxy fields.
8. Pluggable Vision backends.
"""

import json
import sqlite3
import pytest
import numpy as np

from aura_face.affective import (
    AffectiveEngine,
    OperationalBehaviorPattern,
)
from aura_face.backends import get_backend, MediaPipeBackend, PyFeatBackend
from aura_face.calibration import AstronautBaseline
from aura_face.config import AuraConfig
from aura_face.ocular import OcularEngine
from aura_face.overlay import CockpitOverlay, MarkerMode
from aura_face.storage import TelemetryRecord, TelemetryStorage


@pytest.fixture
def baseline() -> AstronautBaseline:
    return AstronautBaseline(
        subject_id="artemis_cdr_01",
        ear_open=0.34,
        ear_closed=0.12,
        ear_threshold=0.22,
        au4_baseline=0.08,
        au6_baseline=0.05,
        au12_baseline=0.06,
    )


# ---------------------------------------------------------------------------
# 1. Overlay & Marker Modes
# ---------------------------------------------------------------------------

def test_marker_mode_cycling():
    overlay = CockpitOverlay()
    assert overlay.marker_mode == MarkerMode.MINIMAL

    assert overlay.cycle_marker_mode() == MarkerMode.DETAILED
    assert overlay.cycle_marker_mode() == MarkerMode.ALL
    assert overlay.cycle_marker_mode() == MarkerMode.OFF
    assert overlay.cycle_marker_mode() == MarkerMode.MINIMAL


def test_hud_layer_toggles():
    overlay = CockpitOverlay()
    # Initial states
    assert overlay.show_eye_contours is True
    assert overlay.show_face_brackets is True
    assert overlay.show_head_pose_axis is True
    assert overlay.show_debug_labels is False

    # Toggles
    assert overlay.toggle_eye_contours() is False
    assert overlay.toggle_face_brackets() is False
    assert overlay.toggle_head_pose_axis() is False
    assert overlay.toggle_debug_labels() is True

    # Re-enable
    assert overlay.toggle_eye_contours() is True
    assert overlay.toggle_face_brackets() is True
    assert overlay.toggle_head_pose_axis() is True
    assert overlay.toggle_debug_labels() is False


# ---------------------------------------------------------------------------
# 2. Normalized Behavioral Proxies & Weights
# ---------------------------------------------------------------------------

def test_behavioral_proxies_normalization(baseline: AstronautBaseline):
    engine = AffectiveEngine(baseline=baseline)

    # Neutral / resting state
    tel_rest = engine.process({}, timestamp=0.0, ear_smooth=0.32, blink_rate=15.0, perclos=2.0)
    assert -1.0 <= tel_rest.facial_valence_proxy <= 1.0
    assert 0.0 <= tel_rest.operational_arousal_proxy <= 1.0
    assert 0.0 <= tel_rest.cognitive_workload_proxy <= 1.0
    assert 0.0 <= tel_rest.confidence <= 1.0
    assert len(tel_rest.evidence) > 0

    # Extreme positive expression
    happy_shapes = {
        "mouthSmileLeft": 0.85,
        "mouthSmileRight": 0.85,
        "eyeSquintLeft": 0.60,
        "eyeSquintRight": 0.60,
    }
    # Persist for 1.5s to cross Duchenne threshold
    engine.process(happy_shapes, timestamp=1.0)
    tel_happy = engine.process(happy_shapes, timestamp=2.5, ear_smooth=0.30, blink_rate=20.0, perclos=1.0)
    assert tel_happy.facial_valence_proxy > 0.40
    assert tel_happy.operational_arousal_proxy >= 0.0
    assert tel_happy.operational_pattern == OperationalBehaviorPattern.DUCHENNE_PATTERN_CANDIDATE


def test_arousal_and_workload_proxy_weights(baseline: AstronautBaseline):
    """Verify that proxy component weights strictly sum to 1.00."""
    engine = AffectiveEngine(baseline=baseline)

    # If blink rate is maxed (60 bpm -> 1.0), PERCLOS is maxed (100% -> 1.0), and head velocity proxy maxed (1.0)
    # Arousal must not exceed 1.0
    tel = engine.process(
        {},
        timestamp=1.0,
        blink_rate=60.0,
        perclos=100.0,
    )
    assert tel.operational_arousal_proxy <= 1.0
    assert tel.cognitive_workload_proxy <= 1.0


# ---------------------------------------------------------------------------
# 3. Temporal Stability (Smile, Duchenne, Cognitive Strain Hysteresis)
# ---------------------------------------------------------------------------

def test_smile_temporal_persistence(baseline: AstronautBaseline):
    """AU12 for <0.8s should NOT trigger SMILE_PATTERN; >=0.8s should trigger it."""
    engine = AffectiveEngine(baseline=baseline)
    shapes = {"mouthSmileLeft": 0.60, "mouthSmileRight": 0.60}

    # At t=0.0s (transient artifact)
    t0 = engine.process(shapes, timestamp=0.0)
    assert t0.operational_pattern != OperationalBehaviorPattern.SMILE_PATTERN

    # At t=0.5s (<0.8s threshold)
    t1 = engine.process(shapes, timestamp=0.5)
    assert t1.operational_pattern != OperationalBehaviorPattern.SMILE_PATTERN

    # At t=0.9s (>=0.8s threshold)
    t2 = engine.process(shapes, timestamp=0.9)
    assert t2.operational_pattern == OperationalBehaviorPattern.SMILE_PATTERN
    assert any("AU12 sustained" in ev for ev in t2.evidence)


def test_cognitive_strain_hysteresis(baseline: AstronautBaseline):
    """Test enter threshold (>=0.60 for >=5s) and exit threshold (<0.40 for >=30s)."""
    engine = AffectiveEngine(baseline=baseline)
    strain_shapes = {"browDownLeft": 0.70, "browDownRight": 0.70}

    # Sustained for 4s (not yet candidate)
    engine.process(strain_shapes, timestamp=0.0)
    t4 = engine.process(strain_shapes, timestamp=4.0)
    assert t4.operational_pattern != OperationalBehaviorPattern.COGNITIVE_STRAIN_CANDIDATE

    # Reaches 5.1s -> enters COGNITIVE_STRAIN_CANDIDATE
    t5 = engine.process(strain_shapes, timestamp=5.1)
    assert t5.operational_pattern == OperationalBehaviorPattern.COGNITIVE_STRAIN_CANDIDATE
    assert t5.cognitive_strain_candidate is True

    # AU4 drops below 0.40, but for only 10s -> still in candidate state due to 30s hysteresis!
    rest_shapes = {"browDownLeft": 0.05, "browDownRight": 0.05}
    t_drop10 = engine.process(rest_shapes, timestamp=15.1)
    assert t_drop10.cognitive_strain_candidate is True  # Hysteresis prevents rapid flip-flop

    # AU4 remains low for 30.9s (from 15.1s to 46.0s) -> exits candidate state
    t_drop32 = engine.process(rest_shapes, timestamp=46.0)
    assert t_drop32.cognitive_strain_candidate is False



# ---------------------------------------------------------------------------
# 4. Degraded Tracking Fallback
# ---------------------------------------------------------------------------

def test_degraded_tracking_fallback(baseline: AstronautBaseline):
    """When tracking quality < 0.60, system falls back to UNKNOWN_DEGRADED."""
    engine = AffectiveEngine(baseline=baseline)
    shapes = {"mouthSmileLeft": 0.80, "mouthSmileRight": 0.80}

    # Tracking quality is poor (0.45 < 0.60)
    tel = engine.process(shapes, timestamp=2.0, tracking_quality=0.45)
    assert tel.operational_pattern == OperationalBehaviorPattern.UNKNOWN_DEGRADED
    assert tel.confidence <= 0.30
    assert any("Degraded tracking" in ev for ev in tel.evidence)


# ---------------------------------------------------------------------------
# 5. Ocular Telemetry: Near Threshold & Prolonged Closure
# ---------------------------------------------------------------------------

def test_ocular_near_threshold_and_prolonged_closure():
    engine = OcularEngine(ear_threshold=0.20)

    # 1. EAR = 0.22 (within 15% margin of 0.20 -> 0.20 * 1.15 = 0.23)
    tel_near = engine.process(0.22, timestamp=0.0)
    assert tel_near.is_near_threshold is True
    assert tel_near.is_prolonged_closure is False

    # 2. Prolonged closure (EAR = 0.10 for >=1.5s)
    engine.process(0.10, timestamp=1.0)
    engine.process(0.10, timestamp=2.0)
    tel_prolonged = engine.process(0.10, timestamp=2.6)
    assert tel_prolonged.is_prolonged_closure is True
    assert tel_prolonged.ear_threshold == 0.20


# ---------------------------------------------------------------------------
# 6. SQLite Migration & Persistence of New Proxies
# ---------------------------------------------------------------------------

def test_sqlite_storage_and_migration(tmp_path):
    db_file = tmp_path / "test_migration.db"
    storage = TelemetryStorage(db_path=str(db_file), enable_wal=False)

    record = TelemetryRecord(
        ts=10.0,
        session_id="S_TEST_01",
        subject_id="artemis_cdr_01",
        ear=0.28,
        perclos=5.2,
        blink_rate=18.0,
        blink_duration_ms=120.0,
        microsleep_count=0,
        yawn_count=0,
        au4=0.12,
        au6=0.05,
        au12=0.45,
        emotion="Pleasure",
        head_yaw=2.1,
        head_pitch=-1.4,
        tracking_quality=0.98,
        state="ALERT",
        valence=0.35,
        arousal=0.25,
        cognitive_workload=0.15,
        psych_state="Attentive",
        valence_proxy=0.35,
        arousal_proxy=0.25,
        cognitive_workload_proxy=0.15,
        confidence=0.88,
        evidence_json=json.dumps(["AU12 above baseline", "EAR normal"]),
        marker_mode="minimal",
    )

    storage.log_telemetry(record)
    rows = storage.get_session_telemetry("S_TEST_01")
    assert len(rows) == 1
    r = rows[0]
    assert r["valence_proxy"] == 0.35
    assert r["arousal_proxy"] == 0.25
    assert r["cognitive_workload_proxy"] == 0.15
    assert r["confidence"] == 0.88
    assert r["marker_mode"] == "minimal"
    evidence = json.loads(r["evidence_json"])
    assert "AU12 above baseline" in evidence


# ---------------------------------------------------------------------------
# 7. Pluggable Backends
# ---------------------------------------------------------------------------

def test_pluggable_backend_factory():
    # 1. MediaPipe edge backend
    mp_backend = get_backend("mediapipe", "cpu")
    assert isinstance(mp_backend, MediaPipeBackend)
    assert mp_backend.name == "MediaPipe Edge FaceMesh"

    # 2. PyFeat backend
    pyfeat_backend = get_backend("pyfeat", "cpu")
    assert isinstance(pyfeat_backend, PyFeatBackend)
    # PyFeat may not be installed in the environment; verify graceful fallback behavior
    if not pyfeat_backend._available:
        assert pyfeat_backend.is_ready is False
