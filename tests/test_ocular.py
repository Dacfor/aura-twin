"""Unit tests for the Ocular Telemetry Engine."""

import numpy as np
import pytest

from aura_face.config import OcularConfig
from aura_face.ocular import EyeClosureType, OcularEngine


@pytest.fixture
def ocular_engine() -> OcularEngine:
    config = OcularConfig(
        ema_alpha=0.6,
        blink_min_ms=100.0,
        blink_max_ms=400.0,
        droop_min_ms=400.0,
        droop_max_ms=500.0,
        microsleep_min_ms=500.0,
        perclos_window_s=60.0,
        metrics_window_s=300.0,
    )
    # ear_threshold = 0.20
    return OcularEngine(config=config, ear_threshold=0.20)


def test_ear_calculation_synthetic():
    """Verifies geometric EAR calculation formula."""
    # Synthetic eye: width = 20, vertical gap = 6
    # P1=(0,0), P2=(6, 3), P3=(14, 3), P4=(20, 0), P5=(14, -3), P6=(6, -3)
    pts = np.array([
        [0.0, 0.0],
        [6.0, 3.0],
        [14.0, 3.0],
        [20.0, 0.0],
        [14.0, -3.0],
        [6.0, -3.0],
    ])
    # ||P2-P6|| = 6, ||P3-P5|| = 6, ||P1-P4|| = 20
    # EAR = (6 + 6) / (2 * 20) = 12 / 40 = 0.30
    ear = OcularEngine.compute_single_ear(pts)
    assert pytest.approx(ear, rel=1e-3) == 0.30


def test_blink_classification(ocular_engine: OcularEngine):
    """Verifies that a 200 ms closure is correctly classified as a BLINK."""
    # Baseline open eye: EAR = 0.30
    t = 0.0
    telemetry = ocular_engine.process(0.30, t)
    assert not telemetry.is_eye_closed

    # Close eye for 200 ms (t=0.1 to t=0.3)
    t = 0.10
    ocular_engine.process(0.10, t)
    t = 0.20
    ocular_engine.process(0.10, t)
    t = 0.30
    # Eye reopens
    telemetry = ocular_engine.process(0.30, t)

    assert telemetry.last_event is not None
    assert telemetry.last_event.event_type == EyeClosureType.BLINK
    assert pytest.approx(telemetry.last_event.duration_ms, abs=10.0) == 200.0


def test_droop_classification_and_perclos(ocular_engine: OcularEngine):
    """Verifies that a 450 ms closure is classified as DROOP and contributes to PERCLOS."""
    t = 0.0
    ocular_engine.process(0.30, t)

    # Close eye for 450 ms (t=1.0 to t=1.45)
    t = 1.00
    ocular_engine.process(0.10, t)
    t = 1.25
    ocular_engine.process(0.10, t)
    t = 1.45
    telemetry = ocular_engine.process(0.30, t)

    assert telemetry.last_event is not None
    assert telemetry.last_event.event_type == EyeClosureType.DROOP
    assert pytest.approx(telemetry.last_event.duration_ms, abs=10.0) == 450.0

    # PERCLOS on 60s window should be (0.45s / 60s) * 100 = 0.75%
    assert telemetry.perclos > 0.5


def test_microsleep_classification(ocular_engine: OcularEngine):
    """Verifies that a closure >500 ms triggers MICROSLEEP."""
    t = 0.0
    ocular_engine.process(0.30, t)

    # Close eye for 700 ms (t=1.0 to t=1.70)
    ocular_engine.process(0.10, 1.0)
    ocular_engine.process(0.10, 1.3)
    ocular_engine.process(0.10, 1.5)
    telemetry = ocular_engine.process(0.30, 1.70)

    assert telemetry.last_event is not None
    assert telemetry.last_event.event_type == EyeClosureType.MICROSLEEP
    assert pytest.approx(telemetry.last_event.duration_ms, abs=10.0) == 700.0
    assert telemetry.microsleep_count_recent == 1


def test_noise_filtered(ocular_engine: OcularEngine):
    """Verifies that closures < 100 ms are marked as NOISE and do not count as blinks."""
    ocular_engine.process(0.30, 0.0)
    ocular_engine.process(0.10, 0.02)
    telemetry = ocular_engine.process(0.30, 0.06)  # 40 ms duration

    assert telemetry.last_event is not None
    assert telemetry.last_event.event_type == EyeClosureType.NOISE
    assert telemetry.blink_rate_bpm == 0.0
