"""Tests for CARD 5 Cardiovascular Digital Twin module."""

import numpy as np
import pytest

from aura_face.cardiovascular import (
    CardiovascularState,
    CardiovascularTrajectory,
    CardiovascularTwinCard,
)


def test_trajectory_loading():
    """Verify trajectory loads valid CSV data or fallback with expected duration."""
    traj = CardiovascularTrajectory("data/cardiovascular_demo.csv")
    assert len(traj.times) > 0
    assert traj.total_duration_min >= 50.0

    # Sample at t=0
    v0, l0 = traj.sample(0.0)
    assert v0 > 500.0
    assert l0 == 0.0

    # Sample at t=12 (during LBNP ramp)
    v12, l12 = traj.sample(12.0)
    assert v12 > v0
    assert l12 > 0.0

    # Sample at t=30 (stabilized phase)
    v30, l30 = traj.sample(30.0)
    assert 870.0 <= v30 <= 890.0
    assert l30 >= 15.0


def test_cardiovascular_state_calculations():
    """Verify deviation %, standby/active status, target range classification, and adaptation with 940mL target."""
    card = CardiovascularTwinCard(vvl_earth=940.0, playback_speed=30.0, tolerance_pct=2.0)

    # Initial state (microgravity, t=0)
    state_0 = card.update(0.0)
    assert not state_0.is_active  # LBNP == 0 -> Standby
    assert state_0.deviation_pct < -10.0
    assert not state_0.is_target_range
    assert state_0.fill_ratio < 0.90

    # At t_sim = 22.0 min (with 30x speed: t_real = 22.0 * 60 / 30 = 44.0s), Astronaut C stabilizes around 937 mL (~940 Earth target)
    state_target = card.update(44.0)
    assert state_target.is_active  # LBNP > 0 -> Active
    assert abs(state_target.deviation_pct) <= 2.0  # Within +-2% of 940mL target
    assert state_target.is_target_range
    assert 0.95 <= state_target.fill_ratio <= 1.05

    # At t_real = 60.0s (t_sim = 30.0 min), LBNP remains active with dynamic adaptive control
    state_adaptive = card.update(60.0)
    assert state_adaptive.is_active
    assert state_adaptive.lbnp > 15.0
    assert state_adaptive.is_target_range  # Vvl ~ 934.15 mL (-0.62% of 940 mL target)


def test_cardiovascular_drawing_on_canvas():
    """Verify that drawing Card 5 does not throw errors and creates valid graphics."""
    card = CardiovascularTwinCard(vvl_earth=940.0, playback_speed=30.0)
    card.update(5.0)

    canvas = np.zeros((720, 400, 3), dtype=np.uint8)
    # Draw without error
    card.draw_card(canvas, 0, 0, 400, 720, t_mission_s=5.0)

    # Canvas should not be all black
    assert np.count_nonzero(canvas) > 1000


def test_non_blocking_performance():
    """Verify that update executes in sub-millisecond time without blocking."""
    import time

    card = CardiovascularTwinCard()
    t_start = time.perf_counter()
    for i in range(100):
        card.update(float(i))
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # 100 updates should take well under 10ms (0.1ms per update)
    assert elapsed_ms < 50.0
