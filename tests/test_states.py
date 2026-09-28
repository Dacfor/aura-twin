"""Unit tests for the FSM alertness state machine and hysteresis."""

import pytest

from aura_face.states import AstronautState, AstronautStateMachine


@pytest.fixture
def state_machine() -> AstronautStateMachine:
    return AstronautStateMachine()


def test_initial_state_alert(state_machine: AstronautStateMachine):
    assert state_machine.current_state == AstronautState.ALERT


def test_immediate_critical_on_high_perclos(state_machine: AstronautStateMachine):
    """PERCLOS >= 15% immediately triggers CRITICAL state."""
    state, transition = state_machine.update(
        perclos=16.5,
        microsleep_count_recent=0,
        tracking_quality=1.0,
        head_gated_out=False,
        pre_rest_pattern=False,
        timestamp=1.0,
    )
    assert state == AstronautState.CRITICAL
    assert transition is not None
    assert transition.to_state == AstronautState.CRITICAL


def test_immediate_critical_on_two_microsleeps(state_machine: AstronautStateMachine):
    """2 microsleeps immediately trigger CRITICAL even if PERCLOS is low."""
    state, transition = state_machine.update(
        perclos=4.0,
        microsleep_count_recent=2,
        tracking_quality=1.0,
        head_gated_out=False,
        pre_rest_pattern=False,
        timestamp=1.0,
    )
    assert state == AstronautState.CRITICAL
    assert transition is not None


def test_moderate_requires_10s_delay(state_machine: AstronautStateMachine):
    """PERCLOS >= 8% must be sustained for 10s to trigger MODERATE."""
    # At t=0s, PERCLOS rises to 10%
    state, trans0 = state_machine.update(
        perclos=10.0,
        microsleep_count_recent=0,
        tracking_quality=1.0,
        head_gated_out=False,
        pre_rest_pattern=False,
        timestamp=0.0,
    )
    assert state == AstronautState.ALERT
    assert trans0 is None

    # At t=5s, still ALERT
    state, trans1 = state_machine.update(
        perclos=10.0,
        microsleep_count_recent=0,
        tracking_quality=1.0,
        head_gated_out=False,
        pre_rest_pattern=False,
        timestamp=5.0,
    )
    assert state == AstronautState.ALERT
    assert trans1 is None

    # At t=10.5s, triggers MODERATE
    state, trans2 = state_machine.update(
        perclos=10.0,
        microsleep_count_recent=0,
        tracking_quality=1.0,
        head_gated_out=False,
        pre_rest_pattern=False,
        timestamp=10.5,
    )
    assert state == AstronautState.MODERATE
    assert trans2 is not None
    assert trans2.to_state == AstronautState.MODERATE


def test_hysteresis_recovery_dwell(state_machine: AstronautStateMachine):
    """Recovering from CRITICAL to ALERT requires 30 seconds below threshold."""
    # Trigger CRITICAL at t=0
    state_machine.update(
        perclos=20.0,
        microsleep_count_recent=0,
        tracking_quality=1.0,
        head_gated_out=False,
        pre_rest_pattern=False,
        timestamp=0.0,
    )
    assert state_machine.current_state == AstronautState.CRITICAL

    # Eyes reopen, PERCLOS drops to 2.0% at t=1.0
    state, _ = state_machine.update(
        perclos=2.0,
        microsleep_count_recent=0,
        tracking_quality=1.0,
        head_gated_out=False,
        pre_rest_pattern=False,
        timestamp=1.0,
    )
    assert state == AstronautState.CRITICAL  # Dwell timer begins

    # At t=25.0 (24s elapsed), must still be CRITICAL
    state, _ = state_machine.update(
        perclos=2.0,
        microsleep_count_recent=0,
        tracking_quality=1.0,
        head_gated_out=False,
        pre_rest_pattern=False,
        timestamp=25.0,
    )
    assert state == AstronautState.CRITICAL

    # At t=32.0 (31s elapsed > 30s dwell), successfully recovers to ALERT
    state, trans = state_machine.update(
        perclos=2.0,
        microsleep_count_recent=0,
        tracking_quality=1.0,
        head_gated_out=False,
        pre_rest_pattern=False,
        timestamp=32.0,
    )
    assert state == AstronautState.ALERT
    assert trans is not None
    assert trans.to_state == AstronautState.ALERT
