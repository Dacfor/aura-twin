"""Psychophysiological Finite State Machine with Hysteresis for AURA-Face.

Manages high-level astronaut alertness states:
ALERT, MODERATE, CRITICAL, PRE_REST, UNKNOWN.

Implements temporal hysteresis to avoid rapid state oscillation ("chattering")
under borderline physiological conditions.

Scientific grounding:
- Schleicher et al. (2008): Multi-stage fatigue classification.
- Dinges & Grace (1998): PERCLOS psychophysiological validity thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from aura_face.config import GatingConfig, OcularConfig, StatesConfig


class AstronautState(str, Enum):
    ALERT = "ALERT"         # Nominal high-alertness operational status (PERCLOS < 8%)
    MODERATE = "MODERATE"   # Elevated drowsiness / fatigue (8% <= PERCLOS < 15%)
    CRITICAL = "CRITICAL"   # Severe vigilance impairment (PERCLOS >= 15% or >=2 microsleeps)
    PRE_REST = "PRE_REST"   # Cognitive exhaustion pattern (sustained AU4 > 3min)
    UNKNOWN = "UNKNOWN"     # Face absent or head pose out-of-gating for >= 10s


@dataclass
class StateTransitionEvent:
    from_state: AstronautState
    to_state: AstronautState
    timestamp: float
    trigger_reason: str


class AstronautStateMachine:
    """Manages alertness state evaluation with strict hysteresis gates."""

    def __init__(
        self,
        states_config: Optional[StatesConfig] = None,
        ocular_config: Optional[OcularConfig] = None,
        gating_config: Optional[GatingConfig] = None,
    ) -> None:
        self.states_config = states_config or StatesConfig()
        self.ocular_config = ocular_config or OcularConfig()
        self.gating_config = gating_config or GatingConfig()

        self.current_state = AstronautState.ALERT
        self.previous_valid_state = AstronautState.ALERT

        # Temporal timers for entry/exit delay
        self._moderate_candidate_start_ts: Optional[float] = None
        self._recovery_start_ts: Optional[float] = None
        self._unknown_candidate_start_ts: Optional[float] = None
        self._tracking_recovery_start_ts: Optional[float] = None

        self._last_transition_event: Optional[StateTransitionEvent] = None

    def update(
        self,
        perclos: float,
        microsleep_count_recent: int,
        tracking_quality: float,
        head_gated_out: bool,
        pre_rest_pattern: bool,
        timestamp: float,
    ) -> Tuple[AstronautState, Optional[StateTransitionEvent]]:
        """Evaluates inputs and transitions the state machine with hysteresis."""
        transition: Optional[StateTransitionEvent] = None

        # 1. Evaluate UNKNOWN condition (face loss or gating)
        tracking_bad = (tracking_quality < self.gating_config.min_tracking_quality) or head_gated_out
        if tracking_bad:
            self._tracking_recovery_start_ts = None
            if self._unknown_candidate_start_ts is None:
                self._unknown_candidate_start_ts = timestamp
            elif (timestamp - self._unknown_candidate_start_ts) >= self.gating_config.unknown_timeout_s:
                if self.current_state != AstronautState.UNKNOWN:
                    self.previous_valid_state = self.current_state
                    transition = self._transition_to(
                        AstronautState.UNKNOWN,
                        timestamp,
                        f"Face tracking lost or head gated out for >={self.gating_config.unknown_timeout_s}s",
                    )
                return self.current_state, transition
        else:
            self._unknown_candidate_start_ts = None
            if self.current_state == AstronautState.UNKNOWN:
                # Tracking recovery requires 10s continuous good tracking
                if self._tracking_recovery_start_ts is None:
                    self._tracking_recovery_start_ts = timestamp
                elif (timestamp - self._tracking_recovery_start_ts) >= self.gating_config.unknown_recovery_s:
                    transition = self._transition_to(
                        self.previous_valid_state,
                        timestamp,
                        f"Tracking recovered stably for >={self.gating_config.unknown_recovery_s}s",
                    )
                    self._tracking_recovery_start_ts = None
                return self.current_state, transition

        # 2. Check CRITICAL conditions (immediate entry, no delay)
        is_critical_perclos = perclos >= self.ocular_config.thresholds.perclos_moderate_max
        is_critical_microsleep = microsleep_count_recent >= self.ocular_config.thresholds.microsleep_critical_count

        if is_critical_perclos or is_critical_microsleep:
            self._recovery_start_ts = None
            self._moderate_candidate_start_ts = None
            if self.current_state != AstronautState.CRITICAL:
                reason = f"Critical fatigue trigger (PERCLOS={perclos:.1f}%, microsleeps={microsleep_count_recent})"
                transition = self._transition_to(AstronautState.CRITICAL, timestamp, reason)
            return self.current_state, transition

        # 3. If currently CRITICAL, test for Hysteresis Recovery to MODERATE or ALERT
        if self.current_state == AstronautState.CRITICAL:
            # Must remain below 15% and < 2 microsleeps for 30 consecutive seconds
            if self._recovery_start_ts is None:
                self._recovery_start_ts = timestamp
            elif (timestamp - self._recovery_start_ts) >= self.states_config.recovery_dwell_s:
                # Target de-escalation state
                if perclos >= self.ocular_config.thresholds.perclos_alert_max:
                    target = AstronautState.MODERATE
                    reason = f"De-escalation from CRITICAL to MODERATE after {self.states_config.recovery_dwell_s}s dwell"
                else:
                    target = AstronautState.ALERT
                    reason = f"De-escalation from CRITICAL to ALERT after {self.states_config.recovery_dwell_s}s dwell"
                transition = self._transition_to(target, timestamp, reason)
                self._recovery_start_ts = None
            return self.current_state, transition

        # 4. Check MODERATE condition (requires 10 seconds sustained)
        is_moderate = perclos >= self.ocular_config.thresholds.perclos_alert_max
        if is_moderate:
            if self._moderate_candidate_start_ts is None:
                self._moderate_candidate_start_ts = timestamp
            elif (timestamp - self._moderate_candidate_start_ts) >= self.states_config.moderate_trigger_delay_s:
                if self.current_state != AstronautState.MODERATE:
                    reason = f"Moderate fatigue sustained for >={self.states_config.moderate_trigger_delay_s}s (PERCLOS={perclos:.1f}%)"
                    transition = self._transition_to(AstronautState.MODERATE, timestamp, reason)
                self._recovery_start_ts = None
            return self.current_state, transition
        else:
            self._moderate_candidate_start_ts = None

        # 5. If currently MODERATE, test for Hysteresis Recovery to ALERT
        if self.current_state == AstronautState.MODERATE:
            if self._recovery_start_ts is None:
                self._recovery_start_ts = timestamp
            elif (timestamp - self._recovery_start_ts) >= self.states_config.recovery_dwell_s:
                reason = f"Recovery to ALERT after {self.states_config.recovery_dwell_s}s sustained below 8% PERCLOS"
                transition = self._transition_to(AstronautState.ALERT, timestamp, reason)
                self._recovery_start_ts = None
            return self.current_state, transition

        # 6. Check PRE_REST pattern
        if pre_rest_pattern and self.current_state == AstronautState.ALERT:
            transition = self._transition_to(
                AstronautState.PRE_REST,
                timestamp,
                "Pre-rest psychophysiological exhaustion pattern detected",
            )
            return self.current_state, transition
        elif not pre_rest_pattern and self.current_state == AstronautState.PRE_REST:
            transition = self._transition_to(
                AstronautState.ALERT,
                timestamp,
                "Pre-rest pattern resolved",
            )
            return self.current_state, transition

        # Default: remains ALERT
        return self.current_state, None

    def _transition_to(
        self,
        new_state: AstronautState,
        timestamp: float,
        reason: str,
    ) -> StateTransitionEvent:
        event = StateTransitionEvent(
            from_state=self.current_state,
            to_state=new_state,
            timestamp=timestamp,
            trigger_reason=reason,
        )
        self.current_state = new_state
        self._last_transition_event = event
        return event

    def reset(self) -> None:
        self.current_state = AstronautState.ALERT
        self.previous_valid_state = AstronautState.ALERT
        self._moderate_candidate_start_ts = None
        self._recovery_start_ts = None
        self._unknown_candidate_start_ts = None
        self._tracking_recovery_start_ts = None
        self._last_transition_event = None
