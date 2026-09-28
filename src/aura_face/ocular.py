"""Ocular telemetry engine for AURA-Face.

Implements bilateral Eye Aspect Ratio (EAR), Exponential Moving Average (EMA) filtering,
timestamp-based eyelid closure event classification (Blink, Droop, Microsleep),
and PERCLOS calculation over sliding temporal windows.

Scientific references:
- Wierwille, W. W. et al. (1994): Definition of PERCLOS (% time slow eyelid closures).
- Dinges, D. F. & Grace, R. (1998): Psychophysiological validation of PERCLOS.
- Schleicher, R. et al. (2008): Blink duration and slow droops as primary drowsiness markers.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

import numpy as np

from aura_face.config import OcularConfig


class EyeClosureType(str, Enum):
    """Classification of eyelid closure duration."""
    NOISE = "NOISE"           # < 100 ms (flicker or tracking artifact)
    BLINK = "BLINK"           # 100 - 400 ms (normal physiological blink)
    DROOP = "DROOP"           # 400 - 500 ms (slow eyelid droop, counts for PERCLOS)
    MICROSLEEP = "MICROSLEEP" # >= 500 ms (critical involuntary lapse of alertness)


@dataclass
class EyeClosureEvent:
    """Represents a discrete eyelid closure event."""
    start_ts: float
    end_ts: float
    duration_ms: float
    event_type: EyeClosureType


@dataclass
class OcularTelemetry:
    """Current snapshot of ocular psychophysiological metrics."""
    timestamp: float
    ear_raw: float
    ear_smooth: float
    is_eye_closed: bool
    current_closure_duration_ms: float
    perclos: float
    blink_rate_bpm: float
    blink_duration_median_ms: float
    microsleep_count_recent: int
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_threshold: float = 0.20
    is_near_threshold: bool = False
    is_prolonged_closure: bool = False
    perclos_window_s: float = 60.0
    last_event: Optional[EyeClosureEvent] = None


class OcularEngine:
    """Real-time ocular analysis engine operating on 3D landmarks with timestamp precision."""

    # Canonical MediaPipe Face Mesh landmark indices for ocular analysis
    # Right eye: 33 (corner outer), 160 (top 1), 158 (top 2), 133 (corner inner), 153 (bot 2), 144 (bot 1)
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    # Left eye: 362 (corner inner), 385 (top 1), 387 (top 2), 263 (corner outer), 373 (bot 2), 380 (bot 1)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def __init__(
        self,
        config: Optional[OcularConfig] = None,
        ear_threshold: float = 0.20,
    ) -> None:
        self.config = config or OcularConfig()
        self.ear_threshold = ear_threshold

        # Smoothing state
        self._smoothed_ear: Optional[float] = None

        # Ongoing closure state
        self._is_closed = False
        self._closure_start_ts: Optional[float] = None

        # Historical event buffers for sliding-window metrics
        self._slow_closures: Deque[Tuple[float, float]] = deque()  # (start_ts, end_ts) for PERCLOS
        self._recent_blinks: Deque[EyeClosureEvent] = deque()      # For blink rate & median duration
        self._microsleeps: Deque[EyeClosureEvent] = deque()        # For recent microsleep count

        # Last completed event
        self._last_event: Optional[EyeClosureEvent] = None

    def set_ear_threshold(self, threshold: float) -> None:
        """Updates EAR threshold based on astronaut's baseline calibration."""
        self.ear_threshold = max(0.05, min(0.40, threshold))

    @staticmethod
    def compute_single_ear(landmarks_subset: np.ndarray) -> float:
        """Computes the Eye Aspect Ratio for 6 points: P1, P2, P3, P4, P5, P6.

        EAR = (||P2 - P6|| + ||P3 - P5||) / (2.0 * ||P1 - P4||)
        """
        if len(landmarks_subset) < 6:
            return 0.0

        p1, p2, p3, p4, p5, p6 = landmarks_subset[:6, :2]

        dist_vertical_1 = float(np.linalg.norm(p2 - p6))
        dist_vertical_2 = float(np.linalg.norm(p3 - p5))
        dist_horizontal = float(np.linalg.norm(p1 - p4))

        if dist_horizontal < 1e-6:
            return 0.0

        ear = (dist_vertical_1 + dist_vertical_2) / (2.0 * dist_horizontal)
        return float(ear)

    def extract_ear_from_landmarks(self, face_landmarks: np.ndarray) -> Tuple[float, float, float]:
        """Extracts right, left, and mean EAR from full 468/478 face landmarks array."""
        right_points = face_landmarks[self.RIGHT_EYE_INDICES]
        left_points = face_landmarks[self.LEFT_EYE_INDICES]

        ear_right = self.compute_single_ear(right_points)
        ear_left = self.compute_single_ear(left_points)
        ear_mean = (ear_right + ear_left) / 2.0

        return ear_mean, ear_right, ear_left

    def process(
        self,
        ear_raw: float,
        timestamp: float,
        ear_left: Optional[float] = None,
        ear_right: Optional[float] = None,
    ) -> OcularTelemetry:
        """Processes a single frame's EAR measurement at the given timestamp (seconds)."""
        # 1. Exponential Moving Average (EMA) smoothing
        alpha = self.config.ema_alpha
        if self._smoothed_ear is None:
            self._smoothed_ear = ear_raw
        else:
            self._smoothed_ear = alpha * ear_raw + (1.0 - alpha) * self._smoothed_ear

        # 2. Eyelid closure tracking
        is_currently_closed = self._smoothed_ear < self.ear_threshold
        current_closure_duration_ms = 0.0
        newly_closed_event: Optional[EyeClosureEvent] = None

        if is_currently_closed:
            if not self._is_closed:
                # Transition: OPEN -> CLOSED
                self._is_closed = True
                self._closure_start_ts = timestamp
            start_c = self._closure_start_ts if self._closure_start_ts is not None else timestamp
            current_closure_duration_ms = (timestamp - start_c) * 1000.0
        else:
            if self._is_closed:
                # Transition: CLOSED -> OPEN (Closure event concluded)
                start_ts = self._closure_start_ts if self._closure_start_ts is not None else timestamp
                duration_ms = (timestamp - start_ts) * 1000.0
                self._is_closed = False
                self._closure_start_ts = None

                # Classify closure type
                if duration_ms < self.config.blink_min_ms:
                    closure_type = EyeClosureType.NOISE
                elif duration_ms < self.config.droop_min_ms:
                    closure_type = EyeClosureType.BLINK
                elif duration_ms < self.config.microsleep_min_ms:
                    closure_type = EyeClosureType.DROOP
                else:
                    closure_type = EyeClosureType.MICROSLEEP

                event = EyeClosureEvent(
                    start_ts=start_ts,
                    end_ts=timestamp,
                    duration_ms=duration_ms,
                    event_type=closure_type,
                )
                self._last_event = event
                newly_closed_event = event

                # Record into sliding windows if valid event
                if closure_type in (EyeClosureType.DROOP, EyeClosureType.MICROSLEEP):
                    self._slow_closures.append((start_ts, timestamp))

                if closure_type in (EyeClosureType.BLINK, EyeClosureType.DROOP):
                    self._recent_blinks.append(event)

                if closure_type == EyeClosureType.MICROSLEEP:
                    self._microsleeps.append(event)

        # 3. Clean up sliding windows
        perclos_cutoff = timestamp - self.config.perclos_window_s
        while self._slow_closures and self._slow_closures[0][1] < perclos_cutoff:
            self._slow_closures.popleft()

        metrics_cutoff = timestamp - self.config.metrics_window_s
        while self._recent_blinks and self._recent_blinks[0].end_ts < metrics_cutoff:
            self._recent_blinks.popleft()

        while self._microsleeps and self._microsleeps[0].end_ts < metrics_cutoff:
            self._microsleeps.popleft()

        # 4. Compute PERCLOS: % of time eyes were in slow closures within rolling 60s
        total_slow_closure_time_s = 0.0
        for start_t, end_t in self._slow_closures:
            clamped_start = max(start_t, perclos_cutoff)
            clamped_end = min(end_t, timestamp)
            if clamped_end > clamped_start:
                total_slow_closure_time_s += (clamped_end - clamped_start)

        # Also account for ongoing slow closure if eye is currently closed > 400ms
        if self._is_closed and self._closure_start_ts is not None:
            ongoing_duration_s = timestamp - self._closure_start_ts
            if ongoing_duration_s >= (self.config.droop_min_ms / 1000.0):
                clamped_start = max(self._closure_start_ts, perclos_cutoff)
                total_slow_closure_time_s += max(0.0, timestamp - clamped_start)

        # Normalization: in 60s window (or time elapsed since start)
        window_duration_s = self.config.perclos_window_s
        perclos = (total_slow_closure_time_s / window_duration_s) * 100.0
        perclos = float(np.clip(perclos, 0.0, 100.0))

        # 5. Compute Blink Rate (blinks per minute) and Median Blink Duration
        num_blinks = len(self._recent_blinks)
        metrics_window_min = self.config.metrics_window_s / 60.0
        blink_rate_bpm = float(num_blinks / metrics_window_min)

        if self._recent_blinks:
            durations = [b.duration_ms for b in self._recent_blinks]
            blink_duration_median_ms = float(np.median(durations))
        else:
            blink_duration_median_ms = 0.0

        microsleep_count = len(self._microsleeps)

        # Threshold proximity and prolonged closure indicators
        is_near_thresh = (not self._is_closed) and (
            float(self._smoothed_ear) <= self.ear_threshold * 1.15
        )
        is_prolonged = self._is_closed and (current_closure_duration_ms > 350.0)

        e_l = ear_left if ear_left is not None else ear_raw
        e_r = ear_right if ear_right is not None else ear_raw

        return OcularTelemetry(
            timestamp=timestamp,
            ear_raw=ear_raw,
            ear_smooth=float(self._smoothed_ear),
            is_eye_closed=self._is_closed,
            current_closure_duration_ms=current_closure_duration_ms,
            perclos=perclos,
            blink_rate_bpm=blink_rate_bpm,
            blink_duration_median_ms=blink_duration_median_ms,
            microsleep_count_recent=microsleep_count,
            ear_left=float(e_l),
            ear_right=float(e_r),
            ear_threshold=float(self.ear_threshold),
            is_near_threshold=is_near_thresh,
            is_prolonged_closure=is_prolonged,
            perclos_window_s=self.config.perclos_window_s,
            last_event=self._last_event,
        )

    def reset(self) -> None:
        """Resets engine temporal state."""
        self._smoothed_ear = None
        self._is_closed = False
        self._closure_start_ts = None
        self._slow_closures.clear()
        self._recent_blinks.clear()
        self._microsleeps.clear()
        self._last_event = None
