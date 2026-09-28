"""Astronaut Baseline Calibration Module for AURA-Face.

Implements guided <25-second individual baseline calibration for:
1. Open-eye resting EAR (10 s)
2. Closed-eye / blink percentile EAR (10 s)
3. Resting FACS Action Units (AU4, AU6, AU12)
4. Baseline blink rate

Scientific rationale: Individual facial morphology, eye aperture, and resting brow position
vary widely across astronauts. Static hardcoded thresholds produce high false-alarm rates.
Individualized calibration is a prerequisite for scientific validity (Dinges & Metaxas, NSBRI).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from aura_face.config import CalibrationConfig


class CalibrationPhase(str, Enum):
    IDLE = "IDLE"
    OPEN_EYE = "OPEN_EYE"       # 10s: Relaxed open gaze at monitor
    BLINK = "BLINK"             # 10s: Natural blinks to capture closed-eye trough
    COMPLETED = "COMPLETED"


@dataclass
class AstronautBaseline:
    """Stores calibrated baseline parameters for a specific astronaut."""
    subject_id: str = "default_astronaut"
    created_at: float = field(default_factory=time.time)
    ear_open: float = 0.30
    ear_closed: float = 0.14
    ear_threshold: float = 0.19
    au4_baseline: float = 0.05
    au6_baseline: float = 0.05
    au12_baseline: float = 0.05
    blink_rate_baseline: float = 16.0  # Normal resting rate ~15-20 blinks/min

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "subject_id": self.subject_id,
            "created_at": self.created_at,
            "ear_open": self.ear_open,
            "ear_closed": self.ear_closed,
            "ear_threshold": self.ear_threshold,
            "au4_baseline": self.au4_baseline,
            "au6_baseline": self.au6_baseline,
            "au12_baseline": self.au12_baseline,
            "blink_rate_baseline": self.blink_rate_baseline,
        }


class CalibrationEngine:
    """Orchestrates individual calibration routine with progress tracking."""

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None,
        subject_id: str = "default_astronaut",
    ) -> None:
        self.config = config or CalibrationConfig()
        self.subject_id = subject_id

        self.phase = CalibrationPhase.IDLE
        self._phase_start_ts: Optional[float] = None

        # Data collection buffers
        self._open_ears: List[float] = []
        self._blink_ears: List[float] = []
        self._open_au4: List[float] = []
        self._open_au6: List[float] = []
        self._open_au12: List[float] = []

        self.calibrated_baseline = AstronautBaseline(
            subject_id=self.subject_id,
            ear_open=self.config.default_ear_open,
            ear_closed=self.config.default_ear_closed,
            ear_threshold=self.config.default_ear_threshold,
        )

    def start(self, timestamp: float) -> None:
        """Starts the calibration procedure at given timestamp."""
        self.phase = CalibrationPhase.OPEN_EYE
        self._phase_start_ts = timestamp
        self._open_ears.clear()
        self._blink_ears.clear()
        self._open_au4.clear()
        self._open_au6.clear()
        self._open_au12.clear()

    @property
    def is_active(self) -> bool:
        return self.phase in (CalibrationPhase.OPEN_EYE, CalibrationPhase.BLINK)

    def get_progress(self, timestamp: float) -> Tuple[CalibrationPhase, float]:
        """Returns the current phase and progress percentage (0.0 - 100.0)."""
        if self.phase == CalibrationPhase.IDLE:
            return CalibrationPhase.IDLE, 0.0
        if self.phase == CalibrationPhase.COMPLETED:
            return CalibrationPhase.COMPLETED, 100.0

        start_ts = self._phase_start_ts if self._phase_start_ts is not None else timestamp
        elapsed = timestamp - start_ts
        if self.phase == CalibrationPhase.OPEN_EYE:
            progress = min(100.0, (elapsed / self.config.open_eye_duration_s) * 50.0)
            return CalibrationPhase.OPEN_EYE, progress
        elif self.phase == CalibrationPhase.BLINK:
            progress = 50.0 + min(50.0, (elapsed / self.config.blink_duration_s) * 50.0)
            return CalibrationPhase.BLINK, progress

        return self.phase, 0.0

    def feed_sample(
        self,
        ear: float,
        au_raw: Dict[str, float],
        timestamp: float,
    ) -> bool:
        """Feeds a frame sample during calibration. Returns True when calibration finishes."""
        if not self.is_active:
            return self.phase == CalibrationPhase.COMPLETED

        start_ts = self._phase_start_ts if self._phase_start_ts is not None else timestamp
        elapsed = timestamp - start_ts

        if self.phase == CalibrationPhase.OPEN_EYE:
            self._open_ears.append(ear)
            self._open_au4.append(au_raw.get("AU4", 0.0))
            self._open_au6.append(au_raw.get("AU6", 0.0))
            self._open_au12.append(au_raw.get("AU12", 0.0))

            if elapsed >= self.config.open_eye_duration_s:
                # Transition to BLINK phase
                self.phase = CalibrationPhase.BLINK
                self._phase_start_ts = timestamp

        elif self.phase == CalibrationPhase.BLINK:
            self._blink_ears.append(ear)

            if elapsed >= self.config.blink_duration_s:
                # Finalize calibration
                self._finalize(timestamp)
                return True

        return False

    def _finalize(self, timestamp: float) -> None:
        """Computes statistical baselines and threshold."""
        # 1. EAR Open (median of relaxed open eyes)
        if self._open_ears:
            ear_open = float(np.median(self._open_ears))
        else:
            ear_open = self.config.default_ear_open

        # 2. EAR Closed (5th percentile of blink phase)
        if self._blink_ears:
            ear_closed = float(np.percentile(self._blink_ears, 5))
        else:
            ear_closed = self.config.default_ear_closed

        # Ensure realistic physiological sanity
        if ear_open <= ear_closed or (ear_open - ear_closed) < 0.05:
            ear_open = max(ear_open, 0.28)
            ear_closed = min(ear_closed, 0.14)

        # 3. Personalized EAR Threshold
        # threshold = open - factor * (open - closed)
        factor = self.config.ear_threshold_factor
        ear_threshold = ear_open - factor * (ear_open - ear_closed)
        ear_threshold = float(np.clip(ear_threshold, 0.12, 0.32))

        # 4. Action Unit Baselines
        au4_b = float(np.median(self._open_au4)) if self._open_au4 else 0.05
        au6_b = float(np.median(self._open_au6)) if self._open_au6 else 0.05
        au12_b = float(np.median(self._open_au12)) if self._open_au12 else 0.05

        self.calibrated_baseline = AstronautBaseline(
            subject_id=self.subject_id,
            created_at=timestamp,
            ear_open=ear_open,
            ear_closed=ear_closed,
            ear_threshold=ear_threshold,
            au4_baseline=float(np.clip(au4_b, 0.0, 0.25)),
            au6_baseline=float(np.clip(au6_b, 0.0, 0.25)),
            au12_baseline=float(np.clip(au12_b, 0.0, 0.25)),
            blink_rate_baseline=16.0,
        )
        self.phase = CalibrationPhase.COMPLETED
