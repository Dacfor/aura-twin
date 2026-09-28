"""Synthetic Astronaut Simulator for AURA-Face.

Generates realistic psychophysiological facial behavior profiles for offline testing,
automated CI validation, and bulletproof hackathon jury demonstrations.

Flight Profiles:
1. `nominal`: High alertness, relaxed open eyes (EAR ~0.30), natural rapid blinks (18/min, 180ms),
   low AU4 (~0.05). Demonstrates stable ALERT status.
2. `fatigue`: Sleep-deprived astronaut under prolonged lunar night. Eyelids progressively droop,
   slow closures (420ms), followed by two microsleep episodes (600ms and 750ms),
   yawn onset, PERCLOS rising to 18%. Demonstrates ALERT -> MODERATE -> CRITICAL transitions.
3. `cognitive_load`: High mental arithmetic / emergency EVA anomaly. AU4 brow lowerer
   sustained at 0.55 for 8 seconds without smile, blink inhibition, followed by
   relief Duchenne smile (AU12 + AU6).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple

import cv2
import numpy as np

from aura_face.capture import FrameAnalysisResult, HeadPose


class SyntheticAstronautSimulator:
    """Generates synthetic physiological telemetry and stylized cockpit video frames."""

    def __init__(
        self,
        profile: str = "nominal",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.profile = profile.lower()
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1.0 / fps

        self._start_time = time.time()
        self._elapsed = 0.0

    def get_frame(self, t: float) -> Tuple[np.ndarray, FrameAnalysisResult]:
        """Generates a synthetic frame and its matching feature parameters."""
        # 1. Compute dynamic EAR and Blendshapes based on flight profile
        ear, blendshapes = self._compute_profile_signals(t)

        # 2. Render stylized Artemis Astronaut wireframe avatar
        frame = self._render_avatar_frame(t, ear, blendshapes)

        # 3. Build FrameAnalysisResult
        result = FrameAnalysisResult(
            timestamp=t,
            face_detected=True,
            tracking_quality=1.0,
            head_pose=HeadPose(yaw=math.sin(t * 0.2) * 4.0, pitch=math.cos(t * 0.15) * 3.0),
            landmarks_3d=None,
            blendshapes=blendshapes,
        )

        return frame, result

    def _compute_profile_signals(self, t: float) -> Tuple[float, Dict[str, float]]:
        """Calculates physiological signals for the given mission elapsed time."""
        blendshapes = {
            "browDownLeft": 0.04,
            "browDownRight": 0.04,
            "eyeSquintLeft": 0.03,
            "eyeSquintRight": 0.03,
            "mouthSmileLeft": 0.02,
            "mouthSmileRight": 0.02,
            "browInnerUp": 0.02,
            "noseSneerLeft": 0.01,
            "noseSneerRight": 0.01,
            "mouthFrownLeft": 0.02,
            "mouthFrownRight": 0.02,
            "browOuterUpLeft": 0.02,
            "browOuterUpRight": 0.02,
            "jawOpen": 0.05,
        }

        # Base nominal open EAR ~ 0.31 with slight physiological tremor
        ear = 0.31 + 0.01 * math.sin(t * 1.5)

        if self.profile == "nominal":
            # Natural blinks every 3.5 seconds lasting ~180 ms
            blink_cycle = t % 3.5
            if blink_cycle < 0.18:
                # Eyelid closed
                ear = 0.12
            else:
                ear = 0.30

        elif self.profile == "fatigue":
            # Progressive fatigue scenario over a 40-second demo loop:
            # 0-10s: Alert nominal (PERCLOS low)
            # 10-22s: Progressive eyelid droop (slow closures 450ms) -> MODERATE
            # 22-30s: Two severe microsleeps (700ms each) + yawn -> CRITICAL
            # 30-40s: Recovery / rest
            loop_t = t % 45.0

            if loop_t < 10.0:
                # Nominal alert blinks
                if (loop_t % 3.0) < 0.18:
                    ear = 0.12
            elif 10.0 <= loop_t < 22.0:
                # Eyelid droops (slow closures of 450 ms every 2.5s)
                cycle = (loop_t - 10.0) % 2.5
                if cycle < 0.45:
                    ear = 0.11
                else:
                    ear = 0.23  # Heavy, drowsy resting eyelids
            elif 22.0 <= loop_t < 32.0:
                # Microsleep episode 1 at 23s (duration 750 ms)
                if 23.0 <= loop_t < 23.75:
                    ear = 0.09
                # Microsleep episode 2 at 26s (duration 850 ms)
                elif 26.0 <= loop_t < 26.85:
                    ear = 0.08
                # Yawn at 28-31s
                elif 28.0 <= loop_t < 31.0:
                    blendshapes["jawOpen"] = 0.75
                    ear = 0.14
                else:
                    ear = 0.20
            else:
                # Recovery
                ear = 0.29
                if (loop_t % 4.0) < 0.20:
                    ear = 0.12

        elif self.profile == "cognitive_load":
            # 0-5s: Baseline
            # 5-15s: Complex mental arithmetic / anomaly -> AU4 rises to 0.58
            # 15-22s: Relief and success -> AU12 + AU6 smile (Happiness Duchenne)
            loop_t = t % 25.0
            if 5.0 <= loop_t < 15.0:
                blendshapes["browDownLeft"] = 0.58
                blendshapes["browDownRight"] = 0.58
                blendshapes["eyeSquintLeft"] = 0.35
                blendshapes["eyeSquintRight"] = 0.35
                # Cognitive blink inhibition (eyes wide open, no blinks)
                ear = 0.33
            elif 15.0 <= loop_t < 21.0:
                # Relief smile
                blendshapes["mouthSmileLeft"] = 0.65
                blendshapes["mouthSmileRight"] = 0.65
                blendshapes["eyeSquintLeft"] = 0.42
                blendshapes["eyeSquintRight"] = 0.42
                ear = 0.28
            else:
                ear = 0.30
                if (loop_t % 3.5) < 0.18:
                    ear = 0.12

        return ear, blendshapes

    def _render_avatar_frame(
        self,
        t: float,
        ear: float,
        blendshapes: Dict[str, float],
    ) -> np.ndarray:
        """Renders stylized deep-space cockpit canvas with animated astronaut wireframe."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Deep space dark gradient background
        frame[:] = (18, 22, 28)

        # Draw lunar habitat window grid / stars
        for i in range(40):
            sx = int((i * 137 + t * 5) % self.width)
            sy = int((i * 97) % self.height)
            brightness = int(120 + 80 * math.sin(t + i))
            cv2.circle(frame, (sx, sy), 1, (brightness, brightness, brightness), -1)

        cx, cy = self.width // 2, self.height // 2

        # Draw astronaut helmet outline
        cv2.circle(frame, (cx, cy), 180, (40, 50, 65), -1)
        cv2.circle(frame, (cx, cy), 180, (0, 180, 240), 2)
        # Gold/cyan visor reflection
        cv2.ellipse(frame, (cx, cy), (140, 110), 0, -40, 70, (20, 140, 200), 2)

        # Draw stylized eyes based on simulated EAR
        eye_y = cy - 20
        eye_dx = 55
        eye_h = int(max(2, ear * 60))
        eye_color = (60, 76, 231) if ear < 0.20 else (113, 204, 46)

        # Right eye
        cv2.ellipse(frame, (cx - eye_dx, eye_y), (25, eye_h), 0, 0, 360, eye_color, 2)
        cv2.circle(frame, (cx - eye_dx, eye_y), 4, (255, 255, 255), -1)

        # Left eye
        cv2.ellipse(frame, (cx + eye_dx, eye_y), (25, eye_h), 0, 0, 360, eye_color, 2)
        cv2.circle(frame, (cx + eye_dx, eye_y), 4, (255, 255, 255), -1)

        # Brow (AU4 effect)
        au4 = max(blendshapes.get("browDownLeft", 0.0), blendshapes.get("browDownRight", 0.0))
        brow_slant = int(au4 * 16)
        cv2.line(frame, (cx - eye_dx - 30, eye_y - 25 + brow_slant), (cx - eye_dx + 25, eye_y - 30 + brow_slant), (0, 230, 255), 2)
        cv2.line(frame, (cx + eye_dx - 25, eye_y - 30 + brow_slant), (cx + eye_dx + 30, eye_y - 25 + brow_slant), (0, 230, 255), 2)

        # Mouth (Smile AU12 / Yawn jawOpen)
        jaw_open = blendshapes.get("jawOpen", 0.0)
        smile = blendshapes.get("mouthSmileLeft", 0.0)
        mouth_y = cy + 55
        if jaw_open > 0.40:
            # Yawn oval
            yawn_h = int(jaw_open * 35)
            cv2.ellipse(frame, (cx, mouth_y + 10), (22, yawn_h), 0, 0, 360, (60, 76, 231), 2)
        elif smile > 0.30:
            # Curved smile
            cv2.ellipse(frame, (cx, mouth_y - 5), (35, 18), 0, 10, 170, (113, 204, 46), 2)
        else:
            # Neutral line
            cv2.line(frame, (cx - 25, mouth_y), (cx + 25, mouth_y), (160, 175, 190), 2)

        # Watermark
        cv2.putText(
            frame,
            f"SIMULATOR MODE // PROFILE: {self.profile.upper()}",
            (cx - 190, cy + 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 230, 255),
            1,
        )

        return frame
