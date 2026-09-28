"""Artemis Lunar Cockpit HUD Overlay for AURA-Face.

Features:
- Configurable Facial Marker Layer (OFF, MINIMAL, DETAILED, ALL)
- State-dependent visual coloring (Cyan, Green, Yellow, Purple, Blue, Red, Gray)
- Dynamic Eye contour feedback (Cyan/Green open, Yellow near-threshold, Red prolonged/microsleep)
- 4 Telemetry Cards (Ocular Vigilance, Behavioral Proxies, Current Pattern, Event Feed)
- Operational Affect Map (2D continuous space with 30-60s historical trail)
- Explainable AI Evidence & Zero-Frame Privacy Guarantee
"""

from __future__ import annotations

import math
import time
from collections import deque
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from aura_face.affective import (
    ActionUnitsVector,
    AffectiveTelemetry,
    OperationalBehaviorPattern,
    PsychologicalFlightState,
)
from aura_face.calibration import CalibrationPhase
from aura_face.capture import HeadPose
from aura_face.cardiovascular import CardiovascularTwinCard
from aura_face.ocular import OcularTelemetry
from aura_face.states import AstronautState


class MarkerMode(str, Enum):
    """Configurable facial landmark overlay mode."""
    OFF = "OFF"
    MINIMAL = "MINIMAL"    # Default: contours, brackets, nose anchor, head-pose
    DETAILED = "DETAILED"  # Selected landmarks on eyes, brows, mouth, nose with labels
    ALL = "ALL"            # All 478 points (dev/debug only)


class CockpitOverlay:
    """Renders real-time side-by-side cockpit interface (Camera + Telemetry Dashboard)."""

    # BGR Color Palette
    COLOR_BG = (16, 20, 26)            # Dark Space Obsidian
    COLOR_PANEL_BG = (22, 28, 36)      # Panel Background
    COLOR_BORDER = (45, 55, 70)        # Subtle Border
    COLOR_TEXT_MAIN = (245, 248, 252)  # Crisp White
    COLOR_TEXT_MUTED = (140, 155, 175) # Slate Gray
    
    # State-Dependent Palette (Section 3)
    COLOR_CYAN = (0, 230, 255)         # Normal tracking
    COLOR_GREEN = (113, 204, 46)       # CALM_ALERT
    COLOR_YELLOW = (18, 215, 250)      # MODERATE
    COLOR_PURPLE = (182, 89, 155)      # COGNITIVE_STRAIN_CANDIDATE / AU4_TENSION
    COLOR_BLUE = (220, 140, 40)        # PRE_REST
    COLOR_RED = (60, 76, 231)          # DROWSINESS / CRITICAL
    COLOR_GRAY = (140, 145, 155)       # UNKNOWN / tracking lost

    # Landmark indices for contours
    RIGHT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]
    LEFT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
    RIGHT_EYEBROW_CONTOUR = [70, 63, 105, 66, 107]
    LEFT_EYEBROW_CONTOUR = [336, 296, 334, 293, 300]
    MOUTH_OUTER_CONTOUR = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
    NOSE_BRIDGE_CONTOUR = [168, 6, 197, 195, 5, 1]

    def __init__(
        self,
        target_cam_w: int = 960,
        dashboard_w: int = 560,
        height: int = 720,
        default_marker_mode: MarkerMode = MarkerMode.MINIMAL,
        cardio_w: int = 400,
        cardio_card: Optional[CardiovascularTwinCard] = None,
    ) -> None:
        self.target_cam_w = target_cam_w
        self.dashboard_w = dashboard_w
        self.height = height
        self.cardio_w = cardio_w
        self.cardio_card = cardio_card
        self.total_width = target_cam_w + dashboard_w + (cardio_w if cardio_card is not None else 0)
        
        # Marker layer configuration
        self.marker_mode = default_marker_mode
        self.show_eye_contours = True
        self.show_face_brackets = True
        self.show_head_pose_axis = True
        self.show_debug_labels = False

        # Event and Affect Map Trail
        self._recent_events: List[Tuple[float, str]] = []
        self._affect_trail: Deque[Tuple[float, float, float]] = deque(maxlen=60) # (valence, arousal, timestamp)

    def cycle_marker_mode(self) -> MarkerMode:
        """Cycles marker mode: OFF -> MINIMAL -> DETAILED -> ALL -> OFF."""
        modes = [MarkerMode.OFF, MarkerMode.MINIMAL, MarkerMode.DETAILED, MarkerMode.ALL]
        idx = modes.index(self.marker_mode)
        self.marker_mode = modes[(idx + 1) % len(modes)]
        return self.marker_mode


    def toggle_eye_contours(self) -> bool:
        self.show_eye_contours = not self.show_eye_contours
        return self.show_eye_contours

    def toggle_face_brackets(self) -> bool:
        self.show_face_brackets = not self.show_face_brackets
        return self.show_face_brackets

    def toggle_head_pose_axis(self) -> bool:
        self.show_head_pose_axis = not self.show_head_pose_axis
        return self.show_head_pose_axis

    def toggle_debug_labels(self) -> bool:
        self.show_debug_labels = not self.show_debug_labels
        return self.show_debug_labels

    def add_event(self, text: str, timestamp: float) -> None:
        """Adds a discrete event to the HUD feed."""
        self._recent_events.append((timestamp, text))
        if len(self._recent_events) > 5:
            self._recent_events.pop(0)

    def get_state_color(self, pattern: OperationalBehaviorPattern, state: AstronautState) -> Tuple[int, int, int]:
        """Returns BGR color matching the current operational state."""
        if state == AstronautState.CRITICAL or pattern == OperationalBehaviorPattern.DROWSINESS_IMPAIRMENT:
            return self.COLOR_RED
        if state == AstronautState.MODERATE:
            return self.COLOR_YELLOW
        if pattern in (OperationalBehaviorPattern.COGNITIVE_STRAIN_CANDIDATE, OperationalBehaviorPattern.AU4_TENSION_PATTERN):
            return self.COLOR_PURPLE
        if state == AstronautState.PRE_REST:
            return self.COLOR_BLUE
        if state == AstronautState.ALERT:
            return self.COLOR_GREEN
        if pattern == OperationalBehaviorPattern.UNKNOWN:
            return self.COLOR_GRAY
        return self.COLOR_CYAN

    def draw_hud(
        self,
        frame: np.ndarray,
        ocular: OcularTelemetry,
        affective: AffectiveTelemetry,
        state: AstronautState,
        session_id: str,
        subject_id: str,
        ear_threshold: float,
        calib_phase: CalibrationPhase = CalibrationPhase.COMPLETED,
        calib_progress: float = 100.0,
        landmarks_3d: Optional[np.ndarray] = None,
        head_pose: Optional[HeadPose] = None,
    ) -> np.ndarray:
        """Composes side-by-side cockpit canvas: Left Camera, Right Telemetry Dashboard."""
        # 1. Resize camera frame to left pane
        cam_resized = cv2.resize(frame, (self.target_cam_w, self.height), interpolation=cv2.INTER_LINEAR)

        # 2. Draw configurable facial markers directly on camera view
        state_color = self.get_state_color(affective.current_pattern, state)
        if landmarks_3d is not None and len(landmarks_3d) >= 68:
            self._draw_facial_markers(
                cam_resized,
                landmarks_3d,
                ocular,
                state_color,
                head_pose=head_pose,
            )

        # 3. Camera overlays (subtle perimeter border and calibration banner)
        self._draw_camera_overlays(cam_resized, state, affective.current_pattern, ocular, calib_phase, calib_progress)

        # 4. Create right telemetry dashboard canvas (4 Cards + Affect Map)
        dashboard = np.zeros((self.height, self.dashboard_w, 3), dtype=np.uint8)
        dashboard[:] = self.COLOR_PANEL_BG

        # Update historical trail for Operational Affect Map
        self._affect_trail.append((affective.facial_valence_proxy, affective.operational_arousal_proxy, ocular.timestamp))

        # Render 4 Dashboard Cards
        self._draw_dashboard_header(dashboard, session_id, subject_id, affective.current_pattern, state_color)
        self._draw_card1_ocular_vigilance(dashboard, ocular, ear_threshold)
        self._draw_card2_behavioral_proxies(dashboard, affective)
        self._draw_card3_current_pattern(dashboard, affective, state_color)
        self._draw_card4_events_and_affect_map(dashboard, ocular.timestamp, affective, state_color)
        self._draw_footer(dashboard)

        # 5. Assemble composite canvas
        composite = np.zeros((self.height, self.total_width, 3), dtype=np.uint8)
        composite[:, :self.target_cam_w] = cam_resized
        composite[:, self.target_cam_w:self.target_cam_w + self.dashboard_w] = dashboard

        # Separator line between camera and dashboard
        cv2.line(composite, (self.target_cam_w, 0), (self.target_cam_w, self.height), self.COLOR_BORDER, 2)

        # 6. Render Card 5: Cardiovascular Digital Twin Column (if enabled)
        if self.cardio_card is not None:
            cardio_x = self.target_cam_w + self.dashboard_w
            self.cardio_card.draw_card(composite, cardio_x, 0, self.cardio_w, self.height, ocular.timestamp)
            cv2.line(composite, (cardio_x, 0), (cardio_x, self.height), self.COLOR_BORDER, 2)

        return composite

    def _draw_facial_markers(
        self,
        cam_frame: np.ndarray,
        landmarks: np.ndarray,
        ocular: OcularTelemetry,
        state_color: Tuple[int, int, int],
        head_pose: Optional[HeadPose] = None,
    ) -> None:
        """Renders configurable marker layer according to active MarkerMode."""
        if self.marker_mode == MarkerMode.OFF:
            return

        h, w = cam_frame.shape[:2]

        # Determine eye color
        if ocular.is_prolonged_closure or ocular.microsleep_count_recent > 0:
            eye_color = self.COLOR_RED
        elif ocular.is_near_threshold:
            eye_color = self.COLOR_YELLOW
        else:
            eye_color = self.COLOR_CYAN

        # --- A. ALL Mode: Render all 478 landmarks as small dots ---
        if self.marker_mode == MarkerMode.ALL:
            for i in range(len(landmarks)):
                px = int(landmarks[i, 0] * w)
                py = int(landmarks[i, 1] * h)
                cv2.circle(cam_frame, (px, py), 1, self.COLOR_CYAN, -1)
                if self.show_debug_labels and i % 10 == 0:
                    cv2.putText(cam_frame, str(i), (px + 2, py - 2), cv2.FONT_HERSHEY_PLAIN, 0.6, self.COLOR_TEXT_MUTED, 1)

        # --- B. MINIMAL and DETAILED Modes ---
        if self.marker_mode in (MarkerMode.MINIMAL, MarkerMode.DETAILED):
            # 1. Eye Contours
            if self.show_eye_contours:
                r_pts = np.array([[int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)] for i in self.RIGHT_EYE_CONTOUR], dtype=np.int32)
                l_pts = np.array([[int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)] for i in self.LEFT_EYE_CONTOUR], dtype=np.int32)
                cv2.polylines(cam_frame, [r_pts], isClosed=True, color=eye_color, thickness=2, lineType=cv2.LINE_AA)
                cv2.polylines(cam_frame, [l_pts], isClosed=True, color=eye_color, thickness=2, lineType=cv2.LINE_AA)

            # 2. Eyebrow Contours
            if len(landmarks) >= 337:
                rb_pts = np.array([[int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)] for i in self.RIGHT_EYEBROW_CONTOUR], dtype=np.int32)
                lb_pts = np.array([[int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)] for i in self.LEFT_EYEBROW_CONTOUR], dtype=np.int32)
                cv2.polylines(cam_frame, [rb_pts], isClosed=False, color=state_color, thickness=1, lineType=cv2.LINE_AA)
                cv2.polylines(cam_frame, [lb_pts], isClosed=False, color=state_color, thickness=1, lineType=cv2.LINE_AA)

            # 3. Mouth Outer Contour
            if len(landmarks) >= 410:
                m_pts = np.array([[int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)] for i in self.MOUTH_OUTER_CONTOUR], dtype=np.int32)
                cv2.polylines(cam_frame, [m_pts], isClosed=True, color=state_color, thickness=1, lineType=cv2.LINE_AA)

            # 4. Nose Bridge Anchor & Central Tracking Point
            if len(landmarks) >= 198:
                nb_pts = np.array([[int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)] for i in self.NOSE_BRIDGE_CONTOUR], dtype=np.int32)
                cv2.polylines(cam_frame, [nb_pts], isClosed=False, color=self.COLOR_BORDER, thickness=1, lineType=cv2.LINE_AA)
            
            # Central tracking anchor point (nose tip #1)
            nx = int(landmarks[1, 0] * w)
            ny = int(landmarks[1, 1] * h)
            cv2.circle(cam_frame, (nx, ny), 3, self.COLOR_CYAN, -1)

            # 5. Face Brackets
            if self.show_face_brackets:
                xs = landmarks[:, 0] * w
                ys = landmarks[:, 1] * h
                min_x, max_x = max(5, int(np.min(xs)) - 15), min(w - 5, int(np.max(xs)) + 15)
                min_y, max_y = max(5, int(np.min(ys)) - 25), min(h - 5, int(np.max(ys)) + 15)
                b_len = 18

                # Top-Left
                cv2.line(cam_frame, (min_x, min_y), (min_x + b_len, min_y), state_color, 2)
                cv2.line(cam_frame, (min_x, min_y), (min_x, min_y + b_len), state_color, 2)
                # Top-Right
                cv2.line(cam_frame, (max_x, min_y), (max_x - b_len, min_y), state_color, 2)
                cv2.line(cam_frame, (max_x, min_y), (max_x, min_y + b_len), state_color, 2)
                # Bottom-Left
                cv2.line(cam_frame, (min_x, max_y), (min_x + b_len, max_y), state_color, 2)
                cv2.line(cam_frame, (min_x, max_y), (min_x, max_y - b_len), state_color, 2)
                # Bottom-Right
                cv2.line(cam_frame, (max_x, max_y), (max_x - b_len, max_y), state_color, 2)
                cv2.line(cam_frame, (max_x, max_y), (max_x, max_y - b_len), state_color, 2)

            # 6. Optional Head Pose 3D Axes
            if self.show_head_pose_axis and head_pose and head_pose.axis_points:
                origin, pt_x, pt_y, pt_z = head_pose.axis_points
                cv2.line(cam_frame, origin, pt_x, (0, 0, 255), 2)  # X-Axis (Red)
                cv2.line(cam_frame, origin, pt_y, (0, 255, 0), 2)  # Y-Axis (Green)
                cv2.line(cam_frame, origin, pt_z, (255, 0, 0), 2)  # Z-Axis (Blue)

        # --- C. DETAILED Mode specific landmark circles ---
        if self.marker_mode == MarkerMode.DETAILED:
            detailed_indices = [33, 133, 362, 263, 70, 107, 336, 300, 61, 291, 0, 17, 1]
            for idx in detailed_indices:
                if idx < len(landmarks):
                    px = int(landmarks[idx, 0] * w)
                    py = int(landmarks[idx, 1] * h)
                    cv2.circle(cam_frame, (px, py), 2, (255, 255, 255), -1)
                    if self.show_debug_labels:
                        cv2.putText(cam_frame, str(idx), (px + 3, py - 3), cv2.FONT_HERSHEY_PLAIN, 0.7, self.COLOR_CYAN, 1)

    def _draw_camera_overlays(
        self,
        cam_frame: np.ndarray,
        state: AstronautState,
        pattern: OperationalBehaviorPattern,
        ocular: OcularTelemetry,
        calib_phase: CalibrationPhase,
        calib_progress: float,
    ) -> None:
        h, w = cam_frame.shape[:2]

        # Top bar with hotkey guide
        cv2.rectangle(cam_frame, (0, 0), (w, 32), self.COLOR_BG, -1)
        cv2.line(cam_frame, (0, 32), (w, 32), self.COLOR_BORDER, 1)
        hotkey_text = f"MARKERS: [{self.marker_mode.value}] (M=mode, O=eyes, B=brackets, H=pose, D=debug, Q=quit)"
        cv2.putText(cam_frame, hotkey_text, (15, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_MUTED, 1)

        # Subtle state-colored perimeter border
        state_col = self.get_state_color(pattern, state)
        cv2.rectangle(cam_frame, (1, 33), (w - 2, h - 2), state_col, 1)

        # Pulsing Red warning frame ONLY on critical drowsiness / prolonged lapse
        if state == AstronautState.CRITICAL or pattern == OperationalBehaviorPattern.DROWSINESS_IMPAIRMENT:
            pulse = int((np.sin(time.time() * 8.0) + 1.0) * 127)
            cv2.rectangle(cam_frame, (0, 32), (w - 1, h - 1), (30, 30, pulse), 5)
            cv2.putText(cam_frame, "! VIGILANCE IMPAIRMENT DETECTED !", (w // 2 - 200, 70), cv2.FONT_HERSHEY_DUPLEX, 0.70, (60, 76, 231), 2)

        # Closure duration indicator (discreet, bottom center)
        if ocular.is_eye_closed and ocular.current_closure_duration_ms > 350:
            msg = f"EYES CLOSED: {ocular.current_closure_duration_ms:.0f} ms"
            cv2.putText(cam_frame, msg, (w // 2 - 110, h - 25), cv2.FONT_HERSHEY_DUPLEX, 0.58, (60, 76, 231), 2)

        # Calibration progress bar
        if calib_phase in (CalibrationPhase.OPEN_EYE, CalibrationPhase.BLINK):
            by = h - 75
            cv2.rectangle(cam_frame, (0, by), (w, h), self.COLOR_BG, -1)
            cv2.line(cam_frame, (0, by), (w, by), self.COLOR_CYAN, 2)
            title = "CALIBRATION: RELAXED OPEN EYE (10s)" if calib_phase == CalibrationPhase.OPEN_EYE else "CALIBRATION: NATURAL BLINKING (10s)"
            cv2.putText(cam_frame, title, (20, by + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.COLOR_CYAN, 1)
            self._draw_bar(cam_frame, 20, by + 34, w - 40, 12, calib_progress / 100.0, self.COLOR_GREEN)

    def _draw_dashboard_header(
        self,
        db: np.ndarray,
        session_id: str,
        subject_id: str,
        pattern: OperationalBehaviorPattern,
        state_color: Tuple[int, int, int],
    ) -> None:
        w = db.shape[1]
        cv2.rectangle(db, (0, 0), (w, 62), self.COLOR_BG, -1)
        cv2.line(db, (0, 62), (w, 62), self.COLOR_BORDER, 1)

        cv2.putText(db, "AURA-FACE // TELEMETRY HUD", (18, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(db, f"CREW: {subject_id.upper()} | {session_id}", (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Status badge top right
        bx, by, bw, bh = w - 185, 12, 170, 38
        cv2.rectangle(db, (bx, by), (bx + bw, by + bh), state_color, -1)
        cv2.rectangle(db, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
        badge_text = pattern.value.replace("_", " ")[:16]
        ts = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)[0]
        tx = bx + (bw - ts[0]) // 2
        ty = by + (bh + ts[1]) // 2
        cv2.putText(db, badge_text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.42, (10, 10, 10), 1, cv2.LINE_AA)

    def _draw_card1_ocular_vigilance(self, db: np.ndarray, ocular: OcularTelemetry, ear_threshold: float) -> None:
        """Card 1: Ocular Vigilance (Section 9 & 8)."""
        y0 = 75
        w = db.shape[1]
        self._draw_card_box(db, 15, y0, w - 30, 140, "CARD 1: OCULAR VIGILANCE DYNAMICS (NASA PERCLOS)")

        # Row 1: EAR Left / Right / Avg / Threshold
        cv2.putText(db, f"EAR Left: {ocular.ear_left:5.3f} | Right: {ocular.ear_right:5.3f}", (25, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(db, f"Avg EAR: {ocular.ear_smooth:5.3f} (Baseline Thresh: {ear_threshold:5.2f})", (25, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

        # Row 2: PERCLOS 60s meter
        p_col = self.COLOR_GREEN if ocular.perclos < 8.0 else (self.COLOR_YELLOW if ocular.perclos < 15.0 else self.COLOR_RED)
        cv2.putText(db, f"PERCLOS (60s): {ocular.perclos:4.1f}%  [Alert <8% | Critical >=15%]", (25, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        self._draw_bar(db, 25, y0 + 82, w - 50, 12, ocular.perclos / 30.0, p_col)
        # 8% and 15% ticks
        x8 = int(25 + (w - 50) * (8.0 / 30.0))
        x15 = int(25 + (w - 50) * (15.0 / 30.0))
        cv2.line(db, (x8, y0 + 80), (x8, y0 + 96), self.COLOR_YELLOW, 2)
        cv2.line(db, (x15, y0 + 80), (x15, y0 + 96), self.COLOR_RED, 2)

        # Row 3: Blink rate, median duration, microsleep count
        cv2.putText(db, f"Blink Rate: {ocular.blink_rate_bpm:4.1f} BPM", (25, y0 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(db, f"Median Dur: {ocular.blink_duration_median_ms:3.0f} ms", (180, y0 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        m_col = self.COLOR_RED if ocular.microsleep_count_recent > 0 else self.COLOR_TEXT_MAIN
        cv2.putText(db, f"Microsleeps (5m): {ocular.microsleep_count_recent}", (340, y0 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40, m_col, 1, cv2.LINE_AA)

    def _draw_card2_behavioral_proxies(self, db: np.ndarray, affective: AffectiveTelemetry) -> None:
        """Card 2: Behavioral Proxies (Section 5 & 9)."""
        y0 = 225
        w = db.shape[1]
        self._draw_card_box(db, 15, y0, w - 30, 155, "CARD 2: BEHAVIORAL PROXIES (BARRETT 2019 XAI)")

        # 1. Facial Valence Proxy [-1, +1]
        v = affective.facial_valence_proxy
        v_col = self.COLOR_GREEN if v >= 0.0 else self.COLOR_YELLOW
        cv2.putText(db, f"Facial Valence Proxy:    {v:+4.2f}  [-1 to +1]", (25, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        self._draw_bipolar_bar(db, 25, y0 + 36, w - 50, 9, v)

        # 2. Operational Arousal Proxy [0, 1]
        a = affective.operational_arousal_proxy
        cv2.putText(db, f"Operational Arousal:     {a:4.2f}  [0.0 to 1.0]", (25, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        self._draw_bar(db, 25, y0 + 66, w - 50, 9, a, self.COLOR_CYAN)

        # 3. Cognitive Workload Proxy [0, 1]
        cw = affective.cognitive_workload_proxy
        cw_col = self.COLOR_RED if cw > 0.60 else (self.COLOR_YELLOW if cw > 0.40 else self.COLOR_GREEN)
        cv2.putText(db, f"Cognitive Workload Proxy: {cw:4.2f}  [0.0 to 1.0]", (25, y0 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        self._draw_bar(db, 25, y0 + 96, w - 50, 9, cw, cw_col)

        # 4. Confidence & Explainability Evidence string
        conf_text = f"Confidence: {affective.confidence*100:.0f}%"
        cv2.putText(db, conf_text, (25, y0 + 124), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_GREEN if affective.confidence >= 0.7 else self.COLOR_YELLOW, 1, cv2.LINE_AA)
        
        # Display top evidence string
        top_evidence = affective.evidence[0] if affective.evidence else "Muscular tone nominal"
        cv2.putText(db, f"Evidence: {top_evidence[:45]}", (25, y0 + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_card3_current_pattern(self, db: np.ndarray, affective: AffectiveTelemetry, state_color: Tuple[int, int, int]) -> None:
        """Card 3: Current Pattern (Section 6, 7 & 9)."""
        y0 = 390
        w = db.shape[1]
        self._draw_card_box(db, 15, y0, w - 30, 85, "CARD 3: CURRENT OPERATIONAL PATTERN")

        pat_name = affective.current_pattern.value
        cv2.rectangle(db, (25, y0 + 26), (w - 25, y0 + 58), (28, 36, 48), -1)
        cv2.rectangle(db, (25, y0 + 26), (w - 25, y0 + 58), state_color, 1)
        cv2.putText(db, pat_name, (35, y0 + 48), cv2.FONT_HERSHEY_DUPLEX, 0.58, state_color, 1, cv2.LINE_AA)

        # Micro-indicators (Smile, Duchenne, Strain candidate, AU4 tension)
        sub_text = []
        if affective.duchenne_candidate_active:
            sub_text.append("DUCHENNE: ACTIVE")
        elif affective.smile_pattern_active:
            sub_text.append("SMILE: ACTIVE")
        if affective.cognitive_strain_candidate_active:
            sub_text.append("STRAIN: ACTIVE")
        elif affective.au4_tension_active:
            sub_text.append("AU4: TENSION")
        if affective.yawn_active:
            sub_text.append("YAWN: ACTIVE")

        status_str = " | ".join(sub_text) if sub_text else "All active facial cues within nominal bounds"
        cv2.putText(db, status_str, (25, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_card4_events_and_affect_map(
        self,
        db: np.ndarray,
        timestamp: float,
        affective: AffectiveTelemetry,
        state_color: Tuple[int, int, int],
    ) -> None:
        """Card 4: Event Feed + Operational Affect Map (Section 9 & 10)."""
        y0 = 485
        w = db.shape[1]
        self._draw_card_box(db, 15, y0, w - 30, 180, "CARD 4: EVENT FEED & OPERATIONAL AFFECT MAP")

        # Left side of Card 4: Event Feed (width ~320px)
        ey = y0 + 30
        if self._recent_events:
            for t_ev, txt in self._recent_events[-4:]:
                age = max(0.0, timestamp - t_ev)
                cv2.putText(db, f"[+{age:4.1f}s ago] {txt[:26]}", (25, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
                ey += 20
        else:
            cv2.putText(db, "[+] Telemetry stream active.", (25, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(db, "[+] No discrete alert events.", (25, ey + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Right side of Card 4: Operational Affect Map 2D Plot (Russell Circumplex)
        cx, cy = w - 105, y0 + 80
        size = 48

        # Affect Map Box
        cv2.rectangle(db, (cx - size, cy - size), (cx + size, cy + size), (14, 18, 24), -1)
        cv2.rectangle(db, (cx - size, cy - size), (cx + size, cy + size), self.COLOR_BORDER, 1)

        # Crosshairs
        cv2.line(db, (cx - size, cy), (cx + size, cy), (40, 50, 65), 1)
        cv2.line(db, (cx, cy - size), (cx, cy + size), (40, 50, 65), 1)

        # Axis markers
        cv2.putText(db, "+V", (cx + size - 16, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(db, "-V", (cx - size + 2, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(db, "+A", (cx + 3, cy - size + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Draw 30-60s Historical Faint Trail
        if len(self._affect_trail) > 1:
            pts = list(self._affect_trail)
            for i in range(len(pts) - 1):
                v_i, a_i, _ = pts[i]
                v_next, a_next, _ = pts[i + 1]
                px1 = int(np.clip(cx + v_i * (size - 6), cx - size + 2, cx + size - 2))
                py1 = int(np.clip(cy + (0.5 - a_i) * (size * 1.6), cy - size + 2, cy + size - 2))
                px2 = int(np.clip(cx + v_next * (size - 6), cx - size + 2, cx + size - 2))
                py2 = int(np.clip(cy + (0.5 - a_next) * (size * 1.6), cy - size + 2, cy + size - 2))
                alpha = (i + 1) / len(pts)
                color = (int(self.COLOR_CYAN[0] * alpha), int(self.COLOR_CYAN[1] * alpha), int(self.COLOR_CYAN[2] * alpha))
                cv2.line(db, (px1, py1), (px2, py2), color, 1, cv2.LINE_AA)

        # Current coordinate point
        cur_v = affective.facial_valence_proxy
        cur_a = affective.operational_arousal_proxy
        px = int(np.clip(cx + cur_v * (size - 6), cx - size + 3, cx + size - 3))
        py = int(np.clip(cy + (0.5 - cur_a) * (size * 1.6), cy - size + 3, cy + size - 3))

        # Size of dot based on confidence
        pt_radius = max(3, int(6 * affective.confidence))
        cv2.circle(db, (px, py), pt_radius, state_color, -1, cv2.LINE_AA)
        cv2.circle(db, (px, py), pt_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(db, "AFFECT MAP", (cx - 38, cy + size + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.COLOR_CYAN, 1, cv2.LINE_AA)
        cv2.putText(db, "Not a clinical diagnosis", (cx - 58, cy + size + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_footer(self, db: np.ndarray) -> None:
        """Footer: Privacy By Design & Ethical Disclaimer (Section 1)."""
        fy = self.height - 42
        w = db.shape[1]
        cv2.rectangle(db, (0, fy), (w, self.height), self.COLOR_BG, -1)
        cv2.line(db, (0, fy), (w, fy), self.COLOR_BORDER, 1)
        cv2.putText(db, "[ETHICS] Observable behavior proxies - no internal emotion inferred.", (18, fy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.COLOR_GREEN, 1, cv2.LINE_AA)
        cv2.putText(db, "ZERO-FRAME RETENTION // 100% LOCAL EDGE INFERENCE", (18, fy + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_card_box(self, db: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> None:
        cv2.rectangle(db, (x, y), (x + w, y + h), (24, 30, 40), -1)
        cv2.rectangle(db, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)
        cv2.putText(db, title, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_CYAN, 1, cv2.LINE_AA)

    def _draw_bar(self, img: np.ndarray, x: int, y: int, w: int, h: int, fraction: float, color: Tuple[int, int, int]) -> None:
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 36, 45), -1)
        fill_w = int(w * float(np.clip(fraction, 0.0, 1.0)))
        if fill_w > 0:
            cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)

    def _draw_bipolar_bar(self, img: np.ndarray, x: int, y: int, w: int, h: int, val: float) -> None:
        """Bipolar bar: center at x + w//2, extends left for negative, right for positive."""
        cv2.rectangle(img, (x, y), (x + w, y + h), (30, 36, 45), -1)
        mid_x = x + w // 2
        cv2.line(img, (mid_x, y - 2), (mid_x, y + h + 2), (100, 115, 130), 1)

        clamped = float(np.clip(val, -1.0, 1.0))
        half_w = w // 2
        bar_len = int(abs(clamped) * half_w)
        if clamped >= 0:
            cv2.rectangle(img, (mid_x, y), (mid_x + bar_len, y + h), self.COLOR_GREEN, -1)
        else:
            cv2.rectangle(img, (mid_x - bar_len, y), (mid_x, y + h), self.COLOR_YELLOW, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)
