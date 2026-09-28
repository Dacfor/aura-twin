"""Cardiovascular Digital Twin module for AURA-Face.

Implements CARD 5: CARDIOVASCULAR DIGITAL TWIN
Demonstrates a simulated personalized Lower Body Negative Pressure (LBNP)
countermeasure running in parallel with real-time behavioral/ocular analysis.

Principles:
- Non-causal parallel model: no link between camera/facial proxies and cardiovascular simulation.
- Non-blocking sequential playback of precomputed CSV trajectory.
- Mission-control dark aesthetic matching Cards 1-4.
- Technical wireframe human skeleton with animated blood circulation,
  LBNP suction chamber, quantitative telemetry, Earth target restoration gauge,
  and mini-trend plot.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CardiovascularState:
    """Current state of the cardiovascular digital twin."""
    sim_time_min: float = 0.0
    vvl: float = 696.0
    vvl_earth: float = 940.0
    lbnp: float = 0.0
    deviation_pct: float = -20.9
    is_active: bool = False
    is_target_range: bool = False
    fill_ratio: float = 0.79


class CardiovascularTrajectory:
    """Loads and samples precomputed CSV trajectory without blocking."""

    def __init__(self, csv_path: str = "data/cardiovascular_demo_C.csv") -> None:
        self.times: List[float] = []
        self.vvls: List[float] = []
        self.lbnps: List[float] = []
        self.total_duration_min: float = 60.0
        self.load(csv_path)

    def load(self, csv_path: str) -> bool:
        """Finds and parses trajectory CSV."""
        candidates = [
            Path(csv_path),
            Path("data") / "cardiovascular_demo_C.csv",
            Path("data") / "cardiovascular_demo.csv",
            Path("..") / "cardiovascular_demo_C.csv",
            Path("..") / "cardiovascular_demo.csv",
            Path(__file__).parent.parent.parent / "data" / "cardiovascular_demo_C.csv",
            Path(__file__).parent.parent.parent / "data" / "cardiovascular_demo.csv",
            Path(__file__).parent.parent.parent.parent / "cardiovascular_demo_C.csv",
            Path(__file__).parent.parent.parent.parent / "cardiovascular_demo.csv",
        ]

        found_path: Optional[Path] = None
        for cand in candidates:
            if cand.is_file():
                found_path = cand
                break

        if not found_path:
            # Fallback synthetic profile (0-10m nominal, 10-15m ramp, 15-60m stabilized)
            self._generate_fallback()
            return False

        try:
            self.times.clear()
            self.vvls.clear()
            self.lbnps.clear()
            with open(found_path, "r", encoding="utf-8") as f:
                header = f.readline()  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        try:
                            t = float(parts[0])
                            v = float(parts[1])
                            l = float(parts[2])
                            self.times.append(t)
                            self.vvls.append(v)
                            self.lbnps.append(l)
                        except ValueError:
                            continue

            if self.times:
                self.total_duration_min = max(self.times)
                return True
        except Exception:
            pass

        self._generate_fallback()
        return False

    def _generate_fallback(self) -> None:
        """Generates realistic fallback data if CSV cannot be read."""
        self.times = [t * 0.5 for t in range(121)]
        self.vvls = []
        self.lbnps = []
        for t in self.times:
            if t < 10.0:
                self.lbnps.append(0.0)
                self.vvls.append(696.0 + 3.0 * math.sin(t))
            elif t < 16.0:
                frac = (t - 10.0) / 6.0
                self.lbnps.append(frac * 20.0)
                self.vvls.append(696.0 + frac * (880.0 - 696.0))
            else:
                self.lbnps.append(20.0 + 0.5 * math.sin(t))
                self.vvls.append(879.0 + 1.2 * math.cos(t * 1.5))
        self.total_duration_min = 60.0

    def sample(self, sim_time_min: float) -> Tuple[float, float]:
        """Interpolates Vvl and LBNP at sim_time_min."""
        if not self.times:
            return 700.0, 0.0

        if sim_time_min <= self.times[0]:
            return self.vvls[0], self.lbnps[0]
        if sim_time_min >= self.times[-1]:
            return self.vvls[-1], self.lbnps[-1]

        # Binary search for interval
        idx = int(np.searchsorted(self.times, sim_time_min))
        if idx == 0:
            return self.vvls[0], self.lbnps[0]

        t0, t1 = self.times[idx - 1], self.times[idx]
        v0, v1 = self.vvls[idx - 1], self.vvls[idx]
        l0, l1 = self.lbnps[idx - 1], self.lbnps[idx]

        dt = t1 - t0
        alpha = 0.0 if dt <= 0 else (sim_time_min - t0) / dt
        v_interp = v0 + alpha * (v1 - v0)
        l_interp = l0 + alpha * (l1 - l0)
        return float(v_interp), float(l_interp)


class CardiovascularTwinCard:
    """Card 5: Cardiovascular Digital Twin HUD module."""

    # Color Palette matching AURA-Face Mission-Control HUD
    COLOR_BG = (16, 20, 26)             # Dark Space Obsidian
    COLOR_PANEL_BG = (22, 28, 36)       # Panel Background
    COLOR_CARD_BG = (24, 30, 40)        # Card Interior
    COLOR_BORDER = (45, 55, 70)         # Subtle Slate Border
    COLOR_AMBER = (18, 215, 250)        # Accent Yellow/Amber (BGR)
    COLOR_CYAN = (0, 230, 255)          # Tech Cyan
    COLOR_GREEN = (113, 204, 46)        # Nominal Green
    COLOR_RED = (40, 50, 220)           # Blood Red
    COLOR_RED_GLOW = (80, 90, 255)      # Pulsing Oxygenated Red
    COLOR_TEXT_MAIN = (245, 248, 252)   # Crisp White
    COLOR_TEXT_MUTED = (140, 155, 175)  # Slate Gray
    COLOR_SKELETON = (130, 155, 175)    # Technical Bone Slate

    def __init__(
        self,
        csv_path: str = "data/cardiovascular_demo_C.csv",
        vvl_earth: float = 940.0,
        playback_speed: float = 30.0,  # 30.0x (1.5x speed): 1 sim min in 2s real-time
        loop: bool = True,
        tolerance_pct: float = 2.0,
    ) -> None:
        self.vvl_earth = float(vvl_earth)
        self.playback_speed = float(playback_speed)
        self.loop = loop
        self.tolerance_pct = float(tolerance_pct)

        self.trajectory = CardiovascularTrajectory(csv_path)
        self.current_state = CardiovascularState(vvl_earth=self.vvl_earth)
        self._trend_history: List[Tuple[float, float]] = []  # (t_sim, vvl)

        # Dynamic visual smoothing and adaptation tracking
        self.visual_lbnp: float = 0.0
        self.prev_lbnp: float = 0.0
        self.adaptation_text: str = ""
        self.adaptation_until_real_s: float = -1.0

        # Continuous calm arrow animation phase
        self.arrow_phase: float = 0.0
        self._last_t_mission_s: float = -1.0

    def update(self, t_mission_s: float) -> CardiovascularState:
        """Non-blocking update called once per frame."""
        # Calculate simulation time
        sim_time = (t_mission_s * (self.playback_speed / 60.0))
        if self.loop and self.trajectory.total_duration_min > 0:
            sim_time = sim_time % self.trajectory.total_duration_min
        else:
            sim_time = min(sim_time, self.trajectory.total_duration_min)

        vvl, lbnp = self.trajectory.sample(sim_time)
        dev_pct = 100.0 * (vvl - self.vvl_earth) / self.vvl_earth if self.vvl_earth > 0 else 0.0
        is_active = (lbnp > 0.5)
        is_target_range = (abs(dev_pct) <= self.tolerance_pct)
        fill_ratio = float(np.clip(vvl / self.vvl_earth, 0.45, 1.25))

        # Adaptation indicator tracking (Section 8)
        delta_p = lbnp - self.prev_lbnp
        if abs(delta_p) >= 0.25:
            # Significant step change in personalized countermeasure
            self.adaptation_text = "ADAPTING ^" if delta_p > 0 else "ADAPTING v"
            self.adaptation_until_real_s = t_mission_s + 2.2  # Visible for 2.2s real-time
            self.prev_lbnp = lbnp

        if t_mission_s > self.adaptation_until_real_s:
            self.adaptation_text = ""

        # Smooth visual LBNP for responsive yet fluid yellow animation (Section 9)
        self.visual_lbnp += 0.20 * (lbnp - self.visual_lbnp)

        # Update continuous arrow animation phase at a calm, slow, graceful speed
        if self._last_t_mission_s < 0:
            dt = 0.033
        else:
            dt = t_mission_s - self._last_t_mission_s
            if dt < 0 or dt > 0.2:
                dt = 0.033
        self._last_t_mission_s = t_mission_s

        intensity = float(np.clip(abs(self.visual_lbnp) / 40.0, 0.0, 1.0))
        arrow_speed = 6.0 + 7.0 * intensity  # Calm, slow speed (6-13 px/s)
        self.arrow_phase = (self.arrow_phase + dt * arrow_speed) % 20.0

        self.current_state = CardiovascularState(
            sim_time_min=sim_time,
            vvl=vvl,
            vvl_earth=self.vvl_earth,
            lbnp=lbnp,
            deviation_pct=dev_pct,
            is_active=is_active,
            is_target_range=is_target_range,
            fill_ratio=fill_ratio,
        )

        # Track history for trend plot
        self._trend_history.append((sim_time, vvl))
        if len(self._trend_history) > 180:
            self._trend_history.pop(0)

        return self.current_state

    def draw_card(self, db: np.ndarray, x: int, y: int, w: int, h: int, t_mission_s: float) -> None:
        """Renders complete Card 5: Cardiovascular Digital Twin."""
        state = self.current_state

        # Outer Card Box
        cv2.rectangle(db, (x, y), (x + w, y + h), self.COLOR_CARD_BG, -1)
        cv2.rectangle(db, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)

        # 1. Header & Technical Titles
        cv2.putText(db, "CARD 5: CARDIOVASCULAR DIGITAL TWIN", (x + 12, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, self.COLOR_AMBER, 1, cv2.LINE_AA)
        cv2.putText(db, "PERSONALIZED TWIN: ASTRONAUT C", (x + 12, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 225, 255), 1, cv2.LINE_AA)
        cv2.line(db, (x + 10, y + 43), (x + w - 10, y + 43), self.COLOR_BORDER, 1)

        # 2. Status Banner (Environment & LBNP Status)
        env_text = "ENV: MICROGRAVITY"
        cv2.putText(db, env_text, (x + 12, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

        # LBNP Status Badge
        badge_w = 135
        badge_h = 22
        bx = x + w - badge_w - 12
        by = y + 48
        if state.is_active:
            status_text = "LBNP ACTIVE"
            badge_bg = (15, 60, 40)
            badge_border = self.COLOR_GREEN
            text_color = self.COLOR_GREEN
        else:
            status_text = "LBNP STANDBY"
            badge_bg = (28, 34, 44)
            badge_border = self.COLOR_BORDER
            text_color = self.COLOR_TEXT_MUTED

        cv2.rectangle(db, (bx, by), (bx + badge_w, by + badge_h), badge_bg, -1)
        cv2.rectangle(db, (bx, by), (bx + badge_w, by + badge_h), badge_border, 1)
        ts = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_DUPLEX, 0.38, 1)[0]
        cv2.putText(db, status_text, (bx + (badge_w - ts[0]) // 2, by + 15), cv2.FONT_HERSHEY_DUPLEX, 0.38, text_color, 1, cv2.LINE_AA)

        # 3. Schematic Human Skeleton & Animated Circulation (y = 75 to 330)
        schem_y = y + 74
        schem_h = 246
        self._draw_schematic_skeleton_and_circulation(db, x + 10, schem_y, w - 20, schem_h, state, t_mission_s)

        # 4. Quantitative Telemetry Metrics Box (y = 328 to 440)
        met_y = y + 328
        met_h = 108
        self._draw_telemetry_metrics(db, x + 10, met_y, w - 20, met_h, state)

        # 5. Earth Target Restoration Gauge (y = 444 to 516)
        rest_y = y + 444
        rest_h = 70
        self._draw_target_restoration_gauge(db, x + 10, rest_y, w - 20, rest_h, state)

        # 6. Mini-Trend Plot Vvl(t) (y = 522 to 642)
        trend_y = y + 522
        trend_h = 118
        self._draw_mini_trend(db, x + 10, trend_y, w - 20, trend_h, state)

        # 7. Ethical & Non-Causal Mission Disclaimer (y = 648 to 670)
        disc_y = y + h - 18
        cv2.putText(db, "PARALLEL DIGITAL TWIN // NON-CAUSAL MODEL", (x + 15, disc_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_schematic_skeleton_and_circulation(
        self,
        db: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        state: CardiovascularState,
        t_mission_s: float,
    ) -> None:
        """Renders technical digital twin silhouette, effective lower-body venous reservoir and dynamic outward LBNP suction."""
        # Schematic box background
        cv2.rectangle(db, (x, y), (x + w, y + h), (14, 18, 24), -1)
        cv2.rectangle(db, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)

        cx = x + w // 2  # Center vertical axis

        # Vertical anatomical landmarks
        head_cy = y + 24
        neck_y = head_cy + 13
        torso_top_y = neck_y + 4
        pelvis_y = y + 102
        knee_y = y + 162
        ankle_y = y + 218

        # --- A. Technical Digital Twin Silhouette ---
        # 1. Cranium & Visor Wireframe
        cv2.ellipse(db, (cx, head_cy), (10, 12), 0, 0, 360, self.COLOR_SKELETON, 1, cv2.LINE_AA)
        cv2.line(db, (cx - 7, head_cy - 1), (cx + 7, head_cy - 1), (180, 205, 225), 1, cv2.LINE_AA)
        cv2.circle(db, (cx, head_cy - 1), 1, (240, 248, 255), -1, cv2.LINE_AA)

        # 2. Cervical & Thoracic Spine (segmented technical vertebrae)
        for sp_y in range(neck_y, pelvis_y, 5):
            cv2.line(db, (cx - 2, sp_y), (cx + 2, sp_y), self.COLOR_SKELETON, 1, cv2.LINE_AA)
        cv2.line(db, (cx, neck_y), (cx, pelvis_y), self.COLOR_SKELETON, 1, cv2.LINE_AA)

        # 3. Thoracic Ribcage (subtle technical arcs)
        rib_offsets = [10, 18, 26, 34, 42]
        rib_widths = [16, 23, 26, 25, 20]
        for ro, rw in zip(rib_offsets, rib_widths):
            ry = torso_top_y + ro
            cv2.ellipse(db, (cx, ry), (rw, 5), 0, 180, 360, (50, 65, 80), 1, cv2.LINE_AA)

        # 4. Clavicles & Arms (clean wireframe posture)
        cv2.line(db, (cx - 24, torso_top_y + 8), (cx + 24, torso_top_y + 8), self.COLOR_SKELETON, 1, cv2.LINE_AA)
        # Left Arm
        cv2.line(db, (cx - 24, torso_top_y + 8), (cx - 36, torso_top_y + 48), (60, 75, 95), 1, cv2.LINE_AA)
        cv2.line(db, (cx - 36, torso_top_y + 48), (cx - 32, torso_top_y + 86), (60, 75, 95), 1, cv2.LINE_AA)
        # Right Arm
        cv2.line(db, (cx + 24, torso_top_y + 8), (cx + 36, torso_top_y + 48), (60, 75, 95), 1, cv2.LINE_AA)
        cv2.line(db, (cx + 36, torso_top_y + 48), (cx + 32, torso_top_y + 86), (60, 75, 95), 1, cv2.LINE_AA)

        # 5. Pelvis Girdle (anatomical technical polygon)
        pelvis_poly = np.array([
            [cx - 20, pelvis_y - 6],
            [cx + 20, pelvis_y - 6],
            [cx + 15, pelvis_y + 12],
            [cx - 15, pelvis_y + 12],
        ], dtype=np.int32)
        cv2.polylines(db, [pelvis_poly], isClosed=True, color=self.COLOR_SKELETON, thickness=1, lineType=cv2.LINE_AA)

        # Limb axes
        leg_sep = 14
        left_leg_x = cx - leg_sep
        right_leg_x = cx + leg_sep

        # Skeletal lower limb lines (Femur, Knee, Tibia)
        for lx in (left_leg_x, right_leg_x):
            cv2.line(db, (lx, pelvis_y + 10), (lx, knee_y), self.COLOR_SKELETON, 1, cv2.LINE_AA)
            cv2.circle(db, (lx, knee_y), 3, self.COLOR_SKELETON, 1, cv2.LINE_AA)
            cv2.line(db, (lx, knee_y), (lx, ankle_y), self.COLOR_SKELETON, 1, cv2.LINE_AA)
            cv2.line(db, (lx - 4, ankle_y), (lx + 5, ankle_y), self.COLOR_SKELETON, 1, cv2.LINE_AA)

        # --- B. EFFECTIVE LOWER-BODY VENOUS RESERVOIR (RED) ---
        # Driven strictly by Vvl(t) relative to Vvl_earth (Section 4 & 7)
        # Low Vvl -> dim, contracted vascular bed, low alpha
        # Restored Vvl -> expansive, rich oxygenated red vascular bed with micro-branching
        v_norm = float(np.clip((state.vvl - 520.0) / (960.0 - 520.0), 0.15, 1.0))
        venous_alpha = float(np.clip(0.30 + 0.45 * v_norm, 0.30, 0.75))

        # Blend diffuse vascular reservoir overlay
        overlay = db.copy()

        # Dynamic vascular reservoir color: deep venous red -> rich oxygenated red
        vr_b = int(22 + 16 * v_norm)
        vr_g = int(30 + 26 * v_norm)
        vr_r = int(140 + 95 * v_norm)
        venous_col = (vr_b, vr_g, vr_r)

        # 1. Pelvic Venous Plexus (diffuse elliptical pool)
        pelvic_w = int(14 + 6 * v_norm)
        pelvic_h = int(8 + 3 * v_norm)
        cv2.ellipse(overlay, (cx, pelvis_y + 5), (pelvic_w, pelvic_h), 0, 0, 360, venous_col, -1, cv2.LINE_AA)

        # 2. Thigh Venous Reservoir Bed (diffuse vascular envelope across thighs)
        thigh_spread = int(4 + 4 * v_norm)
        # Left thigh
        thigh_l = np.array([
            [left_leg_x - thigh_spread, pelvis_y + 8],
            [left_leg_x + thigh_spread, pelvis_y + 8],
            [left_leg_x + thigh_spread - 1, knee_y - 2],
            [left_leg_x - thigh_spread + 1, knee_y - 2],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [thigh_l], venous_col, lineType=cv2.LINE_AA)
        # Right thigh
        thigh_r = np.array([
            [right_leg_x - thigh_spread, pelvis_y + 8],
            [right_leg_x + thigh_spread, pelvis_y + 8],
            [right_leg_x + thigh_spread - 1, knee_y - 2],
            [right_leg_x - thigh_spread + 1, knee_y - 2],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [thigh_r], venous_col, lineType=cv2.LINE_AA)

        # 3. Calf & Lower-Leg Venous Reservoir Bed
        calf_spread = int(3 + 3 * v_norm)
        # Left calf
        calf_l = np.array([
            [left_leg_x - calf_spread, knee_y + 2],
            [left_leg_x + calf_spread, knee_y + 2],
            [left_leg_x + 2, ankle_y - 2],
            [left_leg_x - 2, ankle_y - 2],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [calf_l], venous_col, lineType=cv2.LINE_AA)
        # Right calf
        calf_r = np.array([
            [right_leg_x - calf_spread, knee_y + 2],
            [right_leg_x + calf_spread, knee_y + 2],
            [right_leg_x + 2, ankle_y - 2],
            [right_leg_x - 2, ankle_y - 2],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [calf_r], venous_col, lineType=cv2.LINE_AA)

        # Apply diffuse alpha blending for vascular tissue pool
        cv2.addWeighted(overlay, venous_alpha, db, 1.0 - venous_alpha, 0, db)

        # Micro-vascular branching / perfusion networks (density and luminosity scale with Vvl)
        if v_norm > 0.40:
            branch_alpha = float(np.clip((v_norm - 0.40) / 0.60, 0.0, 1.0))
            branch_col = (
                int(vr_b + 20 * branch_alpha),
                int(vr_g + 30 * branch_alpha),
                int(vr_r * branch_alpha),
            )
            # Lateral dendritic venules branching across thigh muscle beds
            for lx, sign in ((left_leg_x, -1), (right_leg_x, 1)):
                for by_off in (22, 38):
                    cv2.line(db, (lx, pelvis_y + by_off), (lx + sign * (thigh_spread + 2), pelvis_y + by_off - 3), branch_col, 1, cv2.LINE_AA)
                cv2.line(db, (lx, knee_y + 18), (lx + sign * (calf_spread + 2), knee_y + 22), branch_col, 1, cv2.LINE_AA)

        # Central mediastinal heart & systemic circulation
        hx, hy = cx - 4, torso_top_y + 24
        heartbeat_pulse = 0.5 + 0.5 * math.sin(t_mission_s * 7.5)
        hr_radius = 4 + int(2 * heartbeat_pulse)
        cv2.circle(db, (hx, hy), hr_radius, self.COLOR_RED_GLOW, -1, cv2.LINE_AA)
        cv2.circle(db, (hx, hy), hr_radius + 1, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

        # Systemic central vessels (descending aorta & vena cava)
        cv2.line(db, (cx - 2, hy + 4), (cx - 2, pelvis_y + 4), (30, 40, 180), 1, cv2.LINE_AA)
        cv2.line(db, (cx + 2, hy + 4), (cx + 2, pelvis_y + 4), (160, 40, 40), 1, cv2.LINE_AA)
        cv2.line(db, (cx - 2, pelvis_y + 4), (left_leg_x, pelvis_y + 10), (30, 40, 180), 1, cv2.LINE_AA)
        cv2.line(db, (cx + 2, pelvis_y + 4), (right_leg_x, pelvis_y + 10), (30, 40, 180), 1, cv2.LINE_AA)

        # --- C. LBNP INTERVENTION CHAMBER & OUTWARD SUCTION ANIMATION (YELLOW) ---
        # Controlled strictly by LBNP(t) (Section 5, 6 & 7)
        # Intensity scales with abs(LBNP) / 40.0
        # Negative pressure suction: arrows move OUTWARD from body to chamber walls (<- <- BODY -> ->)
        lbnp_top = pelvis_y - 10
        lbnp_bottom = ankle_y + 10
        lbnp_left = cx - 58
        lbnp_right = cx + 58

        lbnp_mag = abs(self.visual_lbnp)
        intensity = float(np.clip(lbnp_mag / 40.0, 0.0, 1.0))

        if state.is_active:
            # Active Chamber: yellow outline scaling with magnitude
            y_b = int(12 + 10 * intensity)
            y_g = int(175 + 65 * intensity)
            y_r = int(210 + 45 * intensity)
            chamber_col = (y_b, y_g, y_r)
            outline_thick = 2 if intensity > 0.45 else 1

            cv2.rectangle(db, (lbnp_left, lbnp_top), (lbnp_right, lbnp_bottom), chamber_col, outline_thick, cv2.LINE_AA)
            cv2.line(db, (lbnp_left, lbnp_top), (lbnp_right, lbnp_top), chamber_col, 2, cv2.LINE_AA)

            # Top Tab with negative pressure reading
            p_display = -abs(state.lbnp)
            l_str = f"LBNP: {p_display:.1f} mmHg"
            tab_w = 110
            cv2.rectangle(db, (cx - tab_w // 2, lbnp_top - 12), (cx + tab_w // 2, lbnp_top + 1), (14, 18, 24), -1)
            cv2.rectangle(db, (cx - tab_w // 2, lbnp_top - 12), (cx + tab_w // 2, lbnp_top + 1), chamber_col, 1, cv2.LINE_AA)
            ts = cv2.getTextSize(l_str, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.putText(db, l_str, (cx - ts[0] // 2, lbnp_top - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, chamber_col, 1, cv2.LINE_AA)

            # Dynamic OUTWARD Negative Pressure Arrows (<- <- BODY -> ->)
            # Calm, slow, smooth gliding motion matching the previous version
            arr_len = int(5 + 6 * intensity)  # 5px to 11px
            travel_range = 20.0
            anim_shift = self.arrow_phase if self.arrow_phase > 0 else (t_mission_s * (6.0 + 7.0 * intensity)) % travel_range

            arrow_levels = [pelvis_y + 14, pelvis_y + 36, knee_y - 4, knee_y + 24, ankle_y - 12]
            for idx, ay in enumerate(arrow_levels):
                phase_shift = (anim_shift + idx * 5.0) % travel_range

                # LEFT SIDE: Origin near left body, moving OUTWARD to LEFT chamber wall (<-)
                start_x_left = cx - 18
                cur_ax_l = int(start_x_left - phase_shift)
                if cur_ax_l - arr_len >= lbnp_left + 3:
                    # Arrow shaft pointing left
                    cv2.line(db, (cur_ax_l, ay), (cur_ax_l - arr_len, ay), chamber_col, 1, cv2.LINE_AA)
                    # Chevron head pointing left
                    cv2.line(db, (cur_ax_l - arr_len, ay), (cur_ax_l - arr_len + 4, ay - 3), chamber_col, 1, cv2.LINE_AA)
                    cv2.line(db, (cur_ax_l - arr_len, ay), (cur_ax_l - arr_len + 4, ay + 3), chamber_col, 1, cv2.LINE_AA)

                # RIGHT SIDE: Origin near right body, moving OUTWARD to RIGHT chamber wall (->)
                start_x_right = cx + 18
                cur_ax_r = int(start_x_right + phase_shift)
                if cur_ax_r + arr_len <= lbnp_right - 3:
                    # Arrow shaft pointing right
                    cv2.line(db, (cur_ax_r, ay), (cur_ax_r + arr_len, ay), chamber_col, 1, cv2.LINE_AA)
                    # Chevron head pointing right
                    cv2.line(db, (cur_ax_r + arr_len, ay), (cur_ax_r + arr_len - 4, ay - 3), chamber_col, 1, cv2.LINE_AA)
                    cv2.line(db, (cur_ax_r + arr_len, ay), (cur_ax_r + arr_len - 4, ay + 3), chamber_col, 1, cv2.LINE_AA)
        else:
            # Standby: Dim dashed outline, minimal activity
            cv2.rectangle(db, (lbnp_left, lbnp_top), (lbnp_right, lbnp_bottom), (38, 46, 58), 1)
            cv2.rectangle(db, (cx - 48, lbnp_top - 12), (cx + 48, lbnp_top + 1), (14, 18, 24), -1)
            cv2.rectangle(db, (cx - 48, lbnp_top - 12), (cx + 48, lbnp_top + 1), (38, 46, 58), 1)
            cv2.putText(db, "LBNP STANDBY", (cx - 42, lbnp_top - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.33, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Schematic Caption
        caption = "MODEL-PREDICTED FLUID REDISTRIBUTION"
        cv2.putText(db, caption, (x + 14, y + h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_telemetry_metrics(
        self,
        db: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        state: CardiovascularState,
    ) -> None:
        """Renders quantitative numeric readouts matching section 5."""
        cv2.rectangle(db, (x, y), (x + w, y + h), (18, 22, 30), -1)
        cv2.rectangle(db, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)

        cv2.putText(db, "LOWER-BODY VENOUS RESERVOIR:", (x + 10, y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.36, self.COLOR_CYAN, 1, cv2.LINE_AA)

        # Row 1: V_Vl and Earth reference
        vvl_str = f"V_Vl: {state.vvl:5.1f} mL"
        ref_str = f"Earth Ref: {state.vvl_earth:5.1f} mL"
        cv2.putText(db, vvl_str, (x + 10, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(db, ref_str, (x + 180, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Row 2: Cardiovascular Deviation
        dev_col = self.COLOR_GREEN if state.is_target_range else self.COLOR_AMBER
        dev_str = f"CARDIOVASCULAR DEVIATION: {state.deviation_pct:+5.1f} %"
        cv2.putText(db, dev_str, (x + 10, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, dev_col, 1, cv2.LINE_AA)

        # Row 3: Prominent Personalized LBNP Pressure Readout & Adaptation Indicator
        p_val = -abs(state.lbnp) if state.is_active else 0.0
        lbnp_str = f"PERSONALIZED LBNP: {p_val:4.1f} mmHg"
        lbnp_col = self.COLOR_AMBER if state.is_active else self.COLOR_TEXT_MUTED
        cv2.putText(db, lbnp_str, (x + 10, y + 80), cv2.FONT_HERSHEY_DUPLEX, 0.42, lbnp_col, 1, cv2.LINE_AA)

        # Dynamic Adaptation Badge (Section 8: "ADAPTING ^" or "ADAPTING v")
        if self.adaptation_text:
            adapt_col = self.COLOR_GREEN if "^" in self.adaptation_text else self.COLOR_AMBER
            cv2.putText(db, f"[{self.adaptation_text}]", (x + 250, y + 80), cv2.FONT_HERSHEY_DUPLEX, 0.38, adapt_col, 1, cv2.LINE_AA)

        # Simulation time indicator
        sim_str = f"Sim Time: {state.sim_time_min:4.1f} min ({self.playback_speed:.0f}x speed)"
        cv2.putText(db, sim_str, (x + 10, y + 99), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _draw_target_restoration_gauge(
        self,
        db: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        state: CardiovascularState,
    ) -> None:
        """Renders restoration toward Earth reference indicator (Section 8)."""
        cv2.rectangle(db, (x, y), (x + w, y + h), (18, 22, 30), -1)
        cv2.rectangle(db, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)

        # Horizontal progress bar: CURRENT Vvl -> EARTH TARGET
        cv2.putText(db, "CURRENT Vvl", (x + 10, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        target_label = f"EARTH TARGET ({int(state.vvl_earth)} mL)"
        cv2.putText(db, target_label, (x + w - 165, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_GREEN, 1, cv2.LINE_AA)

        bar_x = x + 10
        bar_y = y + 22
        bar_w = w - 20
        bar_h = 10

        # Bar background
        cv2.rectangle(db, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 36, 46), -1)
        cv2.rectangle(db, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), self.COLOR_BORDER, 1)

        # Range mapping: 500 mL (0%) to 1050 mL (100%)
        min_v = 500.0
        max_v = 1050.0
        cur_frac = float(np.clip((state.vvl - min_v) / (max_v - min_v), 0.0, 1.0))
        target_frac = float(np.clip((state.vvl_earth - min_v) / (max_v - min_v), 0.0, 1.0))

        # Filled current volume
        fill_col = self.COLOR_GREEN if state.is_target_range else self.COLOR_AMBER
        cur_w = int(bar_w * cur_frac)
        if cur_w > 0:
            cv2.rectangle(db, (bar_x, bar_y), (bar_x + cur_w, bar_y + bar_h), fill_col, -1)

        # Target tick line (Earth target)
        tx = int(bar_x + bar_w * target_frac)
        cv2.line(db, (tx, bar_y - 2), (tx, bar_y + bar_h + 2), (255, 255, 255), 2, cv2.LINE_AA)

        # Status text below bar
        if state.is_target_range:
            status_msg = "EARTH-LIKE TARGET RANGE"
            msg_col = self.COLOR_GREEN
        else:
            status_msg = "RESTORING TOWARD EARTH REFERENCE"
            msg_col = self.COLOR_AMBER

        cv2.putText(db, f"STATUS: {status_msg}", (x + 10, y + 54), cv2.FONT_HERSHEY_DUPLEX, 0.38, msg_col, 1, cv2.LINE_AA)

    def _draw_mini_trend(
        self,
        db: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        state: CardiovascularState,
    ) -> None:
        """Renders mini scrolling trend plot Vvl(t) vs Vvl_earth (Section 9)."""
        cv2.rectangle(db, (x, y), (x + w, y + h), (14, 18, 24), -1)
        cv2.rectangle(db, (x, y), (x + w, y + h), self.COLOR_BORDER, 1)

        # Plot title & scale
        cv2.putText(db, "Vvl(t) DYNAMICS [mL]", (x + 8, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, self.COLOR_CYAN, 1, cv2.LINE_AA)
        cv2.putText(db, "60m Sim", (x + w - 62, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Plot area
        px0 = x + 35
        py0 = y + 22
        pw = w - 45
        ph = h - 32

        # Scale limits: 550 mL to 1000 mL
        v_min, v_max = 550.0, 1000.0

        # Reference line Vvl_earth (e.g. 940 mL)
        ref_y = int(py0 + ph * (1.0 - (state.vvl_earth - v_min) / (v_max - v_min)))
        # Dashed line
        for dash_x in range(px0, px0 + pw, 8):
            cv2.line(db, (dash_x, ref_y), (min(dash_x + 4, px0 + pw), ref_y), self.COLOR_GREEN, 1, cv2.LINE_AA)
        cv2.putText(db, f"{int(state.vvl_earth)}", (x + 6, ref_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.COLOR_GREEN, 1, cv2.LINE_AA)

        # Baseline reference (700 mL)
        base_y = int(py0 + ph * (1.0 - (700.0 - v_min) / (v_max - v_min)))
        cv2.putText(db, "700", (x + 6, base_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Draw full trajectory curve
        times = self.trajectory.times
        vvls = self.trajectory.vvls
        if len(times) > 1:
            total_t = self.trajectory.total_duration_min if self.trajectory.total_duration_min > 0 else 60.0
            pts = []
            for t_i, v_i in zip(times, vvls):
                cx_i = int(px0 + pw * (t_i / total_t))
                cy_i = int(py0 + ph * (1.0 - np.clip((v_i - v_min) / (v_max - v_min), 0.0, 1.0)))
                pts.append((cx_i, cy_i))

            for i in range(len(pts) - 1):
                cv2.line(db, pts[i], pts[i + 1], (70, 90, 110), 1, cv2.LINE_AA)

            # Highlight current playback head
            cur_t = state.sim_time_min
            cur_x = int(px0 + pw * (cur_t / total_t))
            cur_y = int(py0 + ph * (1.0 - np.clip((state.vvl - v_min) / (v_max - v_min), 0.0, 1.0)))
            # Vertical playback needle
            cv2.line(db, (cur_x, py0), (cur_x, py0 + ph), (50, 70, 90), 1, cv2.LINE_AA)
            # Current marker point
            pt_col = self.COLOR_GREEN if state.is_target_range else self.COLOR_AMBER
            cv2.circle(db, (cur_x, cur_y), 4, pt_col, -1, cv2.LINE_AA)
            cv2.circle(db, (cur_x, cur_y), 6, (255, 255, 255), 1, cv2.LINE_AA)
