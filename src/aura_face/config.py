"""Configuration loader and typed dataclasses for AURA-Face."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class SessionConfig:
    subject_id: str = "default_astronaut"
    telemetry_rate_hz: float = 1.0
    db_path: str = "data/aura_face_telemetry.db"
    enable_wal: bool = True


@dataclass
class GatingConfig:
    max_yaw_deg: float = 25.0
    max_pitch_deg: float = 20.0
    min_tracking_quality: float = 0.60
    unknown_timeout_s: float = 10.0
    unknown_recovery_s: float = 10.0


@dataclass
class OcularThresholds:
    perclos_alert_max: float = 8.0
    perclos_moderate_max: float = 15.0
    microsleep_critical_count: int = 2


@dataclass
class OcularConfig:
    ema_alpha: float = 0.6
    blink_min_ms: float = 100.0
    blink_max_ms: float = 400.0
    droop_min_ms: float = 400.0
    droop_max_ms: float = 500.0
    microsleep_min_ms: float = 500.0
    perclos_window_s: float = 60.0
    metrics_window_s: float = 300.0
    thresholds: OcularThresholds = field(default_factory=OcularThresholds)


@dataclass
class SmileConfig:
    au12_smile_threshold: float = 0.30
    au6_duchenne_threshold: float = 0.30
    smile_min_duration_s: float = 0.8
    duchenne_min_duration_s: float = 1.2
    smoothing_window_s: float = 1.5


@dataclass
class CognitiveStrainHysteresis:
    enter_workload: float = 0.60
    enter_duration_s: float = 5.0
    exit_workload: float = 0.40
    exit_duration_s: float = 30.0


@dataclass
class CognitiveLoadConfig:
    au4_min: float = 0.30
    au12_max: float = 0.20
    duration_s: float = 3.0


@dataclass
class YawnConfig:
    jaw_open_min: float = 0.50
    au12_max: float = 0.25
    duration_s: float = 2.0


@dataclass
class PreRestConfig:
    au4_min: float = 0.30
    duration_s: float = 180.0
    exit_absence_s: float = 300.0


@dataclass
class AffectiveConfig:
    au4_threshold: float = 0.40
    au6_threshold: float = 0.30
    au12_threshold: float = 0.35
    au1_threshold: float = 0.35
    au9_threshold: float = 0.35
    au15_threshold: float = 0.35
    brow_outer_up_threshold: float = 0.35
    jaw_open_threshold: float = 0.40
    smile: SmileConfig = field(default_factory=SmileConfig)
    cognitive_strain: CognitiveStrainHysteresis = field(default_factory=CognitiveStrainHysteresis)
    cognitive_load: CognitiveLoadConfig = field(default_factory=CognitiveLoadConfig)
    yawn: YawnConfig = field(default_factory=YawnConfig)
    pre_rest: PreRestConfig = field(default_factory=PreRestConfig)


@dataclass
class CalibrationConfig:
    open_eye_duration_s: float = 10.0
    blink_duration_s: float = 10.0
    ear_threshold_factor: float = 0.70
    default_ear_open: float = 0.30
    default_ear_closed: float = 0.15
    default_ear_threshold: float = 0.20


@dataclass
class StatesConfig:
    moderate_trigger_delay_s: float = 10.0
    critical_trigger_delay_s: float = 0.0
    recovery_dwell_s: float = 30.0


@dataclass
class VisionConfig:
    model_path: str = "models/face_landmarker.task"
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    backend: str = "mediapipe"  # "mediapipe" | "pyfeat"
    device: str = "cpu"         # "cpu" | "cuda"


@dataclass
class HudMarkerConfig:
    default_marker_mode: str = "MINIMAL"  # "OFF" | "MINIMAL" | "DETAILED" | "ALL"
    show_eye_contours: bool = True
    show_face_brackets: bool = True
    show_head_pose_axis: bool = False
    show_debug_labels: bool = False


@dataclass
class CardiovascularConfig:
    enabled: bool = True
    csv_path: str = "data/cardiovascular_demo_C.csv"
    vvl_earth: float = 940.0
    playback_speed: float = 30.0
    loop: bool = True
    tolerance_pct: float = 2.0


@dataclass
class AuraConfig:
    session: SessionConfig = field(default_factory=SessionConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    ocular: OcularConfig = field(default_factory=OcularConfig)
    affective: AffectiveConfig = field(default_factory=AffectiveConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    states: StatesConfig = field(default_factory=StatesConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    markers: HudMarkerConfig = field(default_factory=HudMarkerConfig)
    cardiovascular: CardiovascularConfig = field(default_factory=CardiovascularConfig)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> AuraConfig:
        """Loads configuration from YAML file or falls back to defaults."""
        if config_path is None:
            # Check default locations
            candidates = [
                Path("config/default.yaml"),
                Path(__file__).parent.parent.parent / "config" / "default.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break

        if config_path is None or not Path(config_path).exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        return cls.from_dict(raw_data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuraConfig:
        """Builds an AuraConfig tree from dictionary."""
        sess_data = data.get("session", {})
        gating_data = data.get("gating", {})
        oc_data = data.get("ocular", {})
        oc_thresh_data = oc_data.get("thresholds", {})
        aff_data = data.get("affective", {})
        cog_data = aff_data.get("cognitive_load", {})
        smile_data = aff_data.get("smile", {})
        strain_data = aff_data.get("cognitive_strain", {})
        yawn_data = aff_data.get("yawn", {})
        pr_data = aff_data.get("pre_rest", {})
        cal_data = data.get("calibration", {})
        st_data = data.get("states", {})
        vis_data = data.get("vision", {})
        markers_data = data.get("markers", {})
        cardio_data = data.get("cardiovascular", {})

        return cls(
            session=SessionConfig(**sess_data),
            gating=GatingConfig(**gating_data),
            ocular=OcularConfig(
                ema_alpha=oc_data.get("ema_alpha", 0.6),
                blink_min_ms=oc_data.get("blink_min_ms", 100.0),
                blink_max_ms=oc_data.get("blink_max_ms", 400.0),
                droop_min_ms=oc_data.get("droop_min_ms", 400.0),
                droop_max_ms=oc_data.get("droop_max_ms", 500.0),
                microsleep_min_ms=oc_data.get("microsleep_min_ms", 500.0),
                perclos_window_s=oc_data.get("perclos_window_s", 60.0),
                metrics_window_s=oc_data.get("metrics_window_s", 300.0),
                thresholds=OcularThresholds(**oc_thresh_data),
            ),
            affective=AffectiveConfig(
                au4_threshold=aff_data.get("au4_threshold", 0.40),
                au6_threshold=aff_data.get("au6_threshold", 0.30),
                au12_threshold=aff_data.get("au12_threshold", 0.35),
                au1_threshold=aff_data.get("au1_threshold", 0.35),
                au9_threshold=aff_data.get("au9_threshold", 0.35),
                au15_threshold=aff_data.get("au15_threshold", 0.35),
                brow_outer_up_threshold=aff_data.get("brow_outer_up_threshold", 0.35),
                jaw_open_threshold=aff_data.get("jaw_open_threshold", 0.40),
                smile=SmileConfig(**smile_data),
                cognitive_strain=CognitiveStrainHysteresis(**strain_data),
                cognitive_load=CognitiveLoadConfig(**cog_data),
                yawn=YawnConfig(**yawn_data),
                pre_rest=PreRestConfig(**pr_data),
            ),
            calibration=CalibrationConfig(**cal_data),
            states=StatesConfig(**st_data),
            vision=VisionConfig(**vis_data),
            markers=HudMarkerConfig(**markers_data),
            cardiovascular=CardiovascularConfig(**cardio_data),
        )
