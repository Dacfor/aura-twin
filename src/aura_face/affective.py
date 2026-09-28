"""Observable Facial and Ocular Behavioral Engine for AURA-Face.

Measures observable facial and ocular behavior and derives explainable operational
proxies for vigilance, fatigue, cognitive strain, and positive engagement.
It does not diagnose psychological conditions or infer internal emotions with certainty.

Scientific Grounding:
1. Barrett, L. F. et al. (2019):
   "Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements."
   Psychological Science in the Public Interest, 20(1), 1-68.
   -> Justifies objective behavioral proxies and XAI evidence over naive emotion classification.
2. Russell, J. A. (1980):
   "A Circumplex Model of Affect."
   Journal of Personality and Social Psychology, 39(6), 1161-1178.
   -> Continuous orthogonal space: Facial Valence Proxy and Operational Arousal Proxy.
3. Dinges, D. F. & Metaxas, D. N. et al. (2005, 2007):
   "Monitoring of facial stress during spaceflight." Acta Astronautica, 60(4-7), 341-346.
   -> AU4 (Brow Lowerer) and periocular tension as indicators of acute cognitive effort.
4. Schleicher, R., Galley, N. et al. (2008):
   "Blinks and saccades as indicators of fatigue in computerized tasks."
   British Journal of Ophthalmology, 92(5), 682-687.
   -> Cognitive blink suppression during intense focus vs slow droops in circadian drowsiness.
5. Ekman, P. & Friesen, W. V. (1978, 1984):
   Facial Action Coding System (FACS) and Emotional FACS (EMFACS).
   -> Duchenne behavioral pattern: co-activation of AU12 (zygomatic) and AU6 (orbicularis oculi).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from aura_face.calibration import AstronautBaseline
from aura_face.config import AffectiveConfig


class AffectiveEmotion(str, Enum):
    """FACS prototype behavioral descriptor (retained for backward compatibility)."""
    NEUTRAL = "Neutral"
    HAPPINESS = "Happiness"
    ANGER_STRESS = "Anger/Stress"
    SADNESS = "Sadness"
    DISGUST = "Disgust"
    SURPRISE = "Surprise"


class OperationalBehaviorPattern(str, Enum):
    """Observable operational behavioral patterns (Barrett 2019; Dinges 2005)."""
    CALM_ALERT = "CALM_ALERT"
    SMILE_PATTERN = "SMILE_PATTERN"
    DUCHENNE_PATTERN_CANDIDATE = "DUCHENNE_PATTERN_CANDIDATE"
    AU4_TENSION_PATTERN = "AU4_TENSION_PATTERN"
    COGNITIVE_STRAIN_CANDIDATE = "COGNITIVE_STRAIN_CANDIDATE"
    DROWSINESS_IMPAIRMENT = "DROWSINESS_IMPAIRMENT"
    UNKNOWN = "UNKNOWN"
    UNKNOWN_DEGRADED = "UNKNOWN_DEGRADED"


class PsychologicalFlightState(str, Enum):
    """Operational flight state mapping."""
    CALM_ALERT = "Calm Alert"
    FOCUSED_FLOW = "Focused Flow"
    COGNITIVE_STRAIN = "Cognitive Strain"
    GENUINE_ENGAGEMENT = "Genuine Engagement"
    DROWSINESS_IMPAIRMENT = "Drowsiness Impairment"
    HYPOVIGILANCE_BOREDOM = "Hypovigilance"
    ISOLATION_EXHAUSTION = "Isolation Exhaustion"


@dataclass
class ActionUnitsVector:
    """Normalized FACS Action Units (0.0 - 1.0)."""
    au1: float = 0.0           # Inner Brow Raiser (browInnerUp)
    au4: float = 0.0           # Brow Lowerer (browDown) - Mental Effort / Frustration
    au6: float = 0.0           # Cheek Raiser (eyeSquint) - Duchenne smile component
    au9: float = 0.0           # Nose Wrinkler (noseSneer) - Disgust
    au12: float = 0.0          # Lip Corner Puller (mouthSmile) - Positive Valence
    au15: float = 0.0          # Lip Corner Depressor (mouthFrown) - Sadness
    brow_outer_up: float = 0.0 # Outer Brow Raiser
    jaw_open: float = 0.0      # Jaw Drop (jawOpen)


@dataclass
class AffectiveTelemetry:
    """Comprehensive snapshot of affective, dimensional, and behavioral metrics."""
    timestamp: float
    raw_aus: Dict[str, float]
    norm_aus: ActionUnitsVector
    emotion: AffectiveEmotion
    is_duchenne: bool
    
    # Behavioral Proxies
    valence: float = 0.0                          # Legacy alias for facial_valence_proxy
    arousal: float = 0.0                          # Legacy alias for operational_arousal_proxy
    cognitive_workload: float = 0.0               # Legacy alias for cognitive_workload_proxy
    facial_valence_proxy: float = 0.0             # Facial Valence Proxy: -1.0 to +1.0
    operational_arousal_proxy: float = 0.0        # Operational Arousal Proxy: 0.0 to 1.0
    cognitive_workload_proxy: float = 0.0         # Cognitive Workload Proxy: 0.0 to 1.0
    confidence: float = 1.0                       # Explainability Confidence: 0.0 to 1.0
    evidence: List[str] = field(default_factory=list) # Explainable XAI evidence strings
    
    # Operational Patterns
    current_pattern: OperationalBehaviorPattern = OperationalBehaviorPattern.CALM_ALERT
    psychological_state: PsychologicalFlightState = PsychologicalFlightState.CALM_ALERT
    
    # Pattern activation flags
    smile_pattern_active: bool = False
    duchenne_candidate_active: bool = False
    au4_tension_active: bool = False
    cognitive_strain_candidate_active: bool = False
    cognitive_load_active: bool = False
    cognitive_load_sustained_s: float = 0.0
    yawn_active: bool = False
    yawn_count_total: int = 0
    pre_rest_pattern_active: bool = False
    event_triggered: Optional[str] = None

    @property
    def operational_pattern(self) -> OperationalBehaviorPattern:
        return self.current_pattern

    @property
    def cognitive_strain_candidate(self) -> bool:
        return self.cognitive_strain_candidate_active



class AffectiveEngine:
    """Extracts, normalizes, and classifies facial behavior patterns with temporal stability."""

    def __init__(
        self,
        config: Optional[AffectiveConfig] = None,
        baseline: Optional[AstronautBaseline] = None,
    ) -> None:
        self.config = config or AffectiveConfig()
        self.baseline = baseline or AstronautBaseline()

        # Cognitive load tracking (legacy AU4 sustained)
        self._cog_load_active = False
        self._cog_load_start_ts: Optional[float] = None
        self._cog_load_event_fired = False

        # Yawn tracking
        self._yawn_active = False
        self._yawn_start_ts: Optional[float] = None
        self._yawn_count = 0
        self._yawn_event_fired = False

        # Pre-rest pattern tracking
        self._pre_rest_active = False
        self._pre_rest_start_ts: Optional[float] = None
        self._last_pattern_ts: Optional[float] = None

        # Temporal persistence: Smile (AU12 >= 0.30 for >= 0.8s)
        self._smile_start_ts: Optional[float] = None
        self._smile_active = False
        self._smile_event_fired = False

        # Temporal persistence: Duchenne (AU12 & AU6 >= 0.30 for >= 1.2s)
        self._duchenne_start_ts: Optional[float] = None
        self._duchenne_active = False
        self._duchenne_event_fired = False

        # AU4 Tension tracking
        self._au4_tension_start_ts: Optional[float] = None
        self._au4_tension_active = False
        self._au4_tension_event_fired = False

        # Cognitive Strain Candidate Hysteresis (enter >= 5s, exit >= 30s)
        self._strain_start_ts: Optional[float] = None
        self._strain_exit_start_ts: Optional[float] = None
        self._strain_active = False
        self._strain_event_fired = False

    def update_baseline(self, baseline: AstronautBaseline) -> None:
        """Updates baseline reference parameters from calibration."""
        self.baseline = baseline

    def extract_raw_aus(self, blendshapes: Dict[str, float]) -> Dict[str, float]:
        """Maps MediaPipe/PyFeat blendshape scores to canonical FACS Action Units."""
        au4_left = blendshapes.get("browDownLeft", 0.0)
        au4_right = blendshapes.get("browDownRight", 0.0)
        au4 = max(au4_left, au4_right)

        au6_left = blendshapes.get("eyeSquintLeft", 0.0)
        au6_right = blendshapes.get("eyeSquintRight", 0.0)
        au6 = (au6_left + au6_right) / 2.0

        au12_left = blendshapes.get("mouthSmileLeft", 0.0)
        au12_right = blendshapes.get("mouthSmileRight", 0.0)
        au12 = (au12_left + au12_right) / 2.0

        au1 = blendshapes.get("browInnerUp", 0.0)

        au9_left = blendshapes.get("noseSneerLeft", 0.0)
        au9_right = blendshapes.get("noseSneerRight", 0.0)
        au9 = (au9_left + au9_right) / 2.0

        au15_left = blendshapes.get("mouthFrownLeft", 0.0)
        au15_right = blendshapes.get("mouthFrownRight", 0.0)
        au15 = (au15_left + au15_right) / 2.0

        brow_outer_left = blendshapes.get("browOuterUpLeft", 0.0)
        brow_outer_right = blendshapes.get("browOuterUpRight", 0.0)
        brow_outer_up = (brow_outer_left + brow_outer_right) / 2.0

        jaw_open = blendshapes.get("jawOpen", 0.0)

        return {
            "AU1": float(au1),
            "AU4": float(au4),
            "AU6": float(au6),
            "AU9": float(au9),
            "AU12": float(au12),
            "AU15": float(au15),
            "browOuterUp": float(brow_outer_up),
            "jawOpen": float(jaw_open),
        }

    def normalize_au(self, raw_value: float, baseline_value: float) -> float:
        """Normalizes raw Action Unit relative to individual astronaut baseline."""
        if baseline_value >= 0.95:
            return 0.0
        norm = (raw_value - baseline_value) / (1.0 - baseline_value)
        return float(np.clip(norm, 0.0, 1.0))

    def compute_facial_valence_proxy(self, aus: ActionUnitsVector) -> float:
        """Computes continuous Facial Valence Proxy in [-1.0, +1.0] (Russell 1980; Barrett 2019)."""
        raw_val = aus.au12 - max(0.8 * aus.au4, aus.au15, aus.au9)
        return float(np.clip(raw_val, -1.0, 1.0))

    def compute_operational_arousal_proxy(
        self,
        aus: ActionUnitsVector,
        ear_smooth: Optional[float] = None,
        blink_rate: Optional[float] = None,
    ) -> float:
        """Computes Operational Arousal Proxy in [0.0, 1.0] with weights summing to 1.0.

        Formula: 0.45 * muscle_activity + 0.30 * eye_openness + 0.25 * blink_dynamics
        """
        # 1. Muscle activity (AU4, jawOpen, AU1)
        muscle_activity_norm = float(np.clip(aus.au4 * 0.40 + aus.jaw_open * 0.35 + aus.au1 * 0.25, 0.0, 1.0))

        # 2. Eye openness
        eye_openness_norm = 0.50
        if ear_smooth is not None and self.baseline.ear_open > 0:
            eye_openness_norm = float(np.clip(ear_smooth / self.baseline.ear_open, 0.0, 1.0))

        # 3. Blink dynamics
        blink_dynamics_norm = 0.50
        if blink_rate is not None:
            blink_dynamics_norm = float(np.clip(blink_rate / 32.0, 0.0, 1.0))

        arousal_proxy = 0.45 * muscle_activity_norm + 0.30 * eye_openness_norm + 0.25 * blink_dynamics_norm
        return float(np.clip(arousal_proxy, 0.0, 1.0))

    def compute_cognitive_workload_proxy(
        self,
        aus: ActionUnitsVector,
        sustained_au4_s: float = 0.0,
        blink_rate: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """Estimates Cognitive Workload Proxy in [0.0, 1.0] (Dinges 2005; Schleicher 2008).

        Formula: 0.55 * au4_norm + 0.30 * au4_sustain_norm + 0.15 * blink_inhibition_norm
        """
        au4_norm = float(np.clip(aus.au4, 0.0, 1.0))
        au4_sustain_norm = float(np.clip(sustained_au4_s / 5.0, 0.0, 1.0))

        blink_inhibition_norm = 0.0
        reliable = True
        if blink_rate is not None:
            # Cognitive blink suppression during visual focus (<10 BPM)
            if blink_rate < 10.0 and aus.au4 > 0.15:
                blink_inhibition_norm = float(np.clip((10.0 - blink_rate) / 10.0, 0.0, 1.0))
        else:
            reliable = False

        cw_proxy = 0.55 * au4_norm + 0.30 * au4_sustain_norm + 0.15 * blink_inhibition_norm
        return float(np.clip(cw_proxy, 0.0, 1.0)), reliable

    def process(
        self,
        blendshapes: Dict[str, float],
        timestamp: float,
        ear_smooth: Optional[float] = None,
        blink_rate: Optional[float] = None,
        perclos: float = 0.0,
        tracking_quality: float = 1.0,
    ) -> AffectiveTelemetry:
        """Processes facial blendshapes and computes explainable behavioral proxies."""
        raw = self.extract_raw_aus(blendshapes)

        # Baseline normalization
        norm_au1 = self.normalize_au(raw["AU1"], 0.05)
        norm_au4 = self.normalize_au(raw["AU4"], self.baseline.au4_baseline)
        norm_au6 = self.normalize_au(raw["AU6"], self.baseline.au6_baseline)
        norm_au9 = self.normalize_au(raw["AU9"], 0.05)
        norm_au12 = self.normalize_au(raw["AU12"], self.baseline.au12_baseline)
        norm_au15 = self.normalize_au(raw["AU15"], 0.05)
        norm_brow_outer = self.normalize_au(raw["browOuterUp"], 0.05)
        norm_jaw_open = float(raw["jawOpen"])

        aus = ActionUnitsVector(
            au1=norm_au1,
            au4=norm_au4,
            au6=norm_au6,
            au9=norm_au9,
            au12=norm_au12,
            au15=norm_au15,
            brow_outer_up=norm_brow_outer,
            jaw_open=norm_jaw_open,
        )

        event_triggered: Optional[str] = None
        evidence: List[str] = []

        # 1. Behavioral Proxies
        facial_val = self.compute_facial_valence_proxy(aus)
        op_arousal = self.compute_operational_arousal_proxy(aus, ear_smooth=ear_smooth, blink_rate=blink_rate)

        # Sustained AU4 tracking
        if aus.au4 >= self.config.cognitive_load.au4_min:
            if self._cog_load_start_ts is None:
                self._cog_load_start_ts = timestamp
            sustained_au4_s = timestamp - self._cog_load_start_ts
        else:
            self._cog_load_start_ts = None
            sustained_au4_s = 0.0

        cog_workload, blink_rel = self.compute_cognitive_workload_proxy(
            aus, sustained_au4_s=sustained_au4_s, blink_rate=blink_rate
        )

        # 2. Confidence Estimation
        confidence = float(np.clip(tracking_quality, 0.0, 1.0))
        if not blink_rel:
            confidence = max(0.20, confidence - 0.05)
        if tracking_quality < 0.60:
            confidence = min(confidence, 0.35)

        # 3. Temporal Persistence: Smile Pattern (AU12 >= 0.30 for >= 0.8s)
        smile_cfg = self.config.smile
        if aus.au12 >= smile_cfg.au12_smile_threshold:
            if self._smile_start_ts is None:
                self._smile_start_ts = timestamp
            smile_dur = timestamp - self._smile_start_ts
            if smile_dur >= smile_cfg.smile_min_duration_s:
                self._smile_active = True
                if not self._smile_event_fired:
                    event_triggered = "SMILE_ONSET"
                    self._smile_event_fired = True
        else:
            self._smile_start_ts = None
            self._smile_active = False
            self._smile_event_fired = False

        # 4. Temporal Persistence: Duchenne Candidate (AU12 >= 0.30 & AU6 >= 0.30 for >= 1.2s)
        if (aus.au12 >= smile_cfg.au12_smile_threshold and aus.au6 >= smile_cfg.au6_duchenne_threshold):
            if self._duchenne_start_ts is None:
                self._duchenne_start_ts = timestamp
            duchenne_dur = timestamp - self._duchenne_start_ts
            if duchenne_dur >= smile_cfg.duchenne_min_duration_s:
                self._duchenne_active = True
                if not self._duchenne_event_fired:
                    event_triggered = "DUCHENNE_PATTERN_CANDIDATE"
                    self._duchenne_event_fired = True
        else:
            self._duchenne_start_ts = None
            self._duchenne_active = False
            self._duchenne_event_fired = False

        # 5. Cognitive Strain Candidate Hysteresis (Enter >= 0.60 for >= 5s; Exit < 0.40 for >= 30s)
        strain_cfg = self.config.cognitive_strain
        is_in_strain_condition = (
            cog_workload >= strain_cfg.enter_workload
            and sustained_au4_s >= strain_cfg.enter_duration_s
            and tracking_quality >= 0.60
            and facial_val < -0.10
        )
        if is_in_strain_condition:
            if not self._strain_active:
                self._strain_active = True
                if not self._strain_event_fired:
                    event_triggered = "COGNITIVE_STRAIN_ONSET"
                    self._strain_event_fired = True
            self._strain_exit_start_ts = None
        elif self._strain_active:
            if cog_workload < strain_cfg.exit_workload:
                if self._strain_exit_start_ts is None:
                    self._strain_exit_start_ts = timestamp
                if (timestamp - self._strain_exit_start_ts) >= strain_cfg.exit_duration_s:
                    self._strain_active = False
                    self._strain_exit_start_ts = None
                    self._strain_event_fired = False
            else:
                self._strain_exit_start_ts = None


        # 6. AU4 Tension Pattern (Elevated AU4 without negative valence / cognitive strain)
        if aus.au4 >= 0.30 and not self._strain_active and facial_val >= -0.10:
            if self._au4_tension_start_ts is None:
                self._au4_tension_start_ts = timestamp
            if (timestamp - self._au4_tension_start_ts) >= 3.0:
                self._au4_tension_active = True
                if not self._au4_tension_event_fired:
                    if event_triggered is None:
                        event_triggered = "AU4_TENSION_ONSET"
                    self._au4_tension_event_fired = True
        else:
            self._au4_tension_start_ts = None
            self._au4_tension_active = False
            self._au4_tension_event_fired = False

        # 7. Legacy Cognitive Load Onset
        cog_cfg = self.config.cognitive_load
        if aus.au4 >= cog_cfg.au4_min and aus.au12 <= cog_cfg.au12_max:
            if not self._cog_load_active:
                self._cog_load_active = True
                self._cog_load_start_ts = timestamp
                self._cog_load_event_fired = False
            start_t = self._cog_load_start_ts if self._cog_load_start_ts is not None else timestamp
            sustained_s = timestamp - start_t
            if sustained_s >= cog_cfg.duration_s and not self._cog_load_event_fired:
                if event_triggered is None:
                    event_triggered = "COGNITIVE_LOAD_ONSET"
                self._cog_load_event_fired = True
        else:
            self._cog_load_active = False
            self._cog_load_start_ts = None
            self._cog_load_event_fired = False

        # 8. Yawn Pattern
        yawn_cfg = self.config.yawn
        if norm_jaw_open >= yawn_cfg.jaw_open_min and aus.au12 <= yawn_cfg.au12_max:
            if not self._yawn_active:
                self._yawn_active = True
                self._yawn_start_ts = timestamp
                self._yawn_event_fired = False
            start_y = self._yawn_start_ts if self._yawn_start_ts is not None else timestamp
            if (timestamp - start_y) >= yawn_cfg.duration_s and not self._yawn_event_fired:
                self._yawn_count += 1
                if event_triggered is None:
                    event_triggered = "YAWN"
                self._yawn_event_fired = True
        else:
            self._yawn_active = False
            self._yawn_start_ts = None
            self._yawn_event_fired = False

        # 9. Pre-Rest Pattern
        pr_cfg = self.config.pre_rest
        if aus.au4 >= pr_cfg.au4_min and aus.au12 <= 0.15:
            self._last_pattern_ts = timestamp
            if self._pre_rest_start_ts is None:
                self._pre_rest_start_ts = timestamp
            if (timestamp - self._pre_rest_start_ts) >= pr_cfg.duration_s:
                self._pre_rest_active = True
        else:
            if self._last_pattern_ts and (timestamp - self._last_pattern_ts) >= pr_cfg.exit_absence_s:
                self._pre_rest_active = False
                self._pre_rest_start_ts = None

        # Legacy Emotion mapping for backward compatibility
        cfg = self.config
        legacy_emotion = AffectiveEmotion.NEUTRAL
        if norm_au12 > cfg.au12_threshold:
            legacy_emotion = AffectiveEmotion.HAPPINESS
        elif norm_au4 > cfg.au4_threshold and norm_au6 > cfg.au6_threshold:
            legacy_emotion = AffectiveEmotion.ANGER_STRESS
        elif norm_au1 > cfg.au1_threshold and norm_au15 > cfg.au15_threshold:
            legacy_emotion = AffectiveEmotion.SADNESS
        elif norm_au9 > cfg.au9_threshold:
            legacy_emotion = AffectiveEmotion.DISGUST
        elif norm_brow_outer > cfg.brow_outer_up_threshold and norm_jaw_open > cfg.jaw_open_threshold:
            legacy_emotion = AffectiveEmotion.SURPRISE

        is_duchenne = bool(norm_au12 > cfg.au12_threshold and norm_au6 > cfg.au6_threshold)

        # 10. Classify Current Operational Pattern (Priority hierarchy)
        if tracking_quality < 0.60:
            current_pattern = OperationalBehaviorPattern.UNKNOWN_DEGRADED
            psych_state = PsychologicalFlightState.CALM_ALERT
            confidence = min(confidence, 0.30)
            evidence.append(f"Degraded tracking quality ({tracking_quality:.2f} < 0.60)")
        elif perclos >= 10.0 or (perclos >= 8.0 and op_arousal < 0.25):
            current_pattern = OperationalBehaviorPattern.DROWSINESS_IMPAIRMENT
            psych_state = PsychologicalFlightState.DROWSINESS_IMPAIRMENT
            evidence.append(f"PERCLOS elevated at {perclos:.1f}% (slowing closures)")
        elif self._duchenne_active or (is_duchenne and facial_val > 0.30):
            current_pattern = OperationalBehaviorPattern.DUCHENNE_PATTERN_CANDIDATE if self._duchenne_active else OperationalBehaviorPattern.CALM_ALERT
            psych_state = PsychologicalFlightState.GENUINE_ENGAGEMENT
            evidence.append("AU12 + AU6 co-activation (Duchenne marker)")
        elif self._smile_active:
            current_pattern = OperationalBehaviorPattern.SMILE_PATTERN
            psych_state = PsychologicalFlightState.CALM_ALERT
            evidence.append("AU12 sustained >=0.8s (Smile pattern)")
        elif self._strain_active:
            current_pattern = OperationalBehaviorPattern.COGNITIVE_STRAIN_CANDIDATE
            psych_state = PsychologicalFlightState.COGNITIVE_STRAIN
            evidence.append(f"AU4 sustained for {sustained_au4_s:.1f}s with workload {cog_workload:.2f}")
        elif self._au4_tension_active:
            current_pattern = OperationalBehaviorPattern.AU4_TENSION_PATTERN
            psych_state = PsychologicalFlightState.FOCUSED_FLOW
            evidence.append(f"AU4 elevated at {aus.au4:.2f} without negative valence")
        else:
            current_pattern = OperationalBehaviorPattern.CALM_ALERT
            psych_state = PsychologicalFlightState.CALM_ALERT
            evidence.append("Facial musculature nominal, balanced vigilance")

        evidence.append(f"Facial Valence Proxy: {facial_val:+.2f}")
        evidence.append(f"Operational Arousal Proxy: {op_arousal:.2f}")
        evidence.append(f"Cognitive Workload Proxy: {cog_workload:.2f}")
        evidence.append(f"Tracking Quality: {tracking_quality:.2f}")

        return AffectiveTelemetry(
            timestamp=timestamp,
            raw_aus=raw,
            norm_aus=aus,
            emotion=legacy_emotion,
            is_duchenne=is_duchenne,
            valence=facial_val,
            arousal=op_arousal,
            cognitive_workload=cog_workload,
            facial_valence_proxy=facial_val,
            operational_arousal_proxy=op_arousal,
            cognitive_workload_proxy=cog_workload,
            confidence=confidence,
            evidence=evidence,
            current_pattern=current_pattern,
            psychological_state=psych_state,
            smile_pattern_active=self._smile_active,
            duchenne_candidate_active=self._duchenne_active,
            au4_tension_active=self._au4_tension_active,
            cognitive_strain_candidate_active=self._strain_active,
            cognitive_load_active=self._cog_load_active,
            cognitive_load_sustained_s=sustained_au4_s,
            yawn_active=self._yawn_active,
            yawn_count_total=self._yawn_count,
            pre_rest_pattern_active=self._pre_rest_active,
            event_triggered=event_triggered,
        )

    def reset(self) -> None:
        """Resets engine temporal states."""
        self._cog_load_active = False
        self._cog_load_start_ts = None
        self._cog_load_event_fired = False
        self._yawn_active = False
        self._yawn_start_ts = None
        self._yawn_count = 0
        self._yawn_event_fired = False
        self._pre_rest_active = False
        self._pre_rest_start_ts = None
        self._last_pattern_ts = None
        self._smile_start_ts = None
        self._smile_active = False
        self._smile_event_fired = False
        self._duchenne_start_ts = None
        self._duchenne_active = False
        self._duchenne_event_fired = False
        self._au4_tension_start_ts = None
        self._au4_tension_active = False
        self._au4_tension_event_fired = False
        self._strain_start_ts = None
        self._strain_exit_start_ts = None
        self._strain_active = False
        self._strain_event_fired = False
