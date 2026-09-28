"""Unit tests for Affective and FACS behavioral engine."""

import pytest

from aura_face.affective import AffectiveEmotion, AffectiveEngine
from aura_face.calibration import AstronautBaseline


@pytest.fixture
def affective_engine() -> AffectiveEngine:
    baseline = AstronautBaseline(
        subject_id="test_subject",
        au4_baseline=0.08,
        au6_baseline=0.05,
        au12_baseline=0.06,
    )
    return AffectiveEngine(baseline=baseline)


def test_happiness_duchenne_classification(affective_engine: AffectiveEngine):
    """Verifies that high AU12 + AU6 produces Duchenne Happiness."""
    blendshapes = {
        "mouthSmileLeft": 0.60,
        "mouthSmileRight": 0.60,
        "eyeSquintLeft": 0.45,
        "eyeSquintRight": 0.45,
    }
    telemetry = affective_engine.process(blendshapes, timestamp=1.0)

    assert telemetry.emotion == AffectiveEmotion.HAPPINESS
    assert telemetry.is_duchenne is True


def test_anger_stress_classification(affective_engine: AffectiveEngine):
    """Verifies that high AU4 + AU6 produces Anger/Stress."""
    blendshapes = {
        "browDownLeft": 0.65,
        "browDownRight": 0.65,
        "eyeSquintLeft": 0.45,
        "eyeSquintRight": 0.45,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
    }
    telemetry = affective_engine.process(blendshapes, timestamp=1.0)

    assert telemetry.emotion == AffectiveEmotion.ANGER_STRESS


def test_cognitive_load_onset(affective_engine: AffectiveEngine):
    """Verifies that sustained AU4 > 0.30 for > 3.0 seconds triggers COGNITIVE_LOAD_ONSET."""
    blendshapes = {
        "browDownLeft": 0.50,
        "browDownRight": 0.50,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
    }

    # At t=0s, pattern begins
    tel0 = affective_engine.process(blendshapes, timestamp=0.0)
    assert tel0.cognitive_load_active is True
    assert tel0.event_triggered is None

    # At t=2s, still below 3s duration threshold
    tel1 = affective_engine.process(blendshapes, timestamp=2.0)
    assert tel1.event_triggered is None

    # At t=3.5s, duration > 3.0s -> triggers COGNITIVE_LOAD_ONSET event
    tel2 = affective_engine.process(blendshapes, timestamp=3.5)
    assert tel2.event_triggered == "COGNITIVE_LOAD_ONSET"


def test_yawn_detection(affective_engine: AffectiveEngine):
    """Verifies that jawOpen > 0.50 for > 2.0s triggers YAWN event."""
    blendshapes = {
        "jawOpen": 0.70,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
    }

    affective_engine.process(blendshapes, timestamp=0.0)
    affective_engine.process(blendshapes, timestamp=1.0)
    tel = affective_engine.process(blendshapes, timestamp=2.2)

    assert tel.event_triggered == "YAWN"
    assert tel.yawn_count_total == 1


def test_dimensional_valence_and_psychological_state(affective_engine: AffectiveEngine):
    """Verifies Russell Circumplex valence and psychological flight state mapping."""
    # 1. Genuine positive engagement
    happy_blendshapes = {
        "mouthSmileLeft": 0.70,
        "mouthSmileRight": 0.70,
        "eyeSquintLeft": 0.45,
        "eyeSquintRight": 0.45,
    }
    t_happy = affective_engine.process(happy_blendshapes, timestamp=1.0)
    assert t_happy.valence > 0.40
    assert t_happy.psychological_state.value == "Genuine Engagement"

    # 2. Cognitive Strain / Mental Overload
    strain_blendshapes = {
        "browDownLeft": 0.75,
        "browDownRight": 0.75,
        "mouthSmileLeft": 0.0,
        "mouthSmileRight": 0.0,
        "mouthFrownLeft": 0.40,
        "mouthFrownRight": 0.40,
    }
    t_strain = affective_engine.process(strain_blendshapes, timestamp=2.0)
    assert t_strain.valence < -0.20
    assert t_strain.cognitive_workload > 0.40
