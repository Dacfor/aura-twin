"""Unit tests for SQLite storage and telemetry logging."""

import os
import tempfile
import time
from pathlib import Path

import pytest

from aura_face.calibration import AstronautBaseline
from aura_face.storage import EventRecord, TelemetryRecord, TelemetryStorage


@pytest.fixture
def temp_storage():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_telemetry.db"
        storage = TelemetryStorage(db_path=db_path, enable_wal=False)
        yield storage


def test_db_initialization_and_tables(temp_storage: TelemetryStorage):
    """Verifies that calibration, face_telemetry, and events tables are created."""
    with temp_storage._connection() as conn:
        tables = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "calibration" in tables
        assert "face_telemetry" in tables
        assert "events" in tables


def test_calibration_roundtrip(temp_storage: TelemetryStorage):
    """Verifies saving and loading calibrated astronaut baseline."""
    baseline = AstronautBaseline(
        subject_id="artemis_cdr_01",
        created_at=time.time(),
        ear_open=0.32,
        ear_closed=0.13,
        ear_threshold=0.187,
        au4_baseline=0.07,
        au6_baseline=0.04,
        au12_baseline=0.05,
        blink_rate_baseline=17.5,
    )
    row_id = temp_storage.save_calibration(baseline)
    assert row_id > 0

    loaded = temp_storage.load_latest_calibration("artemis_cdr_01")
    assert loaded is not None
    assert loaded.subject_id == "artemis_cdr_01"
    assert pytest.approx(loaded.ear_threshold, rel=1e-3) == 0.187


def test_telemetry_and_events_logging(temp_storage: TelemetryStorage):
    """Verifies logging and retrieving 1 Hz telemetry and events."""
    session_id = "test_lunar_eva_01"

    rec = TelemetryRecord(
        ts=100.0,
        session_id=session_id,
        subject_id="artemis_cdr_01",
        ear=0.28,
        perclos=6.5,
        blink_rate=15.0,
        blink_duration_ms=180.0,
        microsleep_count=0,
        yawn_count=0,
        au4=0.10,
        au6=0.05,
        au12=0.05,
        emotion="Neutral",
        head_yaw=2.1,
        head_pitch=-1.4,
        tracking_quality=1.0,
        state="ALERT",
    )
    temp_storage.log_telemetry(rec)

    ev = EventRecord(
        ts=100.5,
        session_id=session_id,
        event_type="STATE_CHANGE",
        payload={"from": "UNKNOWN", "to": "ALERT", "reason": "System initialized"},
    )
    temp_storage.log_event(ev)

    telemetry_rows = temp_storage.get_session_telemetry(session_id)
    assert len(telemetry_rows) == 1
    assert telemetry_rows[0]["state"] == "ALERT"
    assert telemetry_rows[0]["perclos"] == 6.5

    event_rows = temp_storage.get_session_events(session_id)
    assert len(event_rows) == 1
    assert event_rows[0]["event_type"] == "STATE_CHANGE"
    assert event_rows[0]["payload"]["to"] == "ALERT"
