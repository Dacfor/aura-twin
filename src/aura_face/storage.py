"""SQLite WAL 1 Hz Telemetry & Event Storage for AURA-Face.

Implements lightweight, robust local database storage adhering to the AURA-Face
data contract specification.

Privacy by Design Guarantee:
- Stores strictly numeric time-series vectors and categorical state labels.
- Zero frame data, zero pixel buffers, zero biometric embedding vectors.
- Runs 100% offline at the edge (no external cloud connections).

Scientific grounding:
- Designed to feed the Habitat Multimodal Digital Twin (HRV, sleep, biodynamic lighting).
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from aura_face.calibration import AstronautBaseline


@dataclass
class TelemetryRecord:
    """Telemetry packet logged once per second (1 Hz)."""
    ts: float
    session_id: str
    subject_id: str
    ear: float
    perclos: float
    blink_rate: float
    blink_duration_ms: float
    microsleep_count: int
    yawn_count: int
    au4: float
    au6: float
    au12: float
    emotion: str
    head_yaw: float
    head_pitch: float
    tracking_quality: float
    state: str  # ALERT | MODERATE | CRITICAL | PRE_REST | UNKNOWN
    valence: float = 0.0
    arousal: float = 0.0
    cognitive_workload: float = 0.0
    psych_state: str = "Calm Alert"
    valence_proxy: float = 0.0
    arousal_proxy: float = 0.0
    cognitive_workload_proxy: float = 0.0
    confidence: float = 1.0
    evidence_json: str = "[]"
    marker_mode: str = "MINIMAL"


@dataclass
class EventRecord:
    """Discrete behavioral or state event."""
    ts: float
    session_id: str
    event_type: str  # MICRO_SLEEP | YAWN | STATE_CHANGE | COGNITIVE_LOAD_ONSET
    payload: Dict[str, Any]


class TelemetryStorage:
    """Thread-safe SQLite storage engine running in WAL mode."""

    def __init__(
        self,
        db_path: str | Path = "data/aura_face_telemetry.db",
        enable_wal: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.enable_wal = enable_wal
        self._init_db()

    @contextlib.contextmanager
    def _connection(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        if self.enable_wal:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes tables and indexes per specification."""
        with self._connection() as conn:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS calibration (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        subject_id TEXT DEFAULT 'default',
                        ear_open REAL,
                        ear_closed REAL,
                        ear_threshold REAL,
                        au4_baseline REAL,
                        au12_baseline REAL,
                        au6_baseline REAL,
                        blink_rate_baseline REAL
                    );
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS face_telemetry (
                        ts REAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        subject_id TEXT DEFAULT 'default',
                        ear REAL,
                        perclos REAL,
                        blink_rate REAL,
                        blink_duration_ms REAL,
                        microsleep_count INTEGER,
                        yawn_count INTEGER,
                        au4 REAL,
                        au6 REAL,
                        au12 REAL,
                        emotion TEXT,
                        head_yaw REAL,
                        head_pitch REAL,
                        tracking_quality REAL,
                        state TEXT NOT NULL,
                        valence REAL DEFAULT 0.0,
                        arousal REAL DEFAULT 0.0,
                        cognitive_workload REAL DEFAULT 0.0,
                        psych_state TEXT DEFAULT 'Calm Alert'
                    );
                """)

                # Non-breaking migration: ensure columns exist for existing tables
                for col, col_type in [
                    ("valence", "REAL DEFAULT 0.0"),
                    ("arousal", "REAL DEFAULT 0.0"),
                    ("cognitive_workload", "REAL DEFAULT 0.0"),
                    ("psych_state", "TEXT DEFAULT 'Calm Alert'"),
                    ("valence_proxy", "REAL DEFAULT 0.0"),
                    ("arousal_proxy", "REAL DEFAULT 0.0"),
                    ("cognitive_workload_proxy", "REAL DEFAULT 0.0"),
                    ("confidence", "REAL DEFAULT 1.0"),
                    ("evidence_json", "TEXT DEFAULT '[]'"),
                    ("marker_mode", "TEXT DEFAULT 'MINIMAL'"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE face_telemetry ADD COLUMN {col} {col_type};")
                    except sqlite3.OperationalError:
                        pass

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        ts REAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        payload TEXT
                    );
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ftel ON face_telemetry(session_id, ts);
                """)

    def save_calibration(self, baseline: AstronautBaseline) -> int:
        """Saves calibrated astronaut baseline."""
        with self._connection() as conn:
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO calibration (
                        created_at, subject_id, ear_open, ear_closed, ear_threshold,
                        au4_baseline, au12_baseline, au6_baseline, blink_rate_baseline
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(baseline.created_at)),
                        baseline.subject_id,
                        baseline.ear_open,
                        baseline.ear_closed,
                        baseline.ear_threshold,
                        baseline.au4_baseline,
                        baseline.au12_baseline,
                        baseline.au6_baseline,
                        baseline.blink_rate_baseline,
                    ),
                )
                return int(cur.lastrowid or 0)

    def load_latest_calibration(self, subject_id: str) -> Optional[AstronautBaseline]:
        """Loads the most recent baseline calibration for the given astronaut."""
        with self._connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM calibration
                WHERE subject_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (subject_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return AstronautBaseline(
                subject_id=row["subject_id"],
                ear_open=row["ear_open"],
                ear_closed=row["ear_closed"],
                ear_threshold=row["ear_threshold"],
                au4_baseline=row["au4_baseline"],
                au6_baseline=row["au6_baseline"],
                au12_baseline=row["au12_baseline"],
                blink_rate_baseline=row["blink_rate_baseline"],
            )

    def log_telemetry(self, record: TelemetryRecord) -> None:
        """Logs a single 1 Hz telemetry row."""
        with self._connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO face_telemetry (
                        ts, session_id, subject_id, ear, perclos, blink_rate, blink_duration_ms,
                        microsleep_count, yawn_count, au4, au6, au12, emotion,
                        head_yaw, head_pitch, tracking_quality, state,
                        valence, arousal, cognitive_workload, psych_state,
                        valence_proxy, arousal_proxy, cognitive_workload_proxy,
                        confidence, evidence_json, marker_mode
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.ts,
                        record.session_id,
                        record.subject_id,
                        record.ear,
                        record.perclos,
                        record.blink_rate,
                        record.blink_duration_ms,
                        record.microsleep_count,
                        record.yawn_count,
                        record.au4,
                        record.au6,
                        record.au12,
                        record.emotion,
                        record.head_yaw,
                        record.head_pitch,
                        record.tracking_quality,
                        record.state,
                        record.valence,
                        record.arousal,
                        record.cognitive_workload,
                        record.psych_state,
                        record.valence_proxy,
                        record.arousal_proxy,
                        record.cognitive_workload_proxy,
                        record.confidence,
                        record.evidence_json,
                        record.marker_mode,
                    ),
                )

    def log_event(self, event: EventRecord) -> None:
        """Logs a discrete behavioral or state transition event."""
        with self._connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO events (ts, session_id, event_type, payload)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        event.ts,
                        event.session_id,
                        event.event_type,
                        json.dumps(event.payload),
                    ),
                )

    def get_session_telemetry(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all telemetry rows for a given session sorted by timestamp."""
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT * FROM face_telemetry WHERE session_id = ? ORDER BY ts ASC",
                (session_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves all discrete events for a given session."""
        with self._connection() as conn:
            cur = conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY ts ASC",
                (session_id,),
            )
            results = []
            for row in cur.fetchall():
                d = dict(row)
                try:
                    d["payload"] = json.loads(d["payload"])
                except Exception:
                    pass
                results.append(d)
            return results
