"""Unified Command-Line Interface for AURA-Face.

Supports:
- Live webcam execution with automatic guided calibration.
- Synthetic Astronaut Simulator (--sim --profile <profile>).
- Pre-recorded video playback (--video <path>).
- SQLite 1 Hz telemetry and event logging.
- Session timeline generation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from aura_face.affective import AffectiveEngine
from aura_face.calibration import CalibrationEngine, CalibrationPhase
from aura_face.capture import FaceLandmarkerPipeline
from aura_face.cardiovascular import CardiovascularTwinCard
from aura_face.config import AuraConfig
from aura_face.ocular import OcularEngine
from aura_face.overlay import CockpitOverlay
from aura_face.simulator import SyntheticAstronautSimulator
from aura_face.states import AstronautState, AstronautStateMachine
from aura_face.storage import EventRecord, TelemetryRecord, TelemetryStorage


def run_pipeline(
    camera_id: int = 0,
    video_path: Optional[str] = None,
    sim_mode: bool = False,
    sim_profile: str = "nominal",
    subject_id: str = "artemis_cdr_01",
    calibrate: bool = True,
    duration_s: Optional[float] = None,
    no_gui: bool = False,
    config_path: Optional[str] = None,
    db_path: str = "data/aura_face_telemetry.db",
    backend: str = "mediapipe",
    device: str = "cpu",
    enable_cardio: bool = True,
    cv_csv: Optional[str] = None,
    cv_speed: Optional[float] = None,
    cv_earth: Optional[float] = None,
) -> None:
    """Executes the complete AURA-Face real-time processing loop."""
    config = AuraConfig.load(config_path)
    if backend:
        config.vision.backend = backend
    if device:
        config.vision.device = device

    session_id = f"SESSION_{int(time.time())}"
    print("=" * 75)
    print("  AURA-FACE // ASTRONAUT MULTIMODAL DIGITAL TWIN SENSOR")
    print("  ASI Space Hackathon (BEX2026 Challenge #3)")
    print("=" * 75)
    print("  [LEGAL / ETHICAL DISCLAIMER - BARRETT ET AL. 2019 COMPLIANCE]")
    print("  AURA-Face measures observable facial and ocular behavior and derives")
    print("  explainable operational proxies for vigilance, fatigue, cognitive strain")
    print("  and positive engagement. It does not diagnose psychological conditions")
    print("  or infer internal emotions with certainty.")
    print("-" * 75)
    print(f"[*] Session ID: {session_id}")
    print(f"[*] Subject:    {subject_id}")
    print(f"[*] Mode:       {'SIMULATOR (' + sim_profile + ')' if sim_mode else ('VIDEO: ' + video_path if video_path else f'WEBCAM {camera_id}')}")
    print(f"[*] Backend:    {config.vision.backend.upper()} ({config.vision.device.upper()})")
    print(f"[*] Telemetry:  {db_path} (SQLite WAL, 1 Hz)")
    print(f"[*] Privacy:    ZERO-FRAME RETENTION // 100% LOCAL PROCESSING")
    if enable_cardio and config.cardiovascular.enabled:
        print(f"[*] Cardio Twin: CARD 5 ACTIVE (LBNP Digital Twin running in parallel)")
    print("-" * 75)
    print("  [CONTROLS] M: Cycle Markers (OFF/MINIMAL/DETAILED/ALL)")
    print("             O: Eyes Contour | B: Face Brackets | H: Head Pose | D: Debug")
    print("             [/]: Adjust Cardio Speed | Q / ESC: Graceful Mission Shutdown")
    print("=" * 75)

    # 1. Initialize Storage
    storage = TelemetryStorage(db_path=db_path, enable_wal=config.session.enable_wal)

    # 2. Check for existing calibration
    existing_baseline = storage.load_latest_calibration(subject_id)
    calib_engine = CalibrationEngine(config=config.calibration, subject_id=subject_id)

    if existing_baseline and not calibrate:
        print(f"[+] Loaded existing baseline: EAR thresh={existing_baseline.ear_threshold:.3f}")
        calib_engine.calibrated_baseline = existing_baseline
        calib_engine.phase = CalibrationPhase.COMPLETED
    elif calibrate:
        print("[*] Initiating guided individual calibration (<25s)...")
        calib_engine.start(0.0)
    else:
        calib_engine.phase = CalibrationPhase.COMPLETED

    # 3. Engines
    ocular_engine = OcularEngine(
        config=config.ocular,
        ear_threshold=calib_engine.calibrated_baseline.ear_threshold,
    )
    affective_engine = AffectiveEngine(
        config=config.affective,
        baseline=calib_engine.calibrated_baseline,
    )
    state_machine = AstronautStateMachine(
        states_config=config.states,
        ocular_config=config.ocular,
        gating_config=config.gating,
    )

    cardio_card: Optional[CardiovascularTwinCard] = None
    if enable_cardio and config.cardiovascular.enabled:
        csv_file = cv_csv or config.cardiovascular.csv_path
        speed = cv_speed if cv_speed is not None else config.cardiovascular.playback_speed
        earth_ref = cv_earth if cv_earth is not None else config.cardiovascular.vvl_earth
        cardio_card = CardiovascularTwinCard(
            csv_path=csv_file,
            vvl_earth=earth_ref,
            playback_speed=speed,
            loop=config.cardiovascular.loop,
            tolerance_pct=config.cardiovascular.tolerance_pct,
        )

    overlay = CockpitOverlay(cardio_card=cardio_card)

    # 4. Input Source Setup
    cap: Optional[cv2.VideoCapture] = None
    simulator: Optional[SyntheticAstronautSimulator] = None
    landmarker_pipeline: Optional[FaceLandmarkerPipeline] = None

    if sim_mode:
        simulator = SyntheticAstronautSimulator(profile=sim_profile, fps=30)
    else:
        source = video_path if video_path else camera_id
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source {source}. Falling back to Simulator!")
            sim_mode = True
            simulator = SyntheticAstronautSimulator(profile=sim_profile, fps=30)
        else:
            try:
                landmarker_pipeline = FaceLandmarkerPipeline(
                    vision_config=config.vision,
                    gating_config=config.gating,
                )
            except Exception as e:
                print(f"[WARN] FaceLandmarkerPipeline initialization failed: {e}. Falling back to Simulator!")
                sim_mode = True
                simulator = SyntheticAstronautSimulator(profile=sim_profile, fps=30)

    start_real_time = time.time()
    last_telemetry_log_t = 0.0
    frame_count = 0
    t_mission = 0.0
    prev_face_detected = True

    # Initial state event
    storage.log_event(
        EventRecord(
            ts=t_mission,
            session_id=session_id,
            event_type="STATE_CHANGE",
            payload={"from": "UNKNOWN", "to": "ALERT", "reason": "Mission session initialized"},
        )
    )

    try:
        while True:
            loop_start = time.time()
            t_mission = loop_start - start_real_time

            if duration_s and t_mission >= duration_s:
                print(f"\n[*] Target duration reached ({duration_s}s). Concluding session.")
                break

            # A. Fetch Frame and Analysis
            if sim_mode and simulator is not None:
                frame, analysis = simulator.get_frame(t_mission)
                ear_mean, _ = simulator._compute_profile_signals(t_mission)
                ear_left, ear_right = ear_mean, ear_mean
            else:
                ret, frame = cap.read() if cap else (False, None)
                if not ret or frame is None:
                    print("\n[*] Video stream ended.")
                    break

                analysis = landmarker_pipeline.process_frame(frame, t_mission) if landmarker_pipeline else None
                if analysis and analysis.face_detected and analysis.landmarks_3d is not None:
                    ear_mean, ear_left, ear_right = ocular_engine.extract_ear_from_landmarks(analysis.landmarks_3d)
                else:
                    ear_mean, ear_left, ear_right = 0.0, 0.0, 0.0

            # Tracking transition detection
            face_detected = analysis.face_detected if analysis else False
            if prev_face_detected and not face_detected:
                overlay.add_event("TRACKING LOST", t_mission)
                storage.log_event(
                    EventRecord(
                        ts=t_mission,
                        session_id=session_id,
                        event_type="TRACKING_LOST",
                        payload={"reason": "Face landmarks lost or degraded below threshold"},
                    )
                )
            elif not prev_face_detected and face_detected:
                overlay.add_event("TRACKING RESTORED", t_mission)
                storage.log_event(
                    EventRecord(
                        ts=t_mission,
                        session_id=session_id,
                        event_type="TRACKING_RESTORED",
                        payload={"reason": "Face landmarks re-acquired"},
                    )
                )
            prev_face_detected = face_detected

            # B. Calibration Phase (if active)
            calib_phase, calib_progress = calib_engine.get_progress(t_mission)
            if calib_engine.is_active:
                raw_aus = affective_engine.extract_raw_aus(analysis.blendshapes if analysis else {})
                finished = calib_engine.feed_sample(ear_mean, raw_aus, t_mission)
                if finished:
                    print(f"\n[+] Calibration completed successfully!")
                    print(f"    Resting EAR Open:      {calib_engine.calibrated_baseline.ear_open:.3f}")
                    print(f"    Closed EAR Level:      {calib_engine.calibrated_baseline.ear_closed:.3f}")
                    print(f"    Individual EAR Thresh: {calib_engine.calibrated_baseline.ear_threshold:.3f}")
                    ocular_engine.set_ear_threshold(calib_engine.calibrated_baseline.ear_threshold)
                    affective_engine.update_baseline(calib_engine.calibrated_baseline)
                    storage.save_calibration(calib_engine.calibrated_baseline)
                    overlay.add_event(f"CALIBRATED: EAR_TH={calib_engine.calibrated_baseline.ear_threshold:.2f}", t_mission)

            # C. Ocular & Affective Telemetry
            ocular_tel = ocular_engine.process(
                ear_mean,
                t_mission,
                ear_left=ear_left,
                ear_right=ear_right,
            )
            tracking_q = analysis.tracking_quality if analysis else 1.0
            affective_tel = affective_engine.process(
                analysis.blendshapes if analysis else {},
                t_mission,
                ear_smooth=ocular_tel.ear_smooth,
                blink_rate=ocular_tel.blink_rate_bpm,
                perclos=ocular_tel.perclos,
                tracking_quality=tracking_q,
            )

            # Check discrete affective events
            if affective_tel.event_triggered:
                overlay.add_event(affective_tel.event_triggered, t_mission)
                storage.log_event(
                    EventRecord(
                        ts=t_mission,
                        session_id=session_id,
                        event_type=affective_tel.event_triggered,
                        payload={"duration_s": affective_tel.cognitive_load_sustained_s},
                    )
                )

            # Check discrete ocular events
            if ocular_tel.last_event and ocular_tel.last_event.event_type.value in ("DROOP", "MICROSLEEP"):
                if ocular_tel.last_event.end_ts == t_mission:
                    ev_type = ocular_tel.last_event.event_type.value
                    overlay.add_event(f"{ev_type} ({ocular_tel.last_event.duration_ms:.0f}ms)", t_mission)
                    if ev_type == "MICROSLEEP":
                        storage.log_event(
                            EventRecord(
                                ts=t_mission,
                                session_id=session_id,
                                event_type="MICRO_SLEEP",
                                payload={"duration_ms": ocular_tel.last_event.duration_ms},
                            )
                        )

            # D. Finite State Machine Update
            head_gated = analysis.head_pose.is_gated_out if analysis else False
            state, transition = state_machine.update(
                perclos=ocular_tel.perclos,
                microsleep_count_recent=ocular_tel.microsleep_count_recent,
                tracking_quality=tracking_q,
                head_gated_out=head_gated,
                pre_rest_pattern=affective_tel.pre_rest_pattern_active,
                timestamp=t_mission,
            )

            if transition:
                msg = f"STATE -> {transition.to_state.value}"
                overlay.add_event(msg, t_mission)
                print(f"[TRANSITION @ {t_mission:5.1f}s] {transition.from_state.value} -> {transition.to_state.value}: {transition.trigger_reason}")
                storage.log_event(
                    EventRecord(
                        ts=t_mission,
                        session_id=session_id,
                        event_type="STATE_CHANGE",
                        payload={"from": transition.from_state.value, "to": transition.to_state.value, "reason": transition.trigger_reason},
                    )
                )

            # E. 1 Hz Telemetry Logging
            if (t_mission - last_telemetry_log_t) >= (1.0 / config.session.telemetry_rate_hz):
                last_telemetry_log_t = t_mission
                record = TelemetryRecord(
                    ts=t_mission,
                    session_id=session_id,
                    subject_id=subject_id,
                    ear=ocular_tel.ear_smooth,
                    perclos=ocular_tel.perclos,
                    blink_rate=ocular_tel.blink_rate_bpm,
                    blink_duration_ms=ocular_tel.blink_duration_median_ms,
                    microsleep_count=ocular_tel.microsleep_count_recent,
                    yawn_count=affective_tel.yawn_count_total,
                    au4=affective_tel.norm_aus.au4,
                    au6=affective_tel.norm_aus.au6,
                    au12=affective_tel.norm_aus.au12,
                    emotion=affective_tel.emotion.value,
                    head_yaw=analysis.head_pose.yaw if analysis else 0.0,
                    head_pitch=analysis.head_pose.pitch if analysis else 0.0,
                    tracking_quality=tracking_q,
                    state=state.value,
                    valence=affective_tel.valence,
                    arousal=affective_tel.arousal,
                    cognitive_workload=affective_tel.cognitive_workload,
                    psych_state=affective_tel.psychological_state.value,
                    valence_proxy=affective_tel.facial_valence_proxy,
                    arousal_proxy=affective_tel.operational_arousal_proxy,
                    cognitive_workload_proxy=affective_tel.cognitive_workload_proxy,
                    confidence=affective_tel.confidence,
                    evidence_json=json.dumps(affective_tel.evidence),
                    marker_mode=overlay.marker_mode.value,
                )
                storage.log_telemetry(record)

            # Non-blocking cardiovascular simulation update
            if cardio_card is not None:
                cardio_card.update(t_mission)

            # F. Render HUD Overlay
            if not no_gui:
                window_name = "AURA-Face // Artemis Lunar Cockpit HUD"
                if frame_count == 0:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

                hud_frame = overlay.draw_hud(
                    frame=frame,
                    ocular=ocular_tel,
                    affective=affective_tel,
                    state=state,
                    session_id=session_id,
                    subject_id=subject_id,
                    ear_threshold=calib_engine.calibrated_baseline.ear_threshold,
                    calib_phase=calib_phase,
                    calib_progress=calib_progress,
                    landmarks_3d=analysis.landmarks_3d if analysis else None,
                    head_pose=analysis.head_pose if analysis else None,
                )
                cv2.imshow(window_name, hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    print("\n[*] User interrupted execution.")
                    break
                elif key in (ord("m"), ord("M")):
                    mode = overlay.cycle_marker_mode()
                    overlay.add_event(f"MARKERS: {mode.value.upper()}", t_mission)
                elif key in (ord("o"), ord("O")):
                    active = overlay.toggle_eye_contours()
                    overlay.add_event(f"EYE CONTOURS: {'ON' if active else 'OFF'}", t_mission)
                elif key in (ord("b"), ord("B")):
                    active = overlay.toggle_face_brackets()
                    overlay.add_event(f"FACE BRACKETS: {'ON' if active else 'OFF'}", t_mission)
                elif key in (ord("h"), ord("H")):
                    active = overlay.toggle_head_pose_axis()
                    overlay.add_event(f"HEAD POSE AXIS: {'ON' if active else 'OFF'}", t_mission)
                elif key in (ord("d"), ord("D")):
                    active = overlay.toggle_debug_labels()
                    overlay.add_event(f"DEBUG LABELS: {'ON' if active else 'OFF'}", t_mission)
                elif key == ord("[") and cardio_card:
                    cardio_card.playback_speed = max(1.0, cardio_card.playback_speed - 5.0)
                    overlay.add_event(f"CARDIO SPEED: {cardio_card.playback_speed:.0f}x", t_mission)
                elif key == ord("]") and cardio_card:
                    cardio_card.playback_speed += 5.0
                    overlay.add_event(f"CARDIO SPEED: {cardio_card.playback_speed:.0f}x", t_mission)

            frame_count += 1
            # Rate limiting for simulation
            if sim_mode:
                elapsed_frame = time.time() - loop_start
                sleep_t = (1.0 / 30.0) - elapsed_frame
                if sleep_t > 0:
                    time.sleep(sleep_t)

    finally:
        if cap:
            cap.release()
        if landmarker_pipeline:
            landmarker_pipeline.close()
        cv2.destroyAllWindows()

    # Post-Session Analytics & Mission Debrief
    telemetry_rows = storage.get_session_telemetry(session_id)
    events_rows = storage.get_session_events(session_id)

    print("\n" + "=" * 75)
    print("  MISSION TELEMETRY DEBRIEF // SESSION COMPLETED")
    print("=" * 75)
    print(f"[*] Session ID:          {session_id}")
    print(f"[*] Crew Member:         {subject_id.upper()}")
    print(f"[*] Total Duration:      {t_mission:.1f} s ({frame_count} frames processed)")
    print(f"[*] Telemetry Records:   {len(telemetry_rows)} rows logged at 1 Hz")

    if telemetry_rows:
        perclos_vals = [r["perclos"] for r in telemetry_rows if r["perclos"] is not None]
        cw_vals = [r["cognitive_workload"] for r in telemetry_rows if "cognitive_workload" in r and r["cognitive_workload"] is not None]
        val_proxy_vals = [r["valence_proxy"] for r in telemetry_rows if "valence_proxy" in r and r["valence_proxy"] is not None]
        arousal_proxy_vals = [r["arousal_proxy"] for r in telemetry_rows if "arousal_proxy" in r and r["arousal_proxy"] is not None]
        conf_vals = [r["confidence"] for r in telemetry_rows if "confidence" in r and r["confidence"] is not None]
        states_list = [r["state"] for r in telemetry_rows if r["state"] is not None]
        psych_list = [r["psych_state"] for r in telemetry_rows if "psych_state" in r and r["psych_state"] is not None]

        mean_perclos = float(np.mean(perclos_vals)) if perclos_vals else 0.0
        max_perclos = float(np.max(perclos_vals)) if perclos_vals else 0.0
        mean_cw = float(np.mean(cw_vals)) if cw_vals else 0.0
        mean_val_proxy = float(np.mean(val_proxy_vals)) if val_proxy_vals else 0.0
        mean_arousal_proxy = float(np.mean(arousal_proxy_vals)) if arousal_proxy_vals else 0.0
        mean_confidence = float(np.mean(conf_vals)) if conf_vals else 0.0

        # State percentages
        total_sec = max(1, len(states_list))
        pct_alert = (states_list.count("ALERT") / total_sec) * 100.0
        pct_moderate = (states_list.count("MODERATE") / total_sec) * 100.0
        pct_critical = (states_list.count("CRITICAL") / total_sec) * 100.0

        # Events count
        microsleep_events = [e for e in events_rows if e["event_type"] == "MICRO_SLEEP"]
        yawn_events = [e for e in events_rows if e["event_type"] == "YAWN"]
        cog_events = [e for e in events_rows if e["event_type"] in ("COGNITIVE_LOAD_ONSET", "COGNITIVE_STRAIN_ONSET")]
        tracking_lost_events = [e for e in events_rows if e["event_type"] == "TRACKING_LOST"]

        # Most frequent psych/pattern
        predominant_psych = max(set(psych_list), key=psych_list.count) if psych_list else "Calm Alert"

        print("-" * 75)
        print("  EXPLAINABLE OPERATIONAL BEHAVIORAL PROXIES (Barrett 2019):")
        print(f"  - Peak PERCLOS:                  {max_perclos:5.1f}% (Mean: {mean_perclos:5.1f}%)")
        print(f"  - Operational Alertness:         ALERT: {pct_alert:4.1f}% | MODERATE: {pct_moderate:4.1f}% | CRITICAL: {pct_critical:4.1f}%")
        print(f"  - Involuntary Microsleeps:       {len(microsleep_events)} episodes logged")
        print(f"  - Fatigue Yawns:                 {len(yawn_events)} detected")
        print(f"  - Mean Cognitive Workload Proxy: {mean_cw:4.2f} (0.0 to 1.0)")
        print(f"  - Cognitive Strain Spikes:       {len(cog_events)} sustained onsets (>=5s)")
        print(f"  - Mean Facial Valence Proxy:     {mean_val_proxy:+.2f} (-1.0 to +1.0)")
        print(f"  - Mean Operational Arousal Proxy:{mean_arousal_proxy:4.2f} (0.0 to 1.0)")
        print(f"  - Mean Inferred Confidence:      {mean_confidence:4.2f} (0.0 to 1.0)")
        print(f"  - Tracking Loss Events:          {len(tracking_lost_events)}")
        print(f"  - Predominant Behavioral Pattern:{predominant_psych}")
        print("-" * 75)

    print(f"[*] Local SQLite Database: {Path(db_path).resolve()}")

    # Automatically generate session timeline graph
    try:
        repo_root = str(Path(__file__).resolve().parent.parent.parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.session_timeline import generate_timeline
        timeline_img = Path("data") / f"{session_id}_timeline.png"
        generate_timeline(db_path=db_path, session_id=session_id, output_image=str(timeline_img))
        import shutil
        shutil.copy(timeline_img, "session_timeline.png")
        print(f"[+] Automated Session Timeline Chart: {Path('session_timeline.png').resolve()}")
    except Exception as e:
        print(f"[WARN] Automated timeline plotting skipped: {e}")

    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AURA-Face: Edge Facial Behavior Analysis for Artemis Crew Digital Twin."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Launch live monitoring cockpit or simulator")
    run_parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    run_parser.add_argument("--video", type=str, default=None, help="Pre-recorded video file path")
    run_parser.add_argument("--sim", action="store_true", help="Launch Synthetic Astronaut Simulator")
    run_parser.add_argument("--profile", choices=["nominal", "fatigue", "cognitive_load"], default="fatigue", help="Simulation profile (default: fatigue)")
    run_parser.add_argument("--backend", choices=["mediapipe", "pyfeat"], default="mediapipe", help="Vision backend (default: mediapipe)")
    run_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for vision backend (default: cpu)")
    run_parser.add_argument("--subject", type=str, default="artemis_cdr_01", help="Astronaut ID")
    run_parser.add_argument("--no-calibrate", action="store_true", help="Skip live calibration and use saved baseline")
    run_parser.add_argument("--duration", type=float, default=None, help="Session duration limit in seconds")
    run_parser.add_argument("--no-gui", action="store_true", help="Headless execution (no OpenCV window)")
    run_parser.add_argument("--config", type=str, default=None, help="Custom YAML config path")
    run_parser.add_argument("--db", type=str, default="data/aura_face_telemetry.db", help="SQLite database path")
    run_parser.add_argument("--no-cardio", action="store_true", help="Disable Card 5 Cardiovascular Digital Twin")
    run_parser.add_argument("--cv-csv", type=str, default=None, help="Path to cardiovascular CSV trajectory")
    run_parser.add_argument("--cv-speed", type=float, default=None, help="Cardiovascular playback speed (default: 30x)")
    run_parser.add_argument("--cv-earth", type=float, default=None, help="Earth reference Vvl in mL (default: 940)")

    # Command: timeline
    timeline_parser = subparsers.add_parser("timeline", help="Generate session timeline visualization")
    timeline_parser.add_argument("--db", type=str, default="data/aura_face_telemetry.db", help="Database path")
    timeline_parser.add_argument("--session", type=str, default=None, help="Session ID (default: latest)")
    timeline_parser.add_argument("--output", type=str, default="session_timeline.png", help="Output PNG path")

    # Command: validate
    subparsers.add_parser("validate", help="Run the BEX2026 PVT-B validation protocol benchmark")

    args = parser.parse_args()

    if args.command == "run" or args.command is None:
        if args.command is None:
            # Default to running simulator in fatigue mode for quick test
            run_pipeline(sim_mode=True, sim_profile="fatigue")
        else:
            run_pipeline(
                camera_id=args.camera,
                video_path=args.video,
                sim_mode=args.sim,
                sim_profile=args.profile,
                subject_id=args.subject,
                calibrate=not args.no_calibrate,
                duration_s=args.duration,
                no_gui=args.no_gui,
                config_path=args.config,
                db_path=args.db,
                backend=args.backend,
                device=args.device,
                enable_cardio=not args.no_cardio,
                cv_csv=args.cv_csv,
                cv_speed=args.cv_speed,
                cv_earth=args.cv_earth,
            )
    elif args.command == "timeline":
        from scripts.session_timeline import generate_timeline
        generate_timeline(args.db, args.session, args.output)
    elif args.command == "validate":
        import subprocess
        subprocess.run([sys.executable, "scripts/validate_protocol.py"])


if __name__ == "__main__":
    main()

