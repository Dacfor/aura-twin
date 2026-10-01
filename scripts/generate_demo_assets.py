"""Generates synthetic cockpit demo assets (MP4 video, animated GIF, and PNG preview).

Zero-Frame compliant: all visuals are generated from the synthetic wireframe astronaut
avatar and simulated telemetry. No real human images or webcam frames are recorded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
from PIL import Image

from aura_face.affective import AffectiveEngine
from aura_face.calibration import AstronautBaseline
from aura_face.cardiovascular import CardiovascularTwinCard
from aura_face.config import AuraConfig
from aura_face.ocular import OcularEngine
from aura_face.overlay import CockpitOverlay, MarkerMode
from aura_face.simulator import SyntheticAstronautSimulator
from aura_face.states import AstronautState, AstronautStateMachine


def generate_demo(
    output_dir: str = "docs/assets",
    duration_s: float = 8.0,
    fps: int = 24,
    video_scale: float = 0.67,  # Scale 1920x720 down to ~1280x480 for crisp compact video
    gif_scale: float = 0.42,    # Scale to ~800x300 for GitHub README animated preview
    gif_fps: int = 12,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing synthetic mission demonstration ({duration_s}s @ {fps} FPS)...")
    config = AuraConfig()

    # 1. Initialize engines
    ocular_engine = OcularEngine(config.ocular)
    affective_engine = AffectiveEngine(config.affective)
    state_machine = AstronautStateMachine(config.states)

    # Set mock calibrated baseline
    baseline = AstronautBaseline(
        subject_id="artemis_cdr_01",
        ear_open=0.31,
        ear_closed=0.12,
        ear_threshold=0.21,
        au4_baseline=0.05,
        au6_baseline=0.04,
        au12_baseline=0.03,
    )
    ocular_engine.set_ear_threshold(baseline.ear_threshold)
    affective_engine.update_baseline(baseline)

    # 2. Setup Cardiovascular Twin (Card 5)
    csv_path = "data/cardiovascular_demo_C.csv"
    if not Path(csv_path).exists():
        csv_path = "data/cardiovascular_demo.csv"

    cardio_card = CardiovascularTwinCard(
        csv_path=csv_path,
        vvl_earth=config.cardiovascular.vvl_earth,
        playback_speed=2.5,  # Speed up slightly for punchy 8-second demo
        loop=True,
    )

    # 3. Setup Simulator & Cockpit Overlay
    simulator = SyntheticAstronautSimulator(profile="fatigue", fps=fps)
    overlay = CockpitOverlay(
        default_marker_mode=MarkerMode.MINIMAL,
        cardio_card=cardio_card,
    )

    overlay.add_event("MISSION INITIALIZED", 0.0)
    overlay.add_event("ZERO-FRAME PRIVACY: ACTIVE", 0.5)
    overlay.add_event("CARDIO TWIN (CARD 5): SYNCED", 1.0)

    # Video Writer setup
    total_frames = int(duration_s * fps)
    mp4_path = out_path / "aura_cockpit_demo.mp4"
    
    # Calculate output dimensions
    raw_w, raw_h = overlay.total_width, overlay.height
    vid_w = int(raw_w * video_scale)
    vid_h = int(raw_h * video_scale)
    # Ensure even dimensions for video codecs
    vid_w = vid_w if vid_w % 2 == 0 else vid_w + 1
    vid_h = vid_h if vid_h % 2 == 0 else vid_h + 1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (vid_w, vid_h))

    gif_frames: list[Image.Image] = []
    gif_step = max(1, fps // gif_fps)

    gif_w = int(raw_w * gif_scale)
    gif_h = int(raw_h * gif_scale)
    gif_w = gif_w if gif_w % 2 == 0 else gif_w + 1
    gif_h = gif_h if gif_h % 2 == 0 else gif_h + 1

    screenshot_captured = False
    card5_screenshot_captured = False

    print(f"[*] Rendering {total_frames} frames to {mp4_path} ({vid_w}x{vid_h})...")

    for i in range(total_frames):
        t = i / float(fps)

        # Simulator frame & signals
        frame, analysis = simulator.get_frame(t)
        ear_mean, _ = simulator._compute_profile_signals(t)

        # Ocular & affective updates
        ocular_tel = ocular_engine.process(ear_mean, t, ear_left=ear_mean, ear_right=ear_mean)
        affective_tel = affective_engine.process(
            analysis.blendshapes if analysis else {},
            t,
            ear_smooth=ocular_tel.ear_smooth,
            blink_rate=ocular_tel.blink_rate_bpm,
            perclos=ocular_tel.perclos,
            tracking_quality=1.0,
        )

        # FSM state
        state, transition = state_machine.update(
            perclos=ocular_tel.perclos,
            microsleep_count_recent=ocular_tel.microsleep_count_recent,
            tracking_quality=1.0,
            head_gated_out=False,
            pre_rest_pattern=affective_tel.pre_rest_pattern_active,
            timestamp=t,
        )
        if transition:
            overlay.add_event(f"STATE -> {transition.to_state.value}", t)

        # Card 5 update
        cardio_card.update(t)

        # Render HUD
        hud_frame = overlay.draw_hud(
            frame=frame,
            ocular=ocular_tel,
            affective=affective_tel,
            state=state,
            session_id="MISSION_DEMO_01",
            subject_id="artemis_cdr_01",
            ear_threshold=baseline.ear_threshold,
            calib_phase=None,
            calib_progress=1.0,
            landmarks_3d=analysis.landmarks_3d if analysis else None,
            head_pose=analysis.head_pose if analysis else None,
        )

        # Capture key screenshots around t=3.5s
        if not screenshot_captured and t >= 3.5:
            cv2.imwrite(str(out_path / "cockpit_overview.png"), hud_frame)
            # Also capture Card 5 zoom
            card5_crop = hud_frame[:, -overlay.cardio_w :]
            cv2.imwrite(str(out_path / "cardiovascular_twin_card5.png"), card5_crop)
            screenshot_captured = True

        # Resize for MP4
        vid_frame = cv2.resize(hud_frame, (vid_w, vid_h), interpolation=cv2.INTER_AREA)
        writer.write(vid_frame)

        # Sample for GIF
        if i % gif_step == 0:
            gif_frame_bgr = cv2.resize(hud_frame, (gif_w, gif_h), interpolation=cv2.INTER_AREA)
            gif_frame_rgb = cv2.cvtColor(gif_frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(gif_frame_rgb).quantize(colors=128, method=Image.Quantize.MEDIANCUT)
            gif_frames.append(pil_img)

    writer.release()
    print(f"[+] MP4 Video written: {mp4_path} ({mp4_path.stat().st_size / 1024:.1f} KB)")

    # Save animated GIF
    if gif_frames:
        gif_path = out_path / "aura_cockpit_demo.gif"
        print(f"[*] Compiling animated GIF ({len(gif_frames)} frames @ {gif_fps} FPS)...")
        gif_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(1000 / gif_fps),
            loop=0,
            optimize=True,
        )
        print(f"[+] Animated GIF written: {gif_path} ({gif_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_demo()
