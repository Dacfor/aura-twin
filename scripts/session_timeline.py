#!/usr/bin/env python3
"""Session Timeline Generator for AURA-Face.

Generates a publication-ready, multi-track psychophysiological timeline
from SQLite mission telemetry data.

Tracks displayed:
1. PERCLOS % with NASA/Wierwille alert thresholds (8% Moderate, 15% Critical).
2. Bilateral EAR signal with discrete Microsleep markers.
3. Action Units (AU4 Mental Workload / AU12 Positive Valence).
4. Alertness State Ribbon (ALERT, MODERATE, CRITICAL, PRE_REST, UNKNOWN).
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def generate_timeline(
    db_path: str = "data/aura_face_telemetry.db",
    session_id: Optional[str] = None,
    output_image: str = "session_timeline.png",
) -> Path:
    p_db = Path(db_path)
    if not p_db.exists():
        raise FileNotFoundError(f"Database not found at {p_db}")

    conn = sqlite3.connect(str(p_db))
    conn.row_factory = sqlite3.Row

    # If session_id not specified, pick the most recent one
    if not session_id:
        cur = conn.execute("SELECT session_id FROM face_telemetry ORDER BY ts DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            print("[WARN] No telemetry data found in database.")
            return Path(output_image)
        session_id = row["session_id"]

    cur = conn.execute(
        "SELECT * FROM face_telemetry WHERE session_id = ? ORDER BY ts ASC",
        (session_id,),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"[WARN] No records found for session {session_id}")
        return Path(output_image)

    cur_ev = conn.execute(
        "SELECT * FROM events WHERE session_id = ? ORDER BY ts ASC",
        (session_id,),
    )
    events = cur_ev.fetchall()
    conn.close()

    # Extract time series (safely handling older schemas)
    keys = rows[0].keys()
    t0 = rows[0]["ts"]
    times_s = np.array([r["ts"] - t0 for r in rows])
    ears = np.array([r["ear"] for r in rows])
    perclos = np.array([r["perclos"] for r in rows])
    au4 = np.array([r["au4"] for r in rows])
    au12 = np.array([r["au12"] for r in rows])
    au6 = np.array([r["au6"] if "au6" in keys and r["au6"] is not None else 0.0 for r in rows])
    states = [r["state"] for r in rows]

    valence = np.array([r["valence"] if "valence" in keys and r["valence"] is not None else 0.0 for r in rows])
    arousal = np.array([r["arousal"] if "arousal" in keys and r["arousal"] is not None else 0.0 for r in rows])
    cog_workload = np.array([r["cognitive_workload"] if "cognitive_workload" in keys and r["cognitive_workload"] is not None else 0.0 for r in rows])
    psych_states = [r["psych_state"] if "psych_state" in keys and r["psych_state"] is not None else "Calm Alert" for r in rows]

    state_color_map = {
        "ALERT": "#2ECC71",     # Crisp Emerald Green
        "MODERATE": "#F39C12",  # Amber Alert
        "CRITICAL": "#E74C3C",  # Vibrant Warning Red
        "PRE_REST": "#9B59B6",  # Amethyst Purple
        "UNKNOWN": "#95A5A6",   # Slate Gray
    }

    # Setup Figure (Dark Artemis Cockpit Theme)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        5, 1, figsize=(13, 11), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.0, 2.2, 1.8, 1.2]}
    )
    fig.suptitle(f"AURA-Face Mission Psychophysiological Telemetry\nSession: {session_id}", fontsize=14, fontweight="bold", color="#ECF0F1")

    # Track 1: PERCLOS
    ax1 = axes[0]
    ax1.plot(times_s, perclos, color="#3498DB", linewidth=2.0, label="PERCLOS (%)")
    ax1.axhline(8.0, color="#F39C12", linestyle="--", alpha=0.85, label="Moderate Threshold (8%)")
    ax1.axhline(15.0, color="#E74C3C", linestyle="--", alpha=0.85, label="Critical Threshold (15%)")
    ax1.fill_between(times_s, perclos, color="#3498DB", alpha=0.2)
    ax1.set_ylabel("PERCLOS (%)", fontsize=10, fontweight="bold")
    ax1.set_ylim(-1, max(25, float(np.max(perclos)) + 5))
    ax1.legend(loc="upper right", framealpha=0.6, fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.3)

    # Track 2: EAR Signal & Microsleeps
    ax2 = axes[1]
    ax2.plot(times_s, ears, color="#1ABC9C", linewidth=1.5, label="Bilateral EAR (Smoothed)")
    ax2.set_ylabel("Eye Aspect Ratio", fontsize=10, fontweight="bold")
    ax2.set_ylim(0.0, 0.40)
    for ev in events:
        if ev["event_type"] == "MICRO_SLEEP":
            ev_t = ev["ts"] - t0
            ax2.axvline(ev_t, color="#E74C3C", linestyle="-", linewidth=2.0, alpha=0.9)
            ax2.text(ev_t, 0.33, " [MICROSLEEP]", color="#E74C3C", fontsize=8, rotation=90)
    ax2.legend(loc="upper right", framealpha=0.6, fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.3)

    # Track 3: Dimensional Psychodynamics & Workload (Russell 1980; Dinges 2005)
    ax3 = axes[2]
    ax3.plot(times_s, valence, color="#00E5FF", linewidth=1.8, label="Valence (Russell -1 to +1)")
    ax3.plot(times_s, cog_workload, color="#FFA726", linewidth=2.0, label="Cognitive Workload Index (0 to 1)")
    ax3.plot(times_s, arousal, color="#AB47BC", linewidth=1.4, linestyle="--", label="Arousal (0 to 1)")
    ax3.axhline(0.0, color="#78909C", linestyle=":", alpha=0.5)
    ax3.set_ylabel("Psychological Index", fontsize=10, fontweight="bold")
    ax3.set_ylim(-1.05, 1.05)
    ax3.legend(loc="upper right", framealpha=0.6, fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.3)

    # Track 4: FACS Action Units (Ekman 1978; Barrett 2019)
    ax4 = axes[3]
    ax4.plot(times_s, au4, color="#FF7043", linewidth=1.8, label="AU4 Brow Lowerer (Mental Strain)")
    ax4.plot(times_s, au12, color="#66BB6A", linewidth=1.5, linestyle="-.", label="AU12 Lip Smile (Positive Valence)")
    ax4.plot(times_s, au6, color="#4DD0E1", linewidth=1.2, linestyle=":", label="AU6 Cheek Raiser (Duchenne)")
    ax4.set_ylabel("Normalized AU (0-1)", fontsize=10, fontweight="bold")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="upper right", framealpha=0.6, fontsize=8)
    ax4.grid(True, linestyle=":", alpha=0.3)

    # Track 5: Alertness State Ribbon
    ax5 = axes[4]
    for i in range(len(times_s) - 1):
        st = states[i]
        color = state_color_map.get(st, "#95A5A6")
        ax5.axvspan(times_s[i], times_s[i+1], color=color, alpha=0.85)

    ax5.set_yticks([])
    ax5.set_ylabel("Alertness State", fontsize=10, fontweight="bold")
    ax5.set_xlabel("Mission Elapsed Time (seconds)", fontsize=11, fontweight="bold")

    # Add legend for states
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=state_color_map[k], label=k)
        for k in ["ALERT", "MODERATE", "CRITICAL", "PRE_REST", "UNKNOWN"]
    ]
    ax5.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.0, 1.55), ncol=5, framealpha=0.7, fontsize=8)

    plt.tight_layout()
    out_path = Path(output_image)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[OK] Mission timeline successfully generated: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AURA-Face session timeline plot.")
    parser.add_argument("--db", default="data/aura_face_telemetry.db", help="Path to telemetry SQLite DB")
    parser.add_argument("--session", default=None, help="Session ID (default: most recent)")
    parser.add_argument("--output", default="session_timeline.png", help="Output PNG file path")
    args = parser.parse_args()

    generate_timeline(args.db, args.session, args.output)
