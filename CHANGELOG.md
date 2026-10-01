# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-10-01

### Added
- **Interactive System Architecture & FSM Diagrams**: Native Mermaid flowcharts embedded in `README.md` visualizing the edge video pipeline, pluggable vision backends, dual-domain processing, zero-frame retention boundaries, and state hysteresis transitions.
- **Executive Value Proposition & Impact Dossier**: Comprehensive pitch section highlighting mission-critical benefits, 5 pillars of the AURA edge, quantifiable ROI (95.5% recall, r=0.9078 PVT-B correlation), $4.8B dual-use terrestrial markets, and TRL 4 to 6 maturation roadmap.
- **Synthetic Cockpit Video & Animated HUD Demo**:
  - `docs/assets/aura_cockpit_demo.mp4` — High-definition 8-second synthetic mission demonstration.
  - `docs/assets/aura_cockpit_demo.gif` — Lightweight animated preview embedded in `README.md`.
  - `docs/assets/cockpit_overview.png` & `docs/assets/cardiovascular_twin_card5.png` — Full-resolution HUD snapshots.
- `scripts/generate_demo_assets.py` — Automated script to generate synthetic demo video and GIF assets headlessly.

---

## [1.0.0] — 2026-09-28

### Added
- **Cardiovascular Digital Twin (Card 5)**: Personalized Lower Body Negative Pressure (LBNP) countermeasure visualization running as a non-causal parallel simulation.
  - Real-time animated blood volume graph with LBNP phase indicators.
  - Suction arrow animation with smooth phase-based rendering.
  - Astronaut C trajectory with 940 mL target baseline at 30× playback.
- `CONTRIBUTING.md` — Open-source contribution guide.
- `CHANGELOG.md` — This file.
- `docs/REFERENCES.md` — Full scientific citations with links (replaces copyrighted PDFs).
- `run.bat` and `run.ps1` — Convenience launch scripts.
- Cardiovascular demo datasets: `data/cardiovascular_demo.csv`, `data/cardiovascular_demo_C.csv`.

### Changed
- Repository sanitized for public release: removed copyrighted PDFs, internal documents, binary model from git tracking, session recordings, and development screenshots.
- Updated `.gitignore` with comprehensive exclusions for models, videos, and runtime data.
- Updated `pyproject.toml` with public-facing contact information.

### Removed
- `document/` folder (copyrighted PDFs replaced by `docs/REFERENCES.md` citations).
- `video_gitignore/` folder (119 MB of demo recordings).
- Session timeline PNGs and telemetry databases from `data/`.
- Development preview screenshots from project root.

---

## [0.8.0] — 2026-09-24 (Hackathon Final)

### Added
- Observable behavioral proxies with explainable evidence JSON.
- Marker mode cycling (`OFF` → `MINIMAL` → `DETAILED` → `ALL`).
- Pluggable backend architecture (MediaPipe Edge CPU + Py-Feat CUDA).
- Full cockpit HUD with 5-card layout.

## [0.7.0] — 2026-09-24

### Added
- Dimensional affect model (valence/arousal continuous mapping).
- Split-screen cockpit HUD design.
- 5-track session timeline generator.
- SQLite database interpretation guide.

## [0.6.0] — 2026-09-24

### Added
- Privacy compliance test suite (`test_privacy.py`).
- PVT-B validation script with synthetic N=12 cohort.
- BEX2026 Challenge #3 dossier.

## [0.5.0] — 2026-09-24

### Added
- Artemis lunar cockpit HUD overlay.
- Synthetic astronaut simulator (fatigue, nominal, stress profiles).
- Unified CLI with hotkey controls.

## [0.4.0] — 2026-09-24

### Added
- SQLite WAL 1 Hz telemetry writer.
- Discrete event logger.
- Post-session timeline generator (`session_timeline.py`).

## [0.3.0] — 2026-09-23

### Added
- Guided astronaut baseline calibration (<25s).
- AU baseline normalizer.
- EMFACS-lite behavioral proxy engine.
- FSM with hysteresis (ALERT → MODERATE → CRITICAL → PRE_REST).

## [0.2.0] — 2026-09-23

### Added
- Bilateral EAR computation with EMA smoothing.
- Blink, droop, and microsleep detection.
- PERCLOS engine (1-minute sliding window).

## [0.1.0] — 2026-09-23

### Added
- Initial repository structure.
- Configuration system (`config/default.yaml`).
- Package setup (`pyproject.toml`).
- Foundational architecture design.
