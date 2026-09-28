# AURA-Face: Edge Facial Behavior Analysis for Astronaut Multimodal Digital Twin

> **ASI Space Hackathon (BEX2026 Challenge #3 — Artemis Community)**  
> *Optical Computer Recognition of Stress, Affect and Fatigue during Performance in Spaceflight*

> [!NOTE]
> 🏆 **Winner — Best Solution for Value Proposition & Impact** at the ASI Space Hackathon BEX2026.

> [!IMPORTANT]
> **Ethical & Scientific Foundation (Barrett et al., 2019):**  
> *AURA-Face measures observable facial and ocular behavior and derives explainable operational proxies for vigilance, fatigue, cognitive strain and positive engagement. It does not diagnose psychological conditions or infer internal emotions with certainty.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Privacy: Zero-Frame](https://img.shields.io/badge/Privacy-Zero--Frame%20Retention-green.svg)](docs/PRIVACY.md)
[![Standard: NASA NSBRI](https://img.shields.io/badge/Science-NASA%20Dinges%20%26%20Metaxas-orange.svg)](docs/SCIENCE.md)

---

## Quickstart

```bash
# 1. Install dependencies
pip install -e .

# 2. Download MediaPipe Face Landmarker model (3.7 MB)
python scripts/download_models.py

# 3. Launch live cockpit with guided individual calibration
python -m aura_face.cli run --camera 0

# 4. Launch the simulator with cardiovascular digital twin
python -m aura_face.cli run --sim --profile fatigue

# 5. Optional: Research backend with NVIDIA CUDA
python -m aura_face.cli run --camera 0 --backend pyfeat --device cuda
```

### Cockpit Controls

| Hotkey | Action | Description |
|:---:|:---|:---|
| **`M`** | Cycle Marker Mode | `OFF` → `MINIMAL` → `DETAILED` → `ALL` |
| **`O`** | Toggle Eye Contours | Color-coded ocular contours (Cyan / Yellow / Red) |
| **`B`** | Toggle Face Brackets | Subtle corner brackets tracking the face region |
| **`H`** | Toggle Head Pose | 3D orientation axes (X=Red, Y=Green, Z=Blue) |
| **`D`** | Toggle Debug Labels | Raw Action Unit and EAR numeric labels |
| **`Q`** / **`ESC`** | Shutdown | Debrief, SQLite export, and timeline chart |

---

## Architecture

AURA-Face is a dual-domain system with two independent, non-causal pipelines running in parallel:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AURA-Face Cockpit HUD                       │
│                                                                     │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐ │
│  │  AFFECTIVE / OCULAR DOMAIN  │  │  CARDIOVASCULAR DOMAIN       │ │
│  │  (Cards 1–4)                │  │  (Card 5)                    │ │
│  │                              │  │                              │ │
│  │  Camera → MediaPipe →        │  │  CSV Trajectory →            │ │
│  │  EAR/PERCLOS/AU → FSM →     │  │  LBNP Simulation →           │ │
│  │  Behavioral Proxies →        │  │  Blood Volume Graph →        │ │
│  │  1 Hz SQLite Telemetry       │  │  Animated Suction Arrows     │ │
│  └──────────────────────────────┘  └──────────────────────────────┘ │
│                                                                     │
│  Design Invariant: NO causal link between the two domains.          │
│  The cardiovascular twin is a proof-of-concept visualization.       │
└─────────────────────────────────────────────────────────────────────┘
```

### Repository Structure

```
aura-face/
├── config/default.yaml          # All thresholds and parameters
├── models/                      # MediaPipe model (auto-downloaded)
├── data/
│   ├── cardiovascular_demo.csv  # Default cardiovascular trajectory
│   └── cardiovascular_demo_C.csv# Astronaut C personalized trajectory
├── src/aura_face/
│   ├── backends.py              # Pluggable backends (MediaPipe + Py-Feat)
│   ├── capture.py               # Video capture & head pose estimation
│   ├── ocular.py                # EAR, blink, PERCLOS engine
│   ├── affective.py             # AU normalization & behavioral proxies
│   ├── cardiovascular.py        # Cardiovascular digital twin (Card 5)
│   ├── calibration.py           # Individual baseline calibration (<25s)
│   ├── states.py                # FSM with hysteresis
│   ├── storage.py               # SQLite WAL 1 Hz telemetry writer
│   ├── overlay.py               # Cockpit HUD renderer
│   ├── simulator.py             # Synthetic astronaut generator
│   ├── config.py                # Configuration dataclasses
│   └── cli.py                   # CLI with hotkeys & backend selection
├── scripts/
│   ├── download_models.py       # Model auto-download
│   ├── session_timeline.py      # Post-session timeline generator
│   └── validate_protocol.py     # PVT-B validation & correlation
├── tests/                       # Pytest suite (privacy, cardiovascular, etc.)
├── docs/                        # Science, privacy, validation, references
├── CONTRIBUTING.md              # Contribution guide
└── CHANGELOG.md                 # Version history
```

---

## What AURA-Face Is & What It Is NOT

| What AURA-Face **IS** | What AURA-Face **IS NOT** |
| :--- | :--- |
| **Objective Facial Behavior Sensor**: Measures Action Units (FACS) and ocular dynamics validated by NASA (Dinges & Metaxas, NSBRI). | **NOT an Emotion Lie-Detector**: Does NOT claim universal emotion mapping (Barrett et al., 2019). Claims are strictly behavioral. |
| **Explainable Behavioral Proxies**: Continuous, normalized proxies with transparent evidence logs and confidence scores. | **NOT a Black-Box Classifier**: Every inferred pattern is justified by explicit temporal and morphological evidence. |
| **Temporal Persistence & Hysteresis**: Rejects transient artifacts (≥0.8s for smile, ≥1.2s for Duchenne, ≥5s for strain). | **NOT a Single-Frame Speculation Engine**: Single-frame twitches or speech movements do not flip states. |
| **Edge-Native & Zero-Frame**: Processes frames in volatile RAM; zero images saved to disk or network. | **NOT a Cloud or Surveillance Tool**: No remote frame streaming or facial recognition biometrics. |
| **Individually Calibrated**: Baseline EAR and resting AU levels learned in <25 seconds per astronaut. | **NOT a Static One-Size-Fits-All Threshold**: Accounts for individual morphology and resting facial asymmetry. |
| **1 Hz Telemetry Provider**: Output stream feeds the crew Multimodal Digital Twin. | **NOT an Actuator**: Supplies objective telemetry; does not make command decisions. |

---

## Privacy: Zero-Frame Retention

AURA-Face implements **Privacy by Design** at the architectural level:

- **No raw frames** are ever saved to disk, database, or network.
- All stored data is purely **numerical telemetry** at 1 Hz (floats, integers, timestamps).
- The SQLite database contains **zero images, zero video, zero biometric identifiers**.
- Full compliance with NASA/ESA crew privacy standards.

See [docs/PRIVACY.md](docs/PRIVACY.md) for the complete privacy architecture.

---

## Cardiovascular Digital Twin (Card 5)

Card 5 provides a **proof-of-concept visualization** of a personalized Lower Body Negative Pressure (LBNP) countermeasure simulation:

- Replays precomputed cardiovascular trajectories from a lumped-parameter 0D model.
- Displays real-time animated blood volume graph with LBNP phase indicators.
- Shows suction arrow animations synchronized to the LBNP protocol phases.
- Uses **Astronaut C** personalized trajectory with a 940 mL target baseline.

> This module runs in **parallel** with the facial analysis and is **not causally linked** to the behavioral outputs. It demonstrates the concept of a multimodal astronaut digital twin where multiple physiological domains are monitored simultaneously.

---

## Experimental Validation

```bash
python scripts/validate_protocol.py
```

### Results (N=12 Analog Cohort):
- **Pearson Correlation r(PERCLOS, PVT-B Lapses)**: **0.9078** (Target r > 0.85 ✅)
- **Pearson Correlation r(PERCLOS, Karolinska KSS)**: **0.9267**
- **Overall Diagnostic Accuracy**: **88.9%**
- **Fatigue Sensitivity (Recall)**: **95.5%**

See [docs/VALIDATION.md](docs/VALIDATION.md) for protocol details.

---

## Spaceflight Mission Context

During Artemis lunar surface operations, astronauts face total confinement in extreme habitats, lunar day/night cycles lasting 14 Earth days (disrupting circadian rhythms), and high cognitive workload during EVAs and system anomalies.

AURA-Face provides **non-invasive, contactless** psychological and vigilance monitoring directly at the habitat workstation.

### Earth-Space Two-Way Spin-Off
- **Space → Earth**: Telemetry algorithms transfer to hospital ICU night-shift staff, high-speed rail operators, air traffic control, and Antarctic research stations.
- **Earth → Space**: Integrates terrestrial PVT-B and Karolinska Sleepiness Scale calibration into deep space operations.

---

## Documentation

| Document | Description |
|:---|:---|
| [SCIENCE.md](docs/SCIENCE.md) | Scientific foundations and methodology |
| [PRIVACY.md](docs/PRIVACY.md) | Zero-frame retention architecture |
| [VALIDATION.md](docs/VALIDATION.md) | Experimental validation protocol |
| [DATABASE_GUIDE.md](docs/DATABASE_GUIDE.md) | SQLite telemetry schema guide |
| [REFERENCES.md](docs/REFERENCES.md) | Full scientific citations with links |
| [ARCHITECTURE_AND_PITCH_GUIDE.md](docs/ARCHITECTURE_AND_PITCH_GUIDE.md) | Technical architecture deep-dive |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Scientific References

1. **Dinges, D. F. & Metaxas, D. et al. (2008–2012)** — *Optical Computer Recognition of Stress, Affect and Fatigue during Performance in Spaceflight*, NSBRI/NASA Taskbook.
2. **Wierwille, W. W. et al. (1994)** — Canonical definition of PERCLOS: % time slow eyelid closures on 1-min window.
3. **Dinges, D. F. & Grace, R. (1998)** — *PERCLOS: A Valid Psychophysiological Measure of Alertness*.
4. **Ekman, P. & Friesen, W. V. (1978)** — *Facial Action Coding System (FACS)*.
5. **Barrett, L. F. et al. (2019)** — *Emotional Expressions Reconsidered*.
6. **Heldt, T. et al. (2004)** — *Computational Model of Cardiovascular Response to Orthostatic Stress*.

Full citations with links: [docs/REFERENCES.md](docs/REFERENCES.md)

---

## License

[MIT](LICENSE) — Copyright (c) 2026 AURA Team — ASI Space Hackathon (BEX2026)
