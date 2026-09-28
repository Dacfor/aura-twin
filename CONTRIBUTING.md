# Contributing to AURA-Face

Thank you for your interest in contributing to AURA-Face! This project emerged from the ASI Space Hackathon (BEX2026) and is now maintained as an open-source tool for non-invasive facial behavior analysis.

## Getting Started

### Prerequisites

- Python 3.10+
- A webcam (for live mode) or use the built-in simulator

### Development Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/aura-face.git
cd aura-face

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Download the MediaPipe model
python scripts/download_models.py

# Run the test suite
pytest
```

## Development Workflow

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Run the full test suite** before submitting:
   ```bash
   pytest
   ```
4. **Follow existing code style** — the project uses `ruff` for linting:
   ```bash
   ruff check src/ tests/
   ```
5. **Submit a Pull Request** with a clear description of what changed and why.

## Project Structure

```
src/aura_face/
├── backends.py        # Pluggable ML backends (MediaPipe, Py-Feat)
├── capture.py         # Video capture and head pose estimation
├── ocular.py          # EAR, blink, PERCLOS engine
├── affective.py       # AU normalization and behavioral proxies
├── cardiovascular.py  # Cardiovascular digital twin visualization
├── calibration.py     # Individual baseline calibration
├── states.py          # FSM with hysteresis
├── storage.py         # SQLite telemetry writer
├── overlay.py         # Cockpit HUD renderer
├── simulator.py       # Synthetic astronaut generator
├── config.py          # Configuration dataclasses
└── cli.py             # Command-line interface
```

## Design Invariants

These are non-negotiable architectural constraints:

1. **Zero-Frame Retention**: No raw video frames are ever saved to disk or transmitted over a network. All stored data is purely numerical telemetry.
2. **Non-Causal Decoupling**: The cardiovascular digital twin is a parallel demonstration. It must NOT be causally linked to the facial/behavioral analysis outputs.
3. **Individual Calibration**: All threshold-based detections must use individually calibrated baselines, never hardcoded universal thresholds.

## Reporting Issues

When reporting bugs, please include:
- Your OS and Python version
- The command you ran
- The full error traceback
- Whether you're using a webcam or the simulator

## Code of Conduct

Be respectful, constructive, and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
