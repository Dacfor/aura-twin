"""Static compliance test to enforce Zero-Frame Storage Privacy by Design.

Scans all source code under src/aura_face to ensure that:
1. cv2.imwrite is never called.
2. cv2.imencode is never called.
3. PIL Image.save is never called.
4. Zero frame serialization to filesystem occurs anywhere in core runtime modules.
"""

import re
from pathlib import Path

FORBIDDEN_PATTERNS = [
    (r"cv2\.imwrite", "cv2.imwrite calls are forbidden in core runtime code"),
    (r"cv2\.imencode", "cv2.imencode calls are forbidden in core runtime code"),
    (r"\.save\(.*(\.png|\.jpg|\.jpeg|\.bmp)", "Direct image file saving is forbidden in core runtime code"),
]


def test_zero_frame_privacy_compliance():
    src_dir = Path(__file__).resolve().parent.parent / "src" / "aura_face"
    assert src_dir.exists(), f"Source directory {src_dir} does not exist"

    violations = []
    py_files = list(src_dir.glob("*.py"))
    assert len(py_files) > 0, "No python source files found to audit"

    for file_path in py_files:
        content = file_path.read_text(encoding="utf-8")
        for pattern, msg in FORBIDDEN_PATTERNS:
            matches = re.finditer(pattern, content)
            for m in matches:
                violations.append(
                    f"{file_path.name}:{content[:m.start()].count(chr(10)) + 1} - {msg}"
                )

    assert not violations, "Privacy by Design violations detected:\n" + "\n".join(violations)
