#!/usr/bin/env python3
"""Script to download official MediaPipe Face Landmarker model for AURA-Face."""

import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEST_DIR = Path(__file__).resolve().parent.parent / "models"
DEST_FILE = DEST_DIR / "face_landmarker.task"


def download_model(force: bool = False) -> Path:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    if DEST_FILE.exists() and not force:
        size = DEST_FILE.stat().st_size
        print(f"[OK] Model already exists at {DEST_FILE} ({size:,} bytes).")
        return DEST_FILE

    print(f"[*] Downloading MediaPipe face_landmarker.task from:")
    print(f"    {MODEL_URL}")
    print(f"    Target: {DEST_FILE} ...")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r    Progress: {percent}% ({count * block_size:,}/{total_size:,} bytes)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, DEST_FILE, reporthook=reporthook)
        sys.stdout.write("\n")
        size = DEST_FILE.stat().st_size
        print(f"[SUCCESS] Download completed ({size:,} bytes).")
        return DEST_FILE
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}")
        if DEST_FILE.exists():
            DEST_FILE.unlink()
        raise


if __name__ == "__main__":
    download_model()
