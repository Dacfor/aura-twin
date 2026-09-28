"""Pluggable facial analysis backends for AURA-Face.

Supports:
1. MediaPipeBackend: Lightweight, 30-60 FPS edge flight default (3.75 MB, CPU/Edge).
2. PyFeatBackend: Optional research-grade backend for GPU/CUDA workstations (RTX 4060).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from aura_face.config import GatingConfig, VisionConfig


class BaseFaceBackend(ABC):
    """Abstract interface for facial feature extraction backends."""

    name: str = "BaseFaceBackend"

    @abstractmethod
    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp_s: float,
    ) -> Tuple[bool, float, Optional[np.ndarray], Dict[str, float]]:
        """Processes a single BGR frame.

        Returns:
            Tuple of:
            - face_detected: bool
            - tracking_quality: float (0.0 to 1.0)
            - landmarks_3d: Optional[np.ndarray] shape (N, 3) in normalized [0, 1]
            - blendshapes: Dict[str, float] normalized [0, 1]
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Releases backend resources."""
        pass


class MediaPipeBackend(BaseFaceBackend):
    """Production Flight Edge backend powered by Google MediaPipe Face Landmarker."""

    name: str = "MediaPipe Edge FaceMesh"

    def __init__(self, vision_config: VisionConfig) -> None:
        self.config = vision_config
        self.landmarker: Any = None
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        import mediapipe as mp

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            alt_path = Path(__file__).resolve().parent.parent.parent / self.config.model_path
            if alt_path.exists():
                model_path = alt_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model file not found at {model_path}. "
                "Run 'python scripts/download_models.py' first."
            )

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp_s: float,
    ) -> Tuple[bool, float, Optional[np.ndarray], Dict[str, float]]:
        import mediapipe as mp

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(timestamp_s * 1000)

        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not detection_result.face_landmarks:
            return False, 0.0, None, {}

        # Extract 478 landmarks
        raw_landmarks = detection_result.face_landmarks[0]
        landmarks_3d = np.array(
            [[lm.x, lm.y, lm.z] for lm in raw_landmarks],
            dtype=np.float32,
        )

        # Extract 52 FACS blendshapes
        blendshapes: Dict[str, float] = {}
        if detection_result.face_blendshapes:
            for category in detection_result.face_blendshapes[0]:
                blendshapes[category.category_name] = float(category.score)

        tracking_quality = 0.95
        if detection_result.facial_transformation_matrixes:
            tracking_quality = 0.98

        return True, tracking_quality, landmarks_3d, blendshapes

    def close(self) -> None:
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None


class PyFeatBackend(BaseFaceBackend):
    """Optional Workstation GPU Backend (Py-Feat) for NVIDIA CUDA (RTX 4060)."""

    name: str = "Py-Feat CUDA Workstation"

    def __init__(self, vision_config: VisionConfig, raise_on_missing: bool = False) -> None:
        self.config = vision_config
        self.device = vision_config.device
        self.detector: Any = None
        self._available: bool = False
        self._init_pyfeat(raise_on_missing=raise_on_missing)

    @property
    def is_ready(self) -> bool:
        return self._available and self.detector is not None

    def _init_pyfeat(self, raise_on_missing: bool = False) -> None:
        try:
            from feat import Detector
            device = "cuda" if self.device == "cuda" else "cpu"
            self.detector = Detector(
                face_model="faceboxes",
                landmark_model="mobilenet",
                au_model="svm",
                emotion_model="resmasknet",
                device=device,
            )
            self._available = True
        except ImportError as e:
            self._available = False
            if raise_on_missing:
                raise ImportError(
                    "Py-Feat is not installed in the current environment.\n"
                    "To use the CUDA/Py-Feat backend, run:\n"
                    "    pip install py-feat\n"
                    "Or run with the default Flight Edge backend: --backend mediapipe"
                ) from e

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp_s: float,
    ) -> Tuple[bool, float, Optional[np.ndarray], Dict[str, float]]:
        if not self.is_ready:
            return False, 0.0, None, {}

        # Py-Feat expects RGB or image path
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        try:
            prediction = self.detector.detect_image(rgb_frame)
            if prediction.empty:
                return False, 0.0, None, {}

            # Convert 68 landmarks to normalized [0, 1]
            h, w = bgr_frame.shape[:2]
            lms_x = [prediction[f"x_{i}"].values[0] for i in range(68)]
            lms_y = [prediction[f"y_{i}"].values[0] for i in range(68)]
            landmarks_2d = np.zeros((68, 3), dtype=np.float32)
            for i in range(68):
                landmarks_2d[i, 0] = lms_x[i] / w
                landmarks_2d[i, 1] = lms_y[i] / h
                landmarks_2d[i, 2] = 0.0

            # Map Py-Feat Action Units to canonical blendshape names
            blendshapes: Dict[str, float] = {}
            au_mapping = {
                "AU01": "browInnerUp",
                "AU04": "browDownLeft",
                "AU06": "eyeSquintLeft",
                "AU09": "noseSneerLeft",
                "AU12": "mouthSmileLeft",
                "AU15": "mouthFrownLeft",
                "AU25": "jawOpen",
                "AU26": "jawOpen",
            }
            for au_col, bs_name in au_mapping.items():
                if au_col in prediction.columns:
                    val = float(prediction[au_col].values[0])
                    norm_val = float(np.clip(val, 0.0, 1.0))
                    blendshapes[bs_name] = norm_val
                    if bs_name.endswith("Left"):
                        right_name = bs_name.replace("Left", "Right")
                        blendshapes[right_name] = norm_val

            return True, 0.92, landmarks_2d, blendshapes
        except Exception:
            return False, 0.0, None, {}

    def close(self) -> None:
        self.detector = None


def get_backend(
    backend_name: str = "mediapipe",
    device: str = "cpu",
    vision_config: Optional[VisionConfig] = None,
) -> BaseFaceBackend:
    """Factory function for instantiating facial analysis backends."""
    cfg = vision_config or VisionConfig(backend=backend_name, device=device)
    cfg.backend = backend_name
    cfg.device = device
    if backend_name.lower() == "pyfeat":
        return PyFeatBackend(cfg, raise_on_missing=False)
    return MediaPipeBackend(cfg)

