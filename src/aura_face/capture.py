"""Computer vision and facial capture pipeline for AURA-Face.

Integrates OpenCV video stream, MediaPipe Face Landmarker (478 3D landmarks + 52 FACS blendshapes),
and head pose estimation (solvePnP) for angle gating and tracking quality estimation.

Privacy by design: Frames exist strictly in transient volatile RAM for feature extraction.
Zero frames or facial images are ever written to disk or transmitted across network interfaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from aura_face.config import GatingConfig, VisionConfig


@dataclass
class HeadPose:
    """Estimated 3D head orientation in degrees."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    is_gated_out: bool = False
    axis_points: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None


@dataclass
class FrameAnalysisResult:
    """Extracted numeric parameters from a single frame."""
    timestamp: float
    face_detected: bool
    tracking_quality: float
    head_pose: HeadPose
    landmarks_3d: Optional[np.ndarray] = None        # Shape (N, 3) in normalized coordinates
    blendshapes: Dict[str, float] = field(default_factory=dict) # 52 canonical blendshapes


class FaceLandmarkerPipeline:
    """Pluggable Facial Landmarker wrapper (MediaPipe Edge / PyFeat CUDA) with Head Pose Gating."""

    # 3D Generic Head Model reference points for solvePnP
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),          # Nose tip (index 1)
        (0.0, -330.0, -65.0),      # Chin (index 152)
        (-225.0, 170.0, -135.0),   # Left eye outer corner (index 263)
        (225.0, 170.0, -135.0),    # Right eye outer corner (index 33)
        (-150.0, -150.0, -125.0),  # Left mouth corner (index 287)
        (150.0, -150.0, -125.0),   # Right mouth corner (index 57)
    ], dtype=np.float64)

    LANDMARK_INDICES_FOR_PNP = [1, 152, 263, 33, 287, 57]

    def __init__(
        self,
        vision_config: Optional[VisionConfig] = None,
        gating_config: Optional[GatingConfig] = None,
    ) -> None:
        self.vision_config = vision_config or VisionConfig()
        self.gating_config = gating_config or GatingConfig()
        
        backend_type = (self.vision_config.backend or "mediapipe").lower()
        if backend_type == "pyfeat":
            from aura_face.backends import PyFeatBackend
            self.backend = PyFeatBackend(self.vision_config)
        else:
            from aura_face.backends import MediaPipeBackend
            self.backend = MediaPipeBackend(self.vision_config)

    def estimate_head_pose(
        self,
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> HeadPose:
        """Estimates head yaw, pitch, and roll using Perspective-n-Point, plus 3D axes."""
        if len(landmarks) < max(self.LANDMARK_INDICES_FOR_PNP) + 1:
            return HeadPose(0.0, 0.0, 0.0, is_gated_out=False)

        image_points = []
        for idx in self.LANDMARK_INDICES_FOR_PNP:
            x = landmarks[idx, 0] * frame_width
            y = landmarks[idx, 1] * frame_height
            image_points.append((x, y))

        image_points = np.array(image_points, dtype=np.float64)

        focal_length = frame_width
        center = (frame_width / 2.0, frame_height / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPose(0.0, 0.0, 0.0, is_gated_out=False)

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose projection matrix into Euler angles
        proj_matrix = np.hstack((rmat, np.zeros((3, 1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0, 0])
        yaw = float(euler_angles[1, 0])
        roll = float(euler_angles[2, 0])

        is_gated = (
            abs(yaw) > self.gating_config.max_yaw_deg
            or abs(pitch) > self.gating_config.max_pitch_deg
        )

        # 3D Head Pose Axes (X=Red/Right, Y=Green/Down, Z=Blue/Front)
        axis_3d = np.array([
            (0.0, 0.0, 0.0),       # Nose tip anchor
            (60.0, 0.0, 0.0),      # X axis
            (0.0, -60.0, 0.0),     # Y axis
            (0.0, 0.0, -60.0),     # Z axis
        ], dtype=np.float64)
        
        img_pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        axis_pts = (
            (int(img_pts[0, 0, 0]), int(img_pts[0, 0, 1])),
            (int(img_pts[1, 0, 0]), int(img_pts[1, 0, 1])),
            (int(img_pts[2, 0, 0]), int(img_pts[2, 0, 1])),
            (int(img_pts[3, 0, 0]), int(img_pts[3, 0, 1])),
        )

        return HeadPose(yaw=yaw, pitch=pitch, roll=roll, is_gated_out=is_gated, axis_points=axis_pts)

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        timestamp_s: float,
    ) -> FrameAnalysisResult:
        """Processes a single BGR OpenCV frame via active backend and returns extracted metrics."""
        h, w = bgr_frame.shape[:2]
        face_detected, tracking_quality, landmarks_3d, blendshapes = self.backend.process_frame(
            bgr_frame, timestamp_s
        )

        if not face_detected or landmarks_3d is None:
            return FrameAnalysisResult(
                timestamp=timestamp_s,
                face_detected=False,
                tracking_quality=0.0,
                head_pose=HeadPose(is_gated_out=True),
            )

        # Head pose & gating
        head_pose = self.estimate_head_pose(landmarks_3d, w, h)
        if head_pose.is_gated_out:
            tracking_quality = min(tracking_quality, 0.50)

        return FrameAnalysisResult(
            timestamp=timestamp_s,
            face_detected=True,
            tracking_quality=tracking_quality,
            head_pose=head_pose,
            landmarks_3d=landmarks_3d,
            blendshapes=blendshapes,
        )

    def close(self) -> None:
        """Cleans up backend resources."""
        if self.backend is not None:
            self.backend.close()
