# AURA-Face Models Directory

This directory stores the official Google MediaPipe `face_landmarker.task` model bundle.

### Model Details
- **File**: `face_landmarker.task`
- **Output**: 478 3D landmarks + 52 FACS blendshapes + head transformation matrix.
- **Provider**: Google MediaPipe Solutions

### Automated Download
Run the project download script:
```bash
python scripts/download_models.py
```

### Manual Download
If offline or behind a proxy, download directly from Google Cloud Storage:
```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```
and save it as `models/face_landmarker.task`.
