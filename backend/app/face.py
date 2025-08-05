"""Utilities for face detection and embedding generation.

This module implements a compact yet robust face recognition pipeline based on
pre‑trained models from the `opencv_zoo` project.  To keep the repository
lightweight the required model weights are downloaded on first use and cached
under ``backend/models``.

If the download step fails (e.g. because network access is unavailable) the
pipeline gracefully falls back to a simple HOG‑based embedding so the rest of
the application continues to function, albeit with reduced accuracy.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import urllib.request
from urllib.error import URLError

import numpy as np
import cv2


# URLs of lightweight detection and recognition models
DETECTION_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
    "face_detector/deploy.prototxt"
)
DETECTION_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)
RECOGNITION_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec_int8bq.onnx"
)


def _ensure_file(path: Path, url: str) -> Path:
    """Download ``url`` to ``path`` if the file does not already exist."""

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp, open(path, "wb") as fh:
            fh.write(resp.read())
    return path


class FacePipeline:
    """Face detection and embedding pipeline.

    When the deep learning models are available we use an SSD based face
    detector and the SFace recognizer (a MobileFaceNet variant).  Both models
    are small and fast while providing good accuracy.  If the models cannot be
    loaded we fall back to a much simpler HOG based approach.
    """

    def __init__(self) -> None:
        models_dir = Path(__file__).resolve().parents[1] / "models"

        try:
            # -- Deep learning detector -------------------------------------
            proto = _ensure_file(models_dir / "deploy.prototxt", DETECTION_PROTO_URL)
            weights = _ensure_file(models_dir / "res10.caffemodel", DETECTION_MODEL_URL)
            self.detector = cv2.dnn.readNetFromCaffe(str(proto), str(weights))

            # -- Deep learning recognizer -----------------------------------
            rec_path = _ensure_file(models_dir / "sface.onnx", RECOGNITION_MODEL_URL)
            self.recognizer = cv2.dnn.readNetFromONNX(str(rec_path))

            # Determine embedding dimensionality
            dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
            self.recognizer.setInput(dummy)
            self.embedding_size = int(self.recognizer.forward().size)
            self._fallback = False

        except (URLError, cv2.error):  # pragma: no cover - network dependency
            # Fallback to Haar + HOG if the models cannot be downloaded/loaded
            cascade_path = (
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.hog = cv2.HOGDescriptor(
                _winSize=(112, 112),
                _blockSize=(32, 32),
                _blockStride=(16, 16),
                _cellSize=(16, 16),
                _nbins=9,
            )
            self.embedding_size = 512
            self._fallback = True

    # ------------------------------------------------------------------
    def detect_and_align(self, image: np.ndarray) -> np.ndarray:
        """Detect the most prominent face and return a 112x112 crop."""

        if self._fallback:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5
            )
            if len(faces) == 0:
                face = image
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face = image[y : y + h, x : x + w]
            return cv2.resize(face, (112, 112))

        # DNN based detection
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.detector.setInput(blob)
        detections = self.detector.forward()
        if detections.shape[2] > 0:
            idx = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, idx, 2]
            if confidence > 0.5:
                box = detections[0, 0, idx, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face = image[y1:y2, x1:x2]
            else:
                face = image
        else:
            face = image
        return cv2.resize(face, (112, 112))

    # ------------------------------------------------------------------
    def embed(self, face: np.ndarray) -> np.ndarray:
        """Generate an embedding vector for ``face``."""

        if self._fallback:
            hog_vec = self.hog.compute(face).flatten().astype(np.float32)
            if hog_vec.size >= 512:
                return hog_vec[:512]
            return np.pad(hog_vec, (0, 512 - hog_vec.size), mode="constant")

        blob = cv2.dnn.blobFromImage(
            face, 1 / 255.0, (112, 112), mean=(0, 0, 0), swapRB=True, crop=False
        )
        self.recognizer.setInput(blob)
        return self.recognizer.forward().flatten().astype(np.float32)


@lru_cache()
def get_pipeline() -> FacePipeline:
    """Return a cached :class:`FacePipeline` instance."""

    return FacePipeline()

