"""Simple face processing helpers."""

from functools import lru_cache
from typing import Tuple
import hashlib
import numpy as np
import cv2


class FacePipeline:
    """Very small helper class used for demo purposes.

    The implementation intentionally keeps the dependencies light weight so that
    the application can be deployed easily in minimal environments. It relies on
    OpenCV's built in Haar cascade for face detection and generates a
    deterministic embedding using a SHA256 hash of the aligned face.  This is
    obviously **not** meant for production quality similarity search but allows
    the rest of the service to be exercised without heavy ML models."""

    def __init__(self):
        # Load OpenCV's face detector. This file is shipped with opencv-python.
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect_and_align(self, image: np.ndarray) -> np.ndarray:
        """Detect the largest face in the image and return a cropped square.

        If no face is detected the original image is resized and returned."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            # Fallback: resize whole image
            face = image
        else:
            x, y, w, h = faces[0]
            face = image[y : y + h, x : x + w]
        return cv2.resize(face, (112, 112))

    def embed(self, face: np.ndarray) -> np.ndarray:
        """Return a deterministic 512 dimensional vector for the face."""
        resized = cv2.resize(face, (64, 64))
        digest = hashlib.sha256(resized.tobytes()).digest()
        vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        # Repeat the digest so we end up with 512 dimensions
        vec = np.tile(vec, 512 // len(vec) + 1)[:512]
        return vec

@lru_cache()
def get_pipeline() -> FacePipeline:
    return FacePipeline()

