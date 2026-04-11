"""Face detection and embedding pipeline.

Detection:  SSD Res10 (OpenCV DNN), unchanged.
Recognition: InsightFace buffalo_l ArcFace model (w600k_r50.onnx, 512-dim).
             Downloaded automatically on first use from public mirrors.

ArcFace significantly outperforms the previous SFace int8bq/fp32 model for
inter-person similarity, because its angular-margin training loss explicitly
maximises separation between different identities while keeping same-person
embeddings tightly clustered.

Public API:  pipeline.get_embedding(bgr_image) -> np.ndarray | None
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
import urllib.request
from urllib.error import URLError

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Detection model (SSD Res10) ───────────────────────────────────────────
DETECTION_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
    "face_detector/deploy.prototxt"
)
DETECTION_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)

# ── Recognition model (ArcFace w600k_r50, 512-dim) ───────────────────────
# Multiple public mirrors — tried in order until one succeeds.
ARCFACE_URLS: List[str] = [
    # Aitrepreneur community mirror (model repo, no auth)
    "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
    # facefusion model pack (publicly hosted, no auth)
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx",
    # InsightFace GitHub release zip is an alternative if both HF mirrors fail
    # (buffalo_l.zip must be extracted manually; we don't handle it here)
]


def _ensure_file(path: Path, url: str) -> Path:
    """Download *url* to *path* if it does not already exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp, open(path, "wb") as fh:
            fh.write(resp.read())
    return path


def _ensure_file_any(path: Path, urls: List[str]) -> Path:
    """Try each URL in *urls* until one succeeds, then cache at *path*."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception = RuntimeError("No URLs provided")
    for url in urls:
        try:
            log.info("Downloading ArcFace model from %s …", url)
            with urllib.request.urlopen(url) as resp, open(path, "wb") as fh:
                fh.write(resp.read())
            return path
        except Exception as exc:
            log.warning("  failed (%s), trying next mirror.", exc)
            last_exc = exc
            if path.exists():          # partial write — remove it
                path.unlink(missing_ok=True)
    raise last_exc


class FacePipeline:
    """Detect the most prominent face in an image and return its ArcFace embedding."""

    embedding_size: int = 512

    def __init__(self) -> None:
        models_dir = Path(__file__).resolve().parents[1] / "models"
        self._fallback = False

        # ── SSD detector ─────────────────────────────────────────────────
        try:
            proto   = _ensure_file(models_dir / "deploy.prototxt",  DETECTION_PROTO_URL)
            weights = _ensure_file(models_dir / "res10.caffemodel",  DETECTION_MODEL_URL)
            self._detector = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
            self._use_dnn_det = True
        except (URLError, cv2.error) as exc:
            log.warning("SSD detector unavailable (%s); using Haar cascade fallback.", exc)
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(cascade)
            self._use_dnn_det = False

        # ── ArcFace recognizer ────────────────────────────────────────────
        try:
            import onnxruntime as ort

            rec_path = _ensure_file_any(
                models_dir / "arcface_w600k_r50.onnx",
                ARCFACE_URLS,
            )
            self._recognizer = ort.InferenceSession(
                str(rec_path), providers=["CPUExecutionProvider"]
            )
            self._rec_input = self._recognizer.get_inputs()[0].name
            log.info("ArcFace model loaded from %s", rec_path)

        except Exception as exc:  # pragma: no cover
            log.warning("ArcFace unavailable (%s); falling back to HOG embedding.", exc)
            self._fallback = True
            self._hog = cv2.HOGDescriptor(
                _winSize=(112, 112), _blockSize=(32, 32),
                _blockStride=(16, 16), _cellSize=(16, 16), _nbins=9,
            )

    # ------------------------------------------------------------------
    def _detect_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return a 112×112 face crop or None if no face is found."""
        ih, iw = image.shape[:2]

        if self._use_dnn_det:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            self._detector.setInput(blob)
            dets = self._detector.forward()
            if dets.shape[2] == 0:
                return None
            idx = int(np.argmax(dets[0, 0, :, 2]))
            if float(dets[0, 0, idx, 2]) <= 0.5:
                return None
            box = dets[0, 0, idx, 3:7] * np.array([iw, ih, iw, ih])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            if x2 <= x1 or y2 <= y1:
                return None
        else:
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._detector.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                return None
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            x1, y1, x2, y2 = x, y, x + w, y + h

        return cv2.resize(image[y1:y2, x1:x2], (112, 112))

    # ------------------------------------------------------------------
    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face and return an L2-normalised float32 embedding.

        Returns ``None`` when no face is detected — callers must check.
        """
        face = self._detect_crop(image)
        if face is None:
            return None

        if self._fallback:
            hog = self._hog.compute(face).flatten().astype(np.float32)
            emb = hog[:512] if hog.size >= 512 else np.pad(hog, (0, 512 - hog.size))
        else:
            # ArcFace expects RGB, normalised to [-1, 1], in NCHW layout
            rgb  = face[:, :, ::-1].astype(np.float32)   # BGR → RGB
            blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis]
            emb  = self._recognizer.run(None, {self._rec_input: blob})[0][0]

        emb  = emb.astype("float32")
        norm = np.linalg.norm(emb)
        return None if norm == 0 else emb / norm


@lru_cache()
def get_pipeline() -> FacePipeline:
    return FacePipeline()
