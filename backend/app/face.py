"""Face detection, 5-point landmark alignment, and ArcFace embedding pipeline.

Detection & Landmarks:
    Primary:  OpenCV YuNet (cv2.FaceDetectorYN, face_detection_yunet_2023mar.onnx)
              Provides bounding box + 5 facial landmarks (eyes, nose, mouth corners).
    Fallback: SSD Res10 (OpenCV DNN) or Haar cascade with aspect-preserving square crop.

Alignment:
    Umeyama 2D similarity transform mapping the 5 detected landmarks to the
    canonical 112x112 ArcFace template (identical to InsightFace norm_crop).

Recognition:
    InsightFace buffalo_l ArcFace model (w600k_r50.onnx, 512-dim) with horizontal
    flip-test embedding aggregation for pose/lighting robustness.

Public API:  pipeline.get_embedding(bgr_image) -> np.ndarray | None
"""

from __future__ import annotations

from functools import lru_cache
import logging
import math
from pathlib import Path
import threading
import time
from typing import List, Optional, Tuple, Dict, Any
import urllib.request
from urllib.error import URLError

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Primary detector + 5-point landmarks (YuNet ONNX, ~230 KB) ────────────
YUNET_URLS: List[str] = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
]

# ── Fallback detection model (SSD Res10) ──────────────────────────────────
DETECTION_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
    "face_detector/deploy.prototxt"
)
DETECTION_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)

# ── Recognition model (ArcFace w600k_r50, 512-dim) ───────────────────────
ARCFACE_URLS: List[str] = [
    "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx",
]

# Canonical 112x112 ArcFace 5-point landmark coordinates:
# [left_eye (viewer left), right_eye (viewer right), nose_tip, left_mouth, right_mouth]
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Canonical 160x160 Full-Head 5-point landmark coordinates (zoomed out to include
# crown hair, side hair length/volume, ears, jawline, and facial hair):
FULL_HEAD_SIZE = 160
FULL_HEAD_DST = np.array(
    [
        [64.5, 80.0],
        [95.5, 80.0],
        [80.0, 96.5],
        [67.5, 114.0],
        [92.5, 114.0],
    ],
    dtype=np.float32,
)


def _ensure_file(path: Path, url: str) -> Path:
    """Download *url* to *path* if it does not already exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp, open(path, "wb") as fh:
            fh.write(resp.read())
    return path


def _ensure_file_any(path: Path, urls: List[str]) -> Path:
    """Try each URL in *urls* until one succeeds, then cache at *path*."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception = RuntimeError("No URLs provided")
    for url in urls:
        try:
            log.info("Downloading model from %s …", url)
            with urllib.request.urlopen(url, timeout=120) as resp, open(path, "wb") as fh:
                fh.write(resp.read())
            return path
        except Exception as exc:
            log.warning("  failed (%s), trying next mirror.", exc)
            last_exc = exc
            if path.exists():
                path.unlink(missing_ok=True)
    raise last_exc


def _umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> np.ndarray:
    """Estimate 2D similarity transform matrix (2x3) via Umeyama's method.

    Matches InsightFace's ``face_align.norm_crop`` alignment transform.
    """
    num, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T[:dim, :]
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        var_sum = src_demean.var(axis=0).sum()
        scale = 1.0 / var_sum * (S @ d) if var_sum > 1e-8 else 1.0
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale
    return T[:dim, :].astype(np.float32)


def _square_crop(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    scale: float = 1.1,
    size: int = 112,
) -> np.ndarray:
    """Extract an aspect-preserving square face/head crop and resize to (size, size)."""
    ih, iw = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw * 0.5, y1 + bh * 0.5
    side = max(bw, bh) * scale
    half = side * 0.5

    sx1 = int(round(cx - half))
    sy1 = int(round(cy - half))
    sx2 = int(round(cx + half))
    sy2 = int(round(cy + half))

    pad_left = max(0, -sx1)
    pad_top = max(0, -sy1)
    pad_right = max(0, sx2 - iw)
    pad_bottom = max(0, sy2 - ih)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101
        )
        sx1 += pad_left
        sx2 += pad_left
        sy1 += pad_top
        sy2 += pad_top

    crop = image[max(0, sy1):max(1, sy2), max(0, sx1):max(1, sx2)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def _extract_full_head_features(head_bgr: np.ndarray) -> Dict[str, Any]:
    """Extract hair color, hair style/silhouette, and full-head vibe vectors + human-readable traits
    from a 160x160 landmark-aligned full-head BGR crop.
    """
    if head_bgr.shape[:2] != (FULL_HEAD_SIZE, FULL_HEAD_SIZE):
        head_bgr = cv2.resize(
            head_bgr, (FULL_HEAD_SIZE, FULL_HEAD_SIZE), interpolation=cv2.INTER_AREA
        )

    # Bilaterally smooth slightly for color analysis while keeping original for texture
    smooth_bgr = cv2.GaussianBlur(head_bgr, (3, 3), 0)
    lab = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Gradients & high-frequency texture energy
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    gangle = (np.arctan2(np.abs(gy), np.abs(gx) + 1e-5) / (math.pi * 0.5))  # [0..1]
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    # Reference skin tone from forehead + left/right cheeks
    skin_fh = lab[52:64, 64:96].reshape(-1, 3)
    skin_ch_l = lab[84:102, 54:72].reshape(-1, 3)
    skin_ch_r = lab[84:102, 88:106].reshape(-1, 3)
    skin_pixels = np.concatenate([skin_fh, skin_ch_l, skin_ch_r], axis=0)
    skin_mean = skin_pixels.mean(axis=0)
    skin_std = skin_pixels.std(axis=0) + 1.0

    # Background corner reference (top-left & top-right 14x14 corners)
    bg_corners = np.concatenate(
        [lab[0:14, 0:14].reshape(-1, 3), lab[0:14, 146:160].reshape(-1, 3)], axis=0
    )
    bg_mean = bg_corners.mean(axis=0)

    # Hair anatomical regions on the 160x160 aligned head
    crown_lab = lab[10:50, 44:116]
    crown_hsv = hsv[10:50, 44:116]
    side_up_lab = np.concatenate(
        [lab[24:76, 20:46].reshape(-1, 3), lab[24:76, 114:140].reshape(-1, 3)], axis=0
    )
    side_up_hsv = np.concatenate(
        [hsv[24:76, 20:46].reshape(-1, 3), hsv[24:76, 114:140].reshape(-1, 3)], axis=0
    )
    side_lo_lab = np.concatenate(
        [lab[76:136, 16:44].reshape(-1, 3), lab[76:136, 116:144].reshape(-1, 3)], axis=0
    )

    # ── 1. Hair Color & Tone Vector ───────────────────────────────────────
    hist_crown_lab = cv2.calcHist(
        [crown_lab.astype(np.uint8)], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256]
    ).flatten()
    hist_crown_hs = cv2.calcHist(
        [crown_hsv.astype(np.uint8)], [0, 1], None, [8, 4], [0, 180, 0, 256]
    ).flatten()

    crown_flat_lab = crown_lab.reshape(-1, 3)
    crown_flat_hsv = crown_hsv.reshape(-1, 3)
    # Filter out obvious background pixels in crown (closest to bg_mean and low gradient)
    crown_bg_dist = np.linalg.norm(crown_flat_lab - bg_mean, axis=1)
    crown_mask = crown_bg_dist > 14.0
    active_hair_lab = crown_flat_lab[crown_mask] if crown_mask.sum() >= 30 else crown_flat_lab
    active_hair_hsv = crown_flat_hsv[crown_mask] if crown_mask.sum() >= 30 else crown_flat_hsv

    hair_lab_mean = active_hair_lab.mean(axis=0)
    hair_lab_std = active_hair_lab.std(axis=0)
    hair_lab_q25 = np.percentile(active_hair_lab, 25, axis=0)
    hair_lab_q75 = np.percentile(active_hair_lab, 75, axis=0)
    hair_hsv_mean = active_hair_hsv.mean(axis=0)
    side_up_mean = side_up_lab.mean(axis=0)
    side_lo_mean = side_lo_lab.mean(axis=0)
    rel_hair_skin = (hair_lab_mean - skin_mean) / 50.0

    stats_color = np.concatenate(
        [
            hair_lab_mean / 255.0,
            hair_lab_std / 128.0,
            hair_lab_q25 / 255.0,
            hair_lab_q75 / 255.0,
            hair_hsv_mean / np.array([180.0, 255.0, 255.0], dtype=np.float32),
            side_up_mean / 255.0,
            side_lo_mean / 255.0,
            rel_hair_skin,
        ]
    ).astype(np.float32)

    hair_color_vec = _norm_vec(
        np.concatenate(
            [
                _norm_vec(hist_crown_lab) * 1.2,
                _norm_vec(hist_crown_hs) * 0.8,
                _norm_vec(stats_color) * 1.4,
            ]
        )
    )

    # ── 2. Hair Style, Silhouette, Length & Texture Vector ────────────────
    # 8x8 spatial grid over the 160x160 head (20x20 cells)
    grid_feats = []
    for gy_i in range(8):
        y0, y1 = gy_i * 20, (gy_i + 1) * 20
        for gx_i in range(8):
            x0, x1 = gx_i * 20, (gx_i + 1) * 20
            cell_gx = float(np.abs(gx[y0:y1, x0:x1]).mean()) / 60.0
            cell_gy = float(np.abs(gy[y0:y1, x0:x1]).mean()) / 60.0
            cell_lap = float(lap[y0:y1, x0:x1].mean()) / 60.0
            cell_lab_mean = lab[y0:y1, x0:x1].mean(axis=(0, 1))
            dist_skin = float(np.linalg.norm(cell_lab_mean - skin_mean)) / 100.0
            dist_bg = float(np.linalg.norm(cell_lab_mean - bg_mean)) / 100.0
            grid_feats.extend([cell_gx, cell_gy, cell_lap, dist_skin, dist_bg])

    # Zone orientation histograms (captures straight vertical vs wavy horizontal vs curly isotropic)
    zone_slices = [
        (slice(8, 48), slice(44, 116)),    # crown
        (slice(36, 66), slice(48, 112)),   # bangs / forehead fringe
        (slice(24, 78), slice(18, 48)),    # upper left temple/hair
        (slice(24, 78), slice(112, 142)),  # upper right temple/hair
        (slice(78, 138), slice(16, 46)),   # lower left side hair
        (slice(78, 138), slice(114, 144)), # lower right side hair
    ]
    orient_feats = []
    for ys, xs in zone_slices:
        z_ang = gangle[ys, xs].flatten()
        z_mag = gmag[ys, xs].flatten()
        hist_o, _ = np.histogram(z_ang, bins=6, range=(0.0, 1.0), weights=z_mag)
        orient_feats.append(_norm_vec(hist_o.astype(np.float32)))

    hair_style_vec = _norm_vec(
        np.concatenate(
            [
                _norm_vec(np.array(grid_feats, dtype=np.float32)) * 1.3,
                _norm_vec(np.concatenate(orient_feats)) * 0.9,
            ]
        )
    )

    # ── 3. Full-Head Vibe, Complexion & Facial-Hair Vector ────────────────
    jaw_lab = lab[106:142, 58:102]
    lip_lab = lab[98:112, 66:94]
    brow_lab = lab[64:76, 52:108]

    jaw_diff = (jaw_lab.mean(axis=(0, 1)) - skin_mean) / 50.0
    lip_diff = (lip_lab.mean(axis=(0, 1)) - skin_mean) / 50.0
    brow_diff = (brow_lab.mean(axis=(0, 1)) - skin_mean) / 50.0

    cheek_tex = float(gmag[84:102, 54:72].mean() + gmag[84:102, 88:106].mean()) * 0.5 + 1.0
    jaw_tex_ratio = float(gmag[108:142, 60:100].mean()) / cheek_tex
    lip_tex_ratio = float(gmag[98:112, 66:94].mean()) / cheek_tex
    brow_tex_ratio = float(gmag[64:76, 52:108].mean()) / cheek_tex

    # Exposure-normalised 14x14 spatial head colour map
    tiny_lab = cv2.resize(lab, (14, 14), interpolation=cv2.INTER_AREA)
    rel_tiny_lab = ((tiny_lab - skin_mean) / 60.0).flatten().astype(np.float32)

    facial_feats = np.concatenate(
        [
            jaw_diff,
            lip_diff,
            brow_diff,
            skin_mean / 255.0,
            skin_std / 64.0,
            np.array([jaw_tex_ratio, lip_tex_ratio, brow_tex_ratio], dtype=np.float32) * 0.5,
        ]
    ).astype(np.float32)

    head_vibe_vec = _norm_vec(
        np.concatenate(
            [
                _norm_vec(facial_feats) * 1.2,
                _norm_vec(rel_tiny_lab) * 1.0,
            ]
        )
    )

    # ── 4. Human-Readable Traits Classification ───────────────────────────
    crown_skin_dist = float(np.linalg.norm(hair_lab_mean - skin_mean))
    crown_tex = float(gmag[12:48, 48:112].mean())
    L_h, a_h, b_h = float(hair_lab_mean[0]), float(hair_lab_mean[1] - 128.0), float(hair_lab_mean[2] - 128.0)
    S_h = float(hair_hsv_mean[1])

    if crown_skin_dist < 13.5 and crown_tex < 20.0:
        hair_color_label = "Close-Cropped / Shaved"
        hair_style_label = "Minimal / Buzzed"
    else:
        if L_h < 62:
            hair_color_label = "Jet Black / Dark"
        elif L_h < 105:
            if a_h > 9 and b_h > 13:
                hair_color_label = "Auburn / Dark Red"
            else:
                hair_color_label = "Dark Brown"
        elif L_h < 152:
            if a_h > 10 and b_h > 16:
                hair_color_label = "Red / Copper"
            elif S_h < 36:
                hair_color_label = "Ash Brown / Salt & Pepper"
            elif b_h > 20:
                hair_color_label = "Golden / Chestnut Brown"
            else:
                hair_color_label = "Medium Brown"
        else:
            if S_h < 32 and abs(b_h) < 12:
                hair_color_label = "Silver / Gray"
            elif a_h > 10 and b_h > 20:
                hair_color_label = "Strawberry Blonde"
            else:
                hair_color_label = "Blonde / Golden"

        # Hair style classification
        lo_side_bg_dist = float(np.linalg.norm(side_lo_mean - bg_mean))
        lo_side_skin_dist = float(np.linalg.norm(side_lo_mean - skin_mean))
        lo_side_tex = float(gmag[78:136, 18:44].mean() + gmag[78:136, 116:142].mean()) * 0.5
        up_side_tex = float(gmag[24:76, 20:46].mean() + gmag[24:76, 114:140].mean()) * 0.5
        curl_index = float(lap[10:60, 36:124].mean()) / max(crown_tex, 1.0)

        if lo_side_bg_dist > 18.0 and lo_side_skin_dist > 15.0 and lo_side_tex > 21.0:
            if curl_index > 1.18:
                hair_style_label = "Long Waves / Curls"
            else:
                hair_style_label = "Long / Side-Framing"
        elif curl_index > 1.22 and crown_tex > 28.0:
            hair_style_label = "Curly / High-Volume"
        elif up_side_tex > 25.0:
            hair_style_label = "Medium / Wavy"
        else:
            hair_style_label = "Short / Classic Cut"

    jaw_darkening = float(skin_mean[0] - jaw_lab.mean(axis=(0, 1))[0])
    if jaw_darkening > 20.0 and jaw_tex_ratio > 1.25:
        facial_hair_label = "Beard / Mustache"
    elif jaw_darkening > 10.0 and jaw_tex_ratio > 1.10:
        facial_hair_label = "Stubble / Defined Jaw"
    else:
        facial_hair_label = "Clean / Smooth"

    return {
        "hair_color_vec": hair_color_vec,
        "hair_style_vec": hair_style_vec,
        "head_vibe_vec": head_vibe_vec,
        "traits": {
            "hair_color": hair_color_label,
            "hair_style": hair_style_label,
            "facial_hair": facial_hair_label,
        },
    }


class FacePipeline:
    """Detect and 5-point align the most prominent face, returning its 512-D ArcFace embedding."""

    embedding_size: int = 512

    def __init__(self) -> None:
        models_dir = Path(__file__).resolve().parents[1] / "models"
        self._fallback = False
        self._yunet = None
        self._detector = None
        self._use_dnn_det = False
        self._det_lock = threading.Lock()

        # ── Primary detector: YuNet (with 5-point landmarks) ─────────────
        if hasattr(cv2, "FaceDetectorYN"):
            try:
                yunet_path = _ensure_file_any(
                    models_dir / "face_detection_yunet_2023mar.onnx",
                    YUNET_URLS,
                )
                self._yunet = cv2.FaceDetectorYN.create(
                    model=str(yunet_path),
                    config="",
                    input_size=(320, 320),
                    score_threshold=0.55,
                    nms_threshold=0.3,
                    top_k=50,
                )
                log.info("YuNet 5-point face detector loaded from %s", yunet_path)
            except Exception as exc:
                log.warning("YuNet detector unavailable (%s); falling back to SSD.", exc)
                self._yunet = None

        # ── Secondary detector: SSD Res10 (if YuNet failed) & Haar cascade ──
        if self._yunet is None and hasattr(cv2.dnn, "readNetFromCaffe"):
            try:
                proto = _ensure_file(models_dir / "deploy.prototxt", DETECTION_PROTO_URL)
                weights = _ensure_file(models_dir / "res10.caffemodel", DETECTION_MODEL_URL)
                self._detector = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
                self._use_dnn_det = True
            except Exception as exc:
                log.warning("SSD detector unavailable (%s); using Haar cascade fallback.", exc)

        if not self._use_dnn_det:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(cascade)

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
                _winSize=(112, 112),
                _blockSize=(32, 32),
                _blockStride=(16, 16),
                _cellSize=(16, 16),
                _nbins=9,
            )

    # ------------------------------------------------------------------
    def _detect_yunet_raw(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Run YuNet on *image* (downscaling large photos for speed/accuracy)."""
        if self._yunet is None:
            return None
        ih, iw = image.shape[:2]
        if ih < 16 or iw < 16:
            return None

        max_side = 640
        scale = 1.0
        work = image
        if max(ih, iw) > max_side:
            scale = max_side / float(max(ih, iw))
            rw = max(16, int(round(iw * scale)))
            rh = max(16, int(round(ih * scale)))
            work = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
        else:
            rw, rh = iw, ih

        with self._det_lock:
            self._yunet.setInputSize((rw, rh))
            _, faces = self._yunet.detect(work)
        if faces is None or len(faces) == 0:
            return None

        faces = faces.astype(np.float32)
        if scale != 1.0:
            faces[:, :14] /= scale
        return faces

    @staticmethod
    def _pick_most_prominent(faces: np.ndarray, iw: int, ih: int) -> np.ndarray:
        """Select the main subject's face based on confidence, size, and centrality."""
        if len(faces) == 1:
            return faces[0]

        img_area = float(max(iw * ih, 1))
        best_idx = 0
        best_score = -1.0
        for i, det in enumerate(faces):
            x, y, w, h = det[0:4]
            conf = float(det[14])
            area_ratio = max(float(w * h), 1.0) / img_area
            cx = x + w * 0.5
            cy = y + h * 0.5
            dx = (cx - iw * 0.5) / max(iw * 0.5, 1.0)
            dy = (cy - ih * 0.5) / max(ih * 0.5, 1.0)
            dist = min(1.0, math.hypot(dx, dy))
            centrality = 1.0 - 0.35 * dist
            prominence = (conf ** 2) * (area_ratio ** 0.45) * centrality
            if prominence > best_score:
                best_score = prominence
                best_idx = i
        return faces[best_idx]

    def _detect_crop_with_meta(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Return (aligned 112x112 face crop, detection metadata dict)."""
        meta: Dict[str, Any] = {
            "detector": "None",
            "face_confidence": 0.0,
            "face_bbox": None,
            "landmarks": None,
        }
        if image is None or image.size == 0:
            return None, meta
        ih, iw = image.shape[:2]
        if ih < 16 or iw < 16:
            return None, meta

        # 1. Try YuNet with 5-point landmark similarity alignment
        if self._yunet is not None:
            faces = self._detect_yunet_raw(image)
            work_img = image
            pad_x = pad_y = 0

            # Tight-crop fallback: pad image by 25% if face fills the frame edges
            if faces is None:
                pad_y, pad_x = int(ih * 0.25), int(iw * 0.25)
                work_img = cv2.copyMakeBorder(
                    image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
                )
                faces = self._detect_yunet_raw(work_img)

            if faces is not None and len(faces) > 0:
                wh, ww = work_img.shape[:2]
                best = self._pick_most_prominent(faces, ww, wh)
                landmarks = best[4:14].reshape(5, 2).astype(np.float32)
                M = _umeyama(landmarks, ARCFACE_DST, estimate_scale=True)
                if not np.isnan(M).any():
                    crop = cv2.warpAffine(
                        work_img,
                        M,
                        (112, 112),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                    M_head = _umeyama(landmarks, FULL_HEAD_DST, estimate_scale=True)
                    if not np.isnan(M_head).any():
                        head_crop = cv2.warpAffine(
                            work_img,
                            M_head,
                            (FULL_HEAD_SIZE, FULL_HEAD_SIZE),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101,
                        )
                    else:
                        head_crop = cv2.resize(crop, (FULL_HEAD_SIZE, FULL_HEAD_SIZE))
                    head_feats = _extract_full_head_features(head_crop)
                    # Map bbox & landmarks back to original image normalized [0..1]
                    bx = (float(best[0]) - pad_x) / max(iw, 1)
                    by = (float(best[1]) - pad_y) / max(ih, 1)
                    bw = float(best[2]) / max(iw, 1)
                    bh = float(best[3]) / max(ih, 1)
                    norm_lms = [
                        [
                            round((float(pt[0]) - pad_x) / max(iw, 1), 4),
                            round((float(pt[1]) - pad_y) / max(ih, 1), 4),
                        ]
                        for pt in landmarks
                    ]
                    meta.update(
                        {
                            "detector": "YuNet 5-Point Landmark",
                            "face_confidence": round(float(best[14]), 4),
                            "face_bbox": [
                                round(max(0.0, bx), 4),
                                round(max(0.0, by), 4),
                                round(min(1.0, bw), 4),
                                round(min(1.0, bh), 4),
                            ],
                            "landmarks": norm_lms,
                            "head_features": head_feats,
                        }
                    )
                    return crop, meta

        # 2. Fallback to SSD / Haar cascade with aspect-preserving square crop
        if self._use_dnn_det and self._detector is not None:
            for pad_ratio in (0.0, 0.2):
                py, px = int(ih * pad_ratio), int(iw * pad_ratio)
                candidate = (
                    image
                    if pad_ratio == 0.0
                    else cv2.copyMakeBorder(
                        image, py, py, px, px, cv2.BORDER_REPLICATE
                    )
                )
                ch, cw = candidate.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(candidate, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0),
                )
                with self._det_lock:
                    self._detector.setInput(blob)
                    dets = self._detector.forward()
                if dets.shape[2] == 0:
                    continue

                valid = [
                    dets[0, 0, i]
                    for i in range(dets.shape[2])
                    if float(dets[0, 0, i, 2]) > 0.5
                ]
                if not valid:
                    continue

                best_box = None
                best_conf = 0.0
                best_area = -1.0
                for d in valid:
                    box = d[3:7] * np.array([cw, ch, cw, ch])
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(cw, x2), min(ch, y2)
                    area = (x2 - x1) * (y2 - y1) * float(d[2])
                    if x2 > x1 and y2 > y1 and area > best_area:
                        best_area = area
                        best_box = (x1, y1, x2, y2)
                        best_conf = float(d[2])
                if best_box is not None:
                    x1, y1, x2, y2 = best_box
                    head_crop = _square_crop(
                        candidate, *best_box, scale=1.75, size=FULL_HEAD_SIZE
                    )
                    meta.update(
                        {
                            "detector": "SSD Res10 DNN",
                            "face_confidence": round(best_conf, 4),
                            "face_bbox": [
                                round(max(0.0, (x1 - px) / max(iw, 1)), 4),
                                round(max(0.0, (y1 - py) / max(ih, 1)), 4),
                                round(min(1.0, (x2 - x1) / max(iw, 1)), 4),
                                round(min(1.0, (y2 - y1) / max(ih, 1)), 4),
                            ],
                            "head_features": _extract_full_head_features(head_crop),
                        }
                    )
                    return _square_crop(candidate, *best_box), meta
            return None, meta

        if self._detector is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._detector.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                return None, meta
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            head_crop = _square_crop(
                image, x, y, x + w, y + h, scale=1.75, size=FULL_HEAD_SIZE
            )
            meta.update(
                {
                    "detector": "Haar Cascade",
                    "face_confidence": 0.85,
                    "face_bbox": [
                        round(x / max(iw, 1), 4),
                        round(y / max(ih, 1), 4),
                        round(w / max(iw, 1), 4),
                        round(h / max(ih, 1), 4),
                    ],
                    "head_features": _extract_full_head_features(head_crop),
                }
            )
            return _square_crop(image, x, y, x + w, y + h), meta

        return None, meta

    def _detect_crop(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return an aligned 112x112 face crop or None if no face is found."""
        crop, _ = self._detect_crop_with_meta(image)
        return crop

    # ------------------------------------------------------------------
    def _embed_crops_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """Run ArcFace (or fallback HOG) on a list of 112x112 BGR crops."""
        if self._fallback:
            out = []
            for face in faces:
                hog = self._hog.compute(face).flatten().astype(np.float32)
                emb = hog[:512] if hog.size >= 512 else np.pad(hog, (0, 512 - hog.size))
                out.append(emb)
            return np.stack(out).astype(np.float32)

        out = []
        for face in faces:
            rgb = face[:, :, ::-1].astype(np.float32)
            blob = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[np.newaxis]
            emb = self._recognizer.run(None, {self._rec_input: blob})[0][0]
            out.append(emb.astype(np.float32))
        return np.stack(out).astype(np.float32)

    def get_embedding_with_info(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Detect, align, and embed face while recording detailed timing and landmark info."""
        t0 = time.perf_counter()
        face, meta = self._detect_crop_with_meta(image)
        detect_align_ms = (time.perf_counter() - t0) * 1000.0
        meta["detect_align_ms"] = round(detect_align_ms, 2)
        meta["model"] = (
            "HOG Descriptor (512-D)"
            if self._fallback
            else "ArcFace ResNet-50 (w600k_r50, 512-D)"
        )
        if face is None:
            meta["embed_ms"] = 0.0
            return None, meta

        t1 = time.perf_counter()
        if self._fallback:
            emb = self._embed_crops_batch([face])[0]
        else:
            embs = self._embed_crops_batch([face, cv2.flip(face, 1)])
            emb = embs[0] + embs[1]
        embed_ms = (time.perf_counter() - t1) * 1000.0
        meta["embed_ms"] = round(embed_ms, 2)

        emb = emb.astype("float32")
        norm = float(np.linalg.norm(emb))
        return (None if norm == 0.0 else emb / norm), meta

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect, 5-point align, and return an L2-normalised 512-D float32 embedding."""
        emb, _ = self.get_embedding_with_info(image)
        return emb


@lru_cache()
def get_pipeline() -> FacePipeline:
    return FacePipeline()
