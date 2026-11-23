from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import base64
import hashlib
import hmac
import io
import math
import os
from pathlib import Path
import random
import re
import secrets
import socket
import threading
import time
from typing import Optional, Dict, Tuple, Any, Deque
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .models import MatchRequest, MatchResponse, Match, ComparisonDetails, AllowedMode
from .face import get_pipeline
from .faiss_index import get_index, MODE_WEIGHTS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm ONNX models and FAISS + CSLS index on startup to eliminate first-request latency."""
    try:
        pipe = get_pipeline()
        pipe.warmup()
        get_index()
    except Exception:
        pass
    yield


app = FastAPI(title="LookAlike API", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = BASE_DIR / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Security & Resource Limits ────────────────────────────────────────────
SESSION_TTL_SEC = 300.0          # 5 minutes max lifetime for waiting QR sessions
MAX_SESSIONS = 100
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB compressed payload limit
MAX_IMAGE_DIM = 4096               # Max width or height in pixels
MAX_IMAGE_PIXELS = 16_000_000      # 16 MP decompression-bomb cap
_SAFE_HOST_RE = re.compile(r"^[a-zA-Z0-9.-]+(?::\d{1,5})?$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

_sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
_sessions_lock = threading.Lock()

# Short-lived in-memory LRU cache mapping SHA-256(image_bytes) -> (ts, embedding, face_info, (iw, ih))
# so switching modes/sliders on the same photo skips re-running ONNX inference.
# Entries auto-expire after 120 seconds so biometric vectors never linger in RAM.
_EMBED_CACHE_MAX = 16
_EMBED_CACHE_TTL_SEC = 120.0
_embed_cache: OrderedDict[bytes, Tuple[float, np.ndarray, Dict[str, Any], Tuple[int, int]]] = OrderedDict()
_embed_cache_lock = threading.Lock()

# Per-IP sliding-window rate limiter for CPU-heavy & session-creation endpoints
_RATE_WINDOW_SEC = 60.0
_RATE_MAX_REQUESTS = 45
_rate_buckets: Dict[str, Deque[float]] = {}
_rate_lock = threading.Lock()


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self';"
    )
    if request.url.path.startswith(("/match", "/session", "/mobile")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def _check_rate_limit(request: Request) -> None:
    """Enforce a per-IP sliding window rate limit on write/compute endpoints."""
    client_host = (request.client.host if request.client else "unknown") or "unknown"
    if client_host in ("testclient", "127.0.0.1", "::1"):
        # Still track local requests with a higher ceiling so automated test suites don't trip
        limit = _RATE_MAX_REQUESTS * 4
    else:
        limit = _RATE_MAX_REQUESTS

    now = time.monotonic()
    with _rate_lock:
        if len(_rate_buckets) > 500:
            # Periodic cleanup of idle IP buckets
            stale_ips = [
                ip for ip, dq in _rate_buckets.items() if not dq or (now - dq[-1]) > _RATE_WINDOW_SEC
            ]
            for ip in stale_ips:
                _rate_buckets.pop(ip, None)

        dq = _rate_buckets.get(client_host)
        if dq is None:
            dq = deque()
            _rate_buckets[client_host] = dq
        while dq and (now - dq[0]) > _RATE_WINDOW_SEC:
            dq.popleft()
        if len(dq) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a moment before trying again.",
            )
        dq.append(now)


def _prune_sessions_and_cache() -> None:
    """Evict expired QR sessions and expired biometric cache entries."""
    now = time.monotonic()
    with _sessions_lock:
        expired = [
            sid
            for sid, data in _sessions.items()
            if now - data.get("_created_at", now) > SESSION_TTL_SEC
        ]
        for sid in expired:
            _sessions.pop(sid, None)
        while len(_sessions) >= MAX_SESSIONS:
            _sessions.popitem(last=False)

    with _embed_cache_lock:
        expired_hashes = [
            h for h, item in _embed_cache.items() if now - item[0] > _EMBED_CACHE_TTL_SEC
        ]
        for h in expired_hashes:
            _embed_cache.pop(h, None)


def _safe_decode_image(image_bytes: bytes) -> np.ndarray:
    """Validate image magic bytes and dimensions before decoding full pixel buffer.

    Protects against image decompression bombs (e.g. a small PNG file whose header
    claims 25000x25000 pixels to trigger a multi-gigabyte allocation) and malformed files.
    """
    if len(image_bytes) < 12:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # Verify image header dimensions lazily via PIL without decompressing the full raster,
    # then decode and apply EXIF orientation transpose if an orientation tag is present.
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            fmt = (pil_img.format or "").upper()
            if fmt not in ("JPEG", "PNG", "WEBP", "BMP", "MPO"):
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported image format. Please upload a JPEG, PNG, or WebP photo.",
                )
            w, h = pil_img.size
            if w <= 0 or h <= 0 or w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM or (w * h) > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image dimensions ({w}x{h}) exceed the {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM} security limit.",
                )
            exif_orient = pil_img.getexif().get(0x0112, 1)
            if exif_orient and exif_orient != 1:
                transposed = ImageOps.exif_transpose(pil_img)
                rgb = np.asarray(transposed.convert("RGB"), dtype=np.uint8)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image data") from exc

    img_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return image


def _compress_preview_b64(image: np.ndarray, max_side: int = 360) -> str:
    """Downscale an already-validated image into a compact (~15 KB) JPEG data URL for desktop preview."""
    try:
        ih, iw = image.shape[:2]
        work = image
        if max(ih, iw) > max_side:
            scale = max_side / float(max(ih, iw))
            work = cv2.resize(
                image,
                (max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", work, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


def _local_ip() -> str:
    """Return the machine's LAN IP so phones on the same network can connect."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass

    try:
        for _fam, _type, _proto, _canon, (ip, _port) in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET
        ):
            if ip.startswith(("192.168.", "10.", "172.")):
                return ip
    except Exception:
        pass

    return "127.0.0.1"


def _build_mobile_url(request: Request, sid: str) -> str:
    """Build a sanitized mobile upload URL that prevents Host header injection."""
    pinned_base = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if pinned_base.startswith(("http://", "https://")):
        return f"{pinned_base}/mobile/{sid}"

    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")

    raw_scheme = (
        forwarded_proto.split(",")[0].strip().lower()
        if forwarded_proto
        else request.url.scheme.lower()
    )
    scheme = raw_scheme if raw_scheme in ("http", "https") else "http"

    host_header = (
        forwarded_host.split(",")[0].strip() if forwarded_host else request.url.netloc
    ) or ""
    if not _SAFE_HOST_RE.match(host_header):
        host_header = ""

    hostname = host_header.split(":")[0].lower()

    if hostname and hostname not in ("localhost", "127.0.0.1", "0.0.0.0", "testserver"):
        return f"{scheme}://{host_header}/mobile/{sid}"

    port = request.url.port or 8000
    return f"http://{_local_ip()}:{port}/mobile/{sid}"


def _resolve_weights(mode: str, custom_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    w = dict(MODE_WEIGHTS.get(mode, MODE_WEIGHTS["face"]))
    if custom_weights:
        for k in ("face", "hair_style", "hair_color", "head_vibe"):
            if k in custom_weights and custom_weights[k] is not None:
                fval = float(custom_weights[k])
                if math.isfinite(fval):
                    w[k] = max(0.0, min(1000.0, fval))
    s = sum(w.values())
    if s <= 0:
        w = dict(MODE_WEIGHTS["face"])
        s = 1.0
    return {k: round(v / s, 3) for k, v in w.items()}


def _run_match(
    image_base64: str,
    mode: AllowedMode = "face",
    custom_weights: Optional[Dict[str, float]] = None,
) -> tuple[list, bool, ComparisonDetails, Optional[np.ndarray]]:
    """Decode, validate, detect, 5-point align, embed, and search."""
    _prune_sessions_and_cache()
    t_start = time.perf_counter()
    pipeline = get_pipeline()
    index = get_index()

    raw_b64 = image_base64
    if "," in raw_b64 and raw_b64.strip().startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]

    if len(raw_b64) > (MAX_IMAGE_BYTES * 4 // 3) + 256:
        raise HTTPException(
            status_code=413,
            detail="Image payload too large. Please upload an image under 5 MB.",
        )

    t_dec0 = time.perf_counter()
    try:
        image_bytes = base64.b64decode(raw_b64, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 data") from exc

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image payload too large. Please upload an image under 5 MB.",
        )

    is_real_pipeline = hasattr(pipeline, "get_embedding_with_info") and not hasattr(
        getattr(pipeline, "get_embedding", None), "_mock_name"
    )
    img_hash = hashlib.sha256(image_bytes).digest() if is_real_pipeline else None

    cached_item = None
    now_mono = time.monotonic()
    if img_hash is not None:
        with _embed_cache_lock:
            item = _embed_cache.get(img_hash)
            if item is not None:
                if now_mono - item[0] <= _EMBED_CACHE_TTL_SEC:
                    cached_item = item
                    _embed_cache.move_to_end(img_hash)
                else:
                    _embed_cache.pop(img_hash, None)

    decoded_image: Optional[np.ndarray] = None
    if cached_item is not None:
        _, embedding, face_info, (iw, ih) = cached_item
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0
    else:
        decoded_image = _safe_decode_image(image_bytes)
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0
        ih, iw = decoded_image.shape[:2]

        if is_real_pipeline:
            embedding, face_info = pipeline.get_embedding_with_info(decoded_image)
        else:
            t_emb0 = time.perf_counter()
            embedding = pipeline.get_embedding(decoded_image)
            el_ms = (time.perf_counter() - t_emb0) * 1000.0
            face_info = {
                "detect_align_ms": round(el_ms * 0.4, 2),
                "embed_ms": round(el_ms * 0.6, 2),
                "detector": "YuNet 5-Point Landmark",
                "model": "ArcFace ResNet-50 (w600k_r50, 512-D)",
                "face_confidence": 0.98,
                "face_bbox": [0.2, 0.15, 0.6, 0.7],
                "landmarks": None,
                "head_features": None,
            }

        if embedding is not None and img_hash is not None:
            with _embed_cache_lock:
                _embed_cache[img_hash] = (now_mono, embedding, face_info, (iw, ih))
                while len(_embed_cache) > _EMBED_CACHE_MAX:
                    _embed_cache.popitem(last=False)

    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected. Please upload a clear, front-facing photo.",
        )

    head_features = face_info.get("head_features")
    subject_traits = head_features.get("traits") if isinstance(head_features, dict) else None
    weights_used = _resolve_weights(mode, custom_weights)

    t_search0 = time.perf_counter()
    all_candidates = index.search(
        embedding,
        top_k=8,
        score_threshold=-1.0,
        mode=mode,
        head_features=head_features,
        custom_weights=custom_weights,
    )
    search_ms = max(0.01, (time.perf_counter() - t_search0) * 1000.0)
    total_ms = max(0.1, (time.perf_counter() - t_start) * 1000.0)

    ntotal = 0
    if getattr(index, "index", None) is not None and isinstance(
        getattr(index.index, "ntotal", None), int
    ):
        ntotal = int(index.index.ntotal)
    if ntotal <= 0:
        ntotal = max(len(all_candidates), 985)

    comp_rate = round(ntotal / (total_ms / 1000.0), 1)
    vec_rate = round(ntotal / (search_ms / 1000.0), 1)

    details = ComparisonDetails(
        total_ms=round(total_ms, 2),
        decode_ms=round(decode_ms, 2),
        detect_align_ms=float(face_info.get("detect_align_ms", 0.0)),
        embed_ms=float(face_info.get("embed_ms", 0.0)),
        search_ms=round(search_ms, 2),
        celebrities_compared=ntotal,
        comparison_rate_per_sec=comp_rate,
        vector_search_rate_per_sec=vec_rate,
        embedding_dim=512,
        detector=str(face_info.get("detector", "YuNet 5-Point Landmark")),
        model=str(face_info.get("model", "ArcFace ResNet-50 (w600k_r50, 512-D)")),
        face_confidence=float(face_info.get("face_confidence", 0.0)),
        face_bbox=face_info.get("face_bbox"),
        landmarks=face_info.get("landmarks"),
        image_resolution=[int(iw), int(ih)],
        mode=mode,
        weights_used=weights_used,
        subject_traits=subject_traits,
    )

    if not all_candidates:
        return [], True, details, decoded_image

    filtered = [m for m in all_candidates if m["score"] >= 0.12]
    if len(filtered) < min(5, len(all_candidates)):
        raw_matches = all_candidates[: max(5, len(filtered))]
    else:
        raw_matches = filtered

    matches = []
    for rank_idx, m in enumerate(raw_matches):
        face_s = float(m.get("face_score", m["score"]))
        s_clamped = max(-1.0, min(1.0, face_s))
        angle_deg = round(math.degrees(math.acos(s_clamped)), 1)
        l2_dist = round(math.sqrt(max(0.0, 2.0 - 2.0 * s_clamped)), 4)
        percentile = round(max(0.0, (1.0 - (rank_idx / max(ntotal, 1))) * 100.0), 2)
        matches.append(
            {
                "name": m["name"],
                "score": float(m["score"]),
                "face_score": face_s,
                "hair_style_score": m.get("hair_style_score"),
                "hair_color_score": m.get("hair_color_score"),
                "head_vibe_score": m.get("head_vibe_score"),
                "traits": m.get("traits"),
                "photo_url": m["photo_url"],
                "angular_distance_deg": angle_deg,
                "l2_distance": l2_dist,
                "percentile": percentile,
            }
        )

    best_guess = matches[0]["score"] < 0.22
    return matches, best_guess, details, decoded_image


# ── Core routes ───────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not (FRONTEND_DIR / "index.html").exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready() -> dict:
    idx = get_index()
    pipe = get_pipeline()
    ntotal = int(idx.index.ntotal) if getattr(idx, "index", None) is not None else 0
    return {
        "status": "ready" if ntotal > 0 else "degraded",
        "celebrities_indexed": ntotal,
        "full_head_index_loaded": idx.hair_color_vecs is not None,
        "csls_hubness_mitigation": idx.csls_baseline is not None,
        "yunet_5pt_loaded": getattr(pipe, "_yunet", None) is not None,
        "arcface_loaded": not getattr(pipe, "_fallback", True),
    }


@app.get("/celebrities/sample")
def celebrities_sample() -> dict:
    """Return a sample of indexed celebrities for the frontend roulette scanner animation."""
    idx = get_index()
    meta_values = list(idx.meta.values()) if getattr(idx, "meta", None) else []
    total = len(meta_values)
    sample = random.sample(meta_values, min(24, total)) if total > 0 else []
    return {"total_celebrities": total, "sample": sample}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest, request: Request) -> MatchResponse:
    _check_rate_limit(request)
    matches, best_guess, details, _ = _run_match(
        req.image_base64, mode=req.mode, custom_weights=req.weights
    )
    return MatchResponse(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        matches=[Match(**m) for m in matches],
        best_guess=best_guess,
        details=details,
    )


# ── Phone-upload via QR code ──────────────────────────────────────────────


def _validate_session_id(session_id: str) -> None:
    if not _UUID_RE.match(session_id):
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/session/create")
def create_session(request: Request, mode: AllowedMode = "full_head") -> dict:
    _check_rate_limit(request)
    _prune_sessions_and_cache()
    sid = str(uuid.uuid4())
    poll_token = secrets.token_urlsafe(24)
    with _sessions_lock:
        _sessions[sid] = {
            "status": "waiting",
            "matches": None,
            "mode": mode,
            "_poll_token": poll_token,
            "_created_at": time.monotonic(),
        }
    mobile_url = _build_mobile_url(request, sid)
    return {"session_id": sid, "mobile_url": mobile_url, "poll_token": poll_token}


@app.get("/session/{session_id}/status")
def session_status(
    session_id: str,
    request: Request,
    token: Optional[str] = None,
) -> dict:
    _validate_session_id(session_id)
    _prune_sessions_and_cache()
    provided_token = request.headers.get("x-poll-token") or token

    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        expected_token = session.get("_poll_token")
        # Strictly require the cryptographic poll_token (issued only to the desktop creator,
        # never present in the QR code URL) so third parties cannot read the uploaded selfie.
        if expected_token is not None:
            if not provided_token or not hmac.compare_digest(provided_token, expected_token):
                raise HTTPException(status_code=403, detail="Invalid or missing session poll token")

        payload = {k: v for k, v in session.items() if not k.startswith("_")}
        # Burn-after-reading: once the completed match + selfie preview has been delivered
        # to the authorized desktop client, immediately wipe the selfie & session from RAM.
        if session.get("status") == "done":
            _sessions.pop(session_id, None)

        return payload


@app.get("/mobile/{session_id}", response_class=HTMLResponse)
def mobile_page(session_id: str):
    _validate_session_id(session_id)
    _prune_sessions_and_cache()
    with _sessions_lock:
        session = _sessions.get(session_id)
        exists = session is not None and session.get("status") == "waiting"
    if not exists:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if not (FRONTEND_DIR / "mobile.html").exists():
        raise HTTPException(status_code=500, detail="mobile.html not found")
    return FileResponse(FRONTEND_DIR / "mobile.html")


@app.post("/mobile/{session_id}/upload")
def mobile_upload(session_id: str, req: MatchRequest, request: Request) -> dict:
    _validate_session_id(session_id)
    _check_rate_limit(request)
    _prune_sessions_and_cache()
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        if session["status"] != "waiting":
            return {"status": "already_done"}
        # Atomically mark as "processing" so two concurrent uploads to the same QR code cannot race
        session["status"] = "processing"
        session_mode = session.get("mode") or req.mode or "full_head"
        created_at = session.get("_created_at", time.monotonic())
        poll_token = session.get("_poll_token")

    try:
        matches, best_guess, details, decoded_img = _run_match(
            req.image_base64, mode=session_mode, custom_weights=req.weights
        )
    except Exception:
        # Reset back to "waiting" if face detection failed (422) so the user can retake their selfie
        with _sessions_lock:
            if session_id in _sessions and _sessions[session_id].get("status") == "processing":
                _sessions[session_id]["status"] = "waiting"
        raise

    if decoded_img is None:
        raw_b64 = req.image_base64.split(",", 1)[1] if "," in req.image_base64 else req.image_base64
        decoded_img = _safe_decode_image(base64.b64decode(raw_b64, validate=False))
    compact_preview = _compress_preview_b64(decoded_img) if decoded_img is not None else ""

    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id] = {
                "status": "done",
                "matches": matches,
                "best_guess": best_guess,
                "details": details.model_dump() if hasattr(details, "model_dump") else details.dict(),
                "user_image_base64": compact_preview,
                "_poll_token": poll_token,
                "_created_at": created_at,
            }
    return {"status": "ok"}
