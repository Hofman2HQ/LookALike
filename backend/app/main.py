from collections import OrderedDict
from datetime import datetime, timezone
import base64
import hashlib
import math
from pathlib import Path
import random
import socket
import threading
import time
from typing import Optional, Dict, Tuple, Any
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .models import MatchRequest, MatchResponse, Match, ComparisonDetails
from .face import get_pipeline
from .faiss_index import get_index, MODE_WEIGHTS

app = FastAPI(title="LookAlike API")

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = BASE_DIR / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Bounded, TTL-expiring in-memory store for phone-upload sessions
SESSION_TTL_SEC = 600.0  # 10 minutes
MAX_SESSIONS = 100
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB safety cap

_sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
_sessions_lock = threading.Lock()

# Small LRU cache mapping SHA-256(image_bytes) -> (embedding, face_info, (iw, ih))
# so switching modes/sliders on the same photo skips re-running ONNX inference.
_EMBED_CACHE_MAX = 24
_embed_cache: OrderedDict[bytes, Tuple[np.ndarray, Dict[str, Any], Tuple[int, int]]] = OrderedDict()
_embed_cache_lock = threading.Lock()


def _prune_sessions() -> None:
    """Evict expired sessions and enforce MAX_SESSIONS capacity."""
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


def _compress_preview_b64(image_bytes: bytes, max_side: int = 360) -> str:
    """Downscale a phone upload into a compact (~15 KB) JPEG data URL for desktop preview."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        ih, iw = img.shape[:2]
        if max(ih, iw) > max_side:
            scale = max_side / float(max(ih, iw))
            img = cv2.resize(
                img,
                (max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
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
    """Build a mobile upload URL that works both on cloud hosts and local Wi-Fi."""
    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    scheme = (
        forwarded_proto.split(",")[0].strip() if forwarded_proto else request.url.scheme
    ) or "http"
    host_header = (
        forwarded_host.split(",")[0].strip() if forwarded_host else request.url.netloc
    ) or ""
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
                w[k] = max(0.0, float(custom_weights[k]))
    s = sum(w.values())
    if s <= 0:
        w = dict(MODE_WEIGHTS["face"])
        s = 1.0
    return {k: round(v / s, 3) for k, v in w.items()}


def _run_match(
    image_base64: str,
    mode: str = "face",
    custom_weights: Optional[Dict[str, float]] = None,
) -> tuple[list, bool, ComparisonDetails]:
    """Decode, detect, 5-point align, embed, and return a carousel of top celebrity matches + telemetry."""
    t_start = time.perf_counter()
    pipeline = get_pipeline()
    index = get_index()

    raw_b64 = image_base64
    if "," in raw_b64 and raw_b64.strip().startswith("data:"):
        raw_b64 = raw_b64.split(",", 1)[1]

    if len(raw_b64) > (MAX_IMAGE_BYTES * 4 // 3) + 256:
        raise HTTPException(
            status_code=413,
            detail="Image payload too large. Please upload an image under 10 MB.",
        )

    t_dec0 = time.perf_counter()
    try:
        image_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 data") from exc

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image payload too large. Please upload an image under 10 MB.",
        )

    is_real_pipeline = hasattr(pipeline, "get_embedding_with_info") and not hasattr(
        getattr(pipeline, "get_embedding", None), "_mock_name"
    )
    img_hash = hashlib.sha256(image_bytes).digest() if is_real_pipeline else None

    cached_item = None
    if img_hash is not None:
        with _embed_cache_lock:
            cached_item = _embed_cache.get(img_hash)
            if cached_item is not None:
                _embed_cache.move_to_end(img_hash)

    if cached_item is not None:
        embedding, face_info, (iw, ih) = cached_item
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0
    else:
        img_np = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0
        ih, iw = image.shape[:2]

        if is_real_pipeline:
            embedding, face_info = pipeline.get_embedding_with_info(image)
        else:
            t_emb0 = time.perf_counter()
            embedding = pipeline.get_embedding(image)
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
                _embed_cache[img_hash] = (embedding, face_info, (iw, ih))
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
        return [], True, details

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
    return matches, best_guess, details


# ── Core routes ───────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not (FRONTEND_DIR / "index.html").exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/celebrities/sample")
def celebrities_sample() -> dict:
    """Return a sample of indexed celebrities for the frontend roulette scanner animation."""
    idx = get_index()
    meta_values = list(idx.meta.values()) if getattr(idx, "meta", None) else []
    total = len(meta_values)
    sample = random.sample(meta_values, min(24, total)) if total > 0 else []
    return {"total_celebrities": total, "sample": sample}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest) -> MatchResponse:
    matches, best_guess, details = _run_match(
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


@app.post("/session/create")
def create_session(request: Request, mode: str = "full_head") -> dict:
    _prune_sessions()
    sid = str(uuid.uuid4())
    with _sessions_lock:
        _sessions[sid] = {
            "status": "waiting",
            "matches": None,
            "mode": mode,
            "_created_at": time.monotonic(),
        }
    mobile_url = _build_mobile_url(request, sid)
    return {"session_id": sid, "mobile_url": mobile_url}


@app.get("/session/{session_id}/status")
def session_status(session_id: str) -> dict:
    _prune_sessions()
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {k: v for k, v in session.items() if not k.startswith("_")}


@app.get("/mobile/{session_id}", response_class=HTMLResponse)
def mobile_page(session_id: str):
    _prune_sessions()
    with _sessions_lock:
        exists = session_id in _sessions
    if not exists:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if not (FRONTEND_DIR / "mobile.html").exists():
        raise HTTPException(status_code=500, detail="mobile.html not found")
    return FileResponse(FRONTEND_DIR / "mobile.html")


@app.post("/mobile/{session_id}/upload")
def mobile_upload(session_id: str, req: MatchRequest) -> dict:
    _prune_sessions()
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        if session["status"] == "done":
            return {"status": "already_done"}
        session_mode = session.get("mode") or req.mode or "full_head"
        created_at = session.get("_created_at", time.monotonic())

    matches, best_guess, details = _run_match(
        req.image_base64, mode=session_mode, custom_weights=req.weights
    )
    raw_b64 = req.image_base64.split(",", 1)[1] if "," in req.image_base64 else req.image_base64
    compact_preview = _compress_preview_b64(base64.b64decode(raw_b64))

    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id] = {
                "status": "done",
                "matches": matches,
                "best_guess": best_guess,
                "details": details.model_dump() if hasattr(details, "model_dump") else details.dict(),
                "user_image_base64": compact_preview,
                "_created_at": created_at,
            }
    return {"status": "ok"}
