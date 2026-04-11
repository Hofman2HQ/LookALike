from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .models import MatchRequest, MatchResponse, Match
from .face import get_pipeline
from .faiss_index import get_index
from datetime import datetime, timezone
import base64
import numpy as np
import cv2
import uuid
import socket

app = FastAPI(title="LookAlike API")

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = BASE_DIR / "static"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory store for phone-upload sessions
_sessions: dict = {}


def _local_ip() -> str:
    """Return the machine's LAN IP so phones on the same network can connect.

    Tries two methods:
    1. Route-based: open a UDP socket toward a public address and read which
       local address the OS selected — reliable on most machines.
    2. Hostname resolution fallback: scan the addresses returned for the
       hostname and return the first private-range one (192.168.x, 10.x,
       172.16-31.x).
    Returns "127.0.0.1" only when both methods fail.
    """
    # Method 1: let the OS pick the outbound interface
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass

    # Method 2: enumerate addresses for this hostname, prefer private ranges
    try:
        for _fam, _type, _proto, _canon, (ip, _port) in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET
        ):
            if ip.startswith(("192.168.", "10.", "172.")):
                return ip
    except Exception:
        pass

    return "127.0.0.1"


def _run_match(image_base64: str) -> list:
    """Decode, detect, embed, and search. Raises HTTPException on bad input."""
    pipeline = get_pipeline()
    index = get_index()

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 data") from exc

    img_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    embedding = pipeline.get_embedding(image)
    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected. Please upload a clear, front-facing photo.",
        )
    # ArcFace same-person threshold ≈ 0.28–0.35 (cosine similarity).
    # We use 0.35 so only meaningful lookalike matches are returned.
    matches = index.search(embedding, top_k=10, score_threshold=0.35)
    if matches:
        return matches, False
    # Nothing cleared the threshold — return the single closest match as a best guess
    closest = index.search(embedding, top_k=1, score_threshold=0.0)
    return closest, True


# ── Core routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not (FRONTEND_DIR / "index.html").exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest) -> MatchResponse:
    matches, best_guess = _run_match(req.image_base64)
    return MatchResponse(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        matches=[Match(**m) for m in matches],
        best_guess=best_guess,
    )


# ── Phone-upload via QR code ──────────────────────────────────────────────

@app.post("/session/create")
def create_session(request: Request) -> dict:
    sid = str(uuid.uuid4())
    _sessions[sid] = {"status": "waiting", "matches": None}
    port = request.url.port or 80
    mobile_url = f"http://{_local_ip()}:{port}/mobile/{sid}"
    return {"session_id": sid, "mobile_url": mobile_url}


@app.get("/session/{session_id}/status")
def session_status(session_id: str) -> dict:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/mobile/{session_id}", response_class=HTMLResponse)
def mobile_page(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if not (FRONTEND_DIR / "mobile.html").exists():
        raise HTTPException(status_code=500, detail="mobile.html not found")
    return FileResponse(FRONTEND_DIR / "mobile.html")


@app.post("/mobile/{session_id}/upload")
def mobile_upload(session_id: str, req: MatchRequest) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if _sessions[session_id]["status"] == "done":
        return {"status": "already_done"}
    matches, best_guess = _run_match(req.image_base64)
    _sessions[session_id] = {"status": "done", "matches": matches, "best_guess": best_guess}
    return {"status": "ok"}
