from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .models import MatchRequest, MatchResponse, Match
from .face import get_pipeline
from .faiss_index import get_index
from datetime import datetime
import base64
import numpy as np
import cv2
import uuid

app = FastAPI(title="LookAlike API")

BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = BASE_DIR / "static"

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
    pipeline = get_pipeline()
    index = get_index()

    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 data") from exc

    img_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    face = pipeline.detect_and_align(image)
    embedding = pipeline.embed(face).astype('float32')
    matches = index.search(embedding, top_k=3)
    match_objs = [Match(**m) for m in matches]
    return MatchResponse(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        matches=match_objs
    )

