from fastapi import FastAPI
from .models import MatchRequest, MatchResponse, Match
from .face import get_pipeline
from .faiss_index import get_index
from datetime import datetime
import base64
import numpy as np
import uuid

app = FastAPI(title="LookAlike API")

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest) -> MatchResponse:
    pipeline = get_pipeline()
    index = get_index()

    image_bytes = base64.b64decode(req.image_base64)
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    face = pipeline.detect_and_align(image)
    embedding = pipeline.embed(face).astype('float32')
    matches = index.search(embedding, top_k=3)
    match_objs = [Match(**m) for m in matches]
    return MatchResponse(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        matches=match_objs
    )

