FROM python:3.11-slim

WORKDIR /app

# System libraries required by faiss-cpu, onnxruntime, and opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend, frontend, pre-built FAISS index, and celebrity thumbnails
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser frontend ./frontend
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser static ./static

USER appuser

# Pre-download YuNet & ArcFace ONNX weights at build time as non-root appuser
RUN python -c "from backend.app.face import get_pipeline; get_pipeline()"

ENV PORT=8000
EXPOSE 8000 7860

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
