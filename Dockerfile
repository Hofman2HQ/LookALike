FROM python:3.11-slim

WORKDIR /app

# System libraries required by faiss-cpu, onnxruntime, and opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend, frontend, pre-built FAISS index, and celebrity thumbnails
COPY backend ./backend
COPY frontend ./frontend
COPY data ./data
COPY static ./static

# Pre-download YuNet & ArcFace ONNX weights at build time so cold starts are instant
RUN python -c "from backend.app.face import get_pipeline; get_pipeline()"

ENV PORT=8000
EXPOSE 8000 7860

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
