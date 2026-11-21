---
title: LookAlike
emoji: ✨
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
---

# LookAlike

AI-powered celebrity lookalike finder built with **FastAPI**, **OpenCV YuNet** (5-point facial landmark alignment), **InsightFace ArcFace** (`w600k_r50.onnx`, 512-D embeddings), and **FAISS**.

## Features

- **5-Point Landmark-Aligned Face Recognition:** Uses OpenCV's YuNet detector to extract 5 facial keypoints (eyes, nose, mouth corners), aligns faces via an Umeyama similarity transform to the canonical `112×112` ArcFace template, and aggregates horizontally flipped embeddings for pose and lighting invariance.
- **Multi-Celebrity Interactive Carousel:** Returns the top distinct celebrity lookalikes in an interactive carousel with calibrated resemblance scores, quick-jump thumbnail pills, keyboard navigation, and touch-swipe support.
- **Pre-Built Celebrity Index Included:** Ships with pre-computed FAISS vectors (`data/celebs.faiss`), metadata (`data/celebs_meta.json`), and compressed representative thumbnails (`static/`) for ~990 modern celebrities (`tonyassi/celebrity-1000`), so deployments and users **never need to download the training dataset or run the model over dataset photos locally**.
- **QR Code Mobile Selfie Upload:** Scan a QR code from desktop to snap and upload a selfie directly from your phone (works both on deployed HTTPS cloud hosts and local Wi-Fi).

---

## Quick Start (No Dataset Download Required)

Because the pre-computed FAISS index (`data/celebs.faiss`, `data/celebs_meta.json`) and celebrity thumbnails (`static/`) are bundled in the repository, you can run or deploy the app immediately:

### Option 1: Run with Docker Compose

```bash
docker compose up --build
```

Open `http://localhost:8000` in your browser.

### Option 2: Run Locally with Python

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

> Face detection (`face_detection_yunet_2023mar.onnx`) and recognition (`arcface_w600k_r50.onnx`) weights are downloaded and cached automatically in `backend/models/` on first start.

---

## Cloud Deployment (Users Only Visit a Web URL)

### Deploy to HuggingFace Spaces (Free, Recommended)
1. Create a new Space on [HuggingFace Spaces](https://huggingface.co/new-space) and select **Docker** as the Space SDK.
2. Push this repository (including `data/celebs.faiss`, `data/celebs_meta.json`, and `static/`) to the Space's Git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. The YAML header in `README.md` and `Dockerfile` automatically configure port `8000` and pre-cache the ONNX models at build time.

### Deploy to Render
1. Push this repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New + → Blueprint** (or **Web Service**) and connect your repository.
3. Render automatically detects `render.yaml` and `Dockerfile`, builds the container with the pre-built index and models, and gives you a public `https://<your-app>.onrender.com` URL.

---

## Rebuilding the Index (Optional)

You only need these steps if you want to customize or rebuild the celebrity index from scratch:

### 1. Download the Modern Celebrity Dataset (`tonyassi/celebrity-1000`)

```bash
python -m backend.scripts.download_celeba --root data/celebs --dataset celebrity-1000 --max_per_celeb 6
```

*(Pass `--dataset lfw` if you prefer the legacy LFW dataset.)*

### 2. Build the FAISS Index & Thumbnails

```bash
python -m backend.scripts.build_vectors data/celebs --min_images 2
```

This generates:
- `data/celebs.faiss` — L2-normalised 512-D ArcFace centroid index with medoid outlier rejection
- `data/celebs_meta.json` — celebrity name and thumbnail URL metadata
- `static/` — compressed `256×256` representative portrait thumbnails

---

## Running Tests

```bash
pytest
```
