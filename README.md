# LookAlike

Prototype backend for the LookAlike project. It exposes a small FastAPI service
with `/health` and `/match` endpoints.  The default pipeline is intentionally
simple so that the service can run in minimal environments without GPU support
or heavyweight ML dependencies.

## Setup

> **Note:** The dataset, FAISS index, static images, and model weights are **not included** in
> this repository.  Follow the steps below to set them up before running the service.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

The project uses the [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/) dataset,
pulled automatically from HuggingFace Hub (~188 MB, no account required):

```bash
python -m backend.scripts.download_celeba
```

Images are saved to `data/lfw/` in the layout that `build_vectors.py` expects.

### 3. Build the FAISS index

```bash
python -m backend.scripts.build_vectors data/lfw --min_images 2
```

This creates:
- `data/celebs.faiss` — the vector index used by the `/match` endpoint
- `data/celebs_meta.json` — name and image-path metadata
- `static/` — one representative image per person, served by the web UI

### 4. Model weights

Face detection and recognition models are **downloaded automatically** on first run —
no manual step is needed.  They are cached in `backend/models/`.

### 5. Run the service

```bash
docker compose up --build
```

Or without Docker:

```bash
uvicorn backend.app.main:app --reload
```

Open `http://localhost:8000/` in your browser.  The web UI lets you upload or
capture a selfie and view the top matches returned by the API.

### Using your own dataset

Place images under separate folders (one folder per person) and run
`build_vectors` pointing to that directory:

```bash
python -m backend.scripts.build_vectors <path_to_dataset> --min_images 1
```

Run tests:

```bash
pip install -r requirements.txt
pytest
```

