# LookAlike

Prototype backend for the LookAlike project. It exposes a small FastAPI service
with `/health` and `/match` endpoints.  The default pipeline is intentionally
simple so that the service can run in minimal environments without GPU support
or heavyweight ML dependencies.

## Development

Build and run using Docker Compose:

```bash
docker compose up --build
```

After starting the service you can open `http://localhost:8000/` in your
browser. A small web UI lets you upload or capture a selfie and view the top
celebrity matches returned by the API.

By default the service will look for a FAISS index in `data/celebs.faiss` with
its accompanying metadata JSON.  You can build this index from a folder of
celebrity images using the provided script:

```bash
pip install -r requirements.txt
python -m backend.scripts.build_vectors <path_to_celeb_dataset>
```

The command above will create `data/celebs.faiss` and `data/celebs_meta.json`
which the API uses for matching.
The images themselves should be available under a `static/` directory so the
web UI can display them. If your dataset lives elsewhere you can symlink or copy
it into `static` before starting the service.

### Downloading the CelebA dataset

If you don't already have a celebrity image dataset you can download the
[CelebA dataset](https://github.com/mireshghallah/CelebA) using the helper
script in this repository:

```bash
pip install -r requirements.txt
python -m backend.scripts.download_celeba
python -m backend.scripts.build_vectors data/celeba
```

The script downloads the images and identity labels and extracts them into
`data/celeba/`. The vector builder then generates the FAISS index and metadata
files which the API will pick up on the next start.

### Using your own dataset

If you would like to use a custom dataset, place your images under separate
folders (one folder per person) and run `backend.scripts.build_vectors` pointing
to that directory as shown above. Ensure the images are accessible under the
`static/` directory so the frontend can display them.

Run tests:

```bash
pip install -r requirements.txt
pytest
```

