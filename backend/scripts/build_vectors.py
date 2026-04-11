"""Build a FAISS index and metadata from a celebrity face dataset.

Expected dataset layout (e.g. LFW, VGGFace2):
    <root>/
        Celebrity_Name/
            image1.jpg
            image2.jpg
            ...

One index entry is created *per celebrity* using the mean of all their valid
face embeddings (re-normalised), which avoids flooding search results with
dozens of entries for the same person.  The photo that best represents the
celebrity (closest to the mean embedding) is copied to <static_dir> so the
frontend can display it.

Usage (run from the project root):
    python -m backend.scripts.build_vectors data/lfw \\
        [--out_index  data/celebs.faiss] \\
        [--out_meta   data/celebs_meta.json] \\
        [--static_dir static/] \\
        [--min_images 2]
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import faiss

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # graceful fallback when tqdm is not installed
        return it

from ..app.face import get_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def _collect_images(celeb_dir: Path) -> List[Path]:
    images: List[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        images.extend(celeb_dir.glob(pattern))
    return sorted(images)  # sorted for reproducibility


def _embed_image(pipeline, img_path: Path) -> Optional[np.ndarray]:
    """Return a unit-norm float32 embedding, or None on any failure."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    try:
        return pipeline.get_embedding(img)
    except Exception as exc:
        log.debug("Embedding failed for %s: %s", img_path, exc)
        return None


def process_dataset(
    root: Path,
    static_dir: Path,
    min_images: int = 1,
) -> tuple:
    pipeline = get_pipeline()

    celeb_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not celeb_dirs:
        raise ValueError(f"No subdirectories found in {root}")

    vectors: List[np.ndarray] = []
    meta: dict = {}
    skipped = 0

    for celeb_dir in tqdm(celeb_dirs, desc="Processing celebrities"):
        images = _collect_images(celeb_dir)
        if len(images) < min_images:
            skipped += 1
            continue

        embeddings: List[np.ndarray] = []
        valid_paths: List[Path] = []
        for img_path in images:
            emb = _embed_image(pipeline, img_path)
            if emb is not None:
                embeddings.append(emb)
                valid_paths.append(img_path)

        if not embeddings:
            log.warning("No valid face embeddings for %s — skipping", celeb_dir.name)
            skipped += 1
            continue

        # Mean of all valid embeddings, re-normalised to unit length
        mean_emb = np.mean(embeddings, axis=0).astype("float32")
        mean_norm = np.linalg.norm(mean_emb)
        if mean_norm == 0:
            skipped += 1
            continue
        mean_emb /= mean_norm

        # Best representative photo: the image whose embedding is nearest to the mean
        sims = [float(np.dot(e, mean_emb)) for e in embeddings]
        best_idx = int(np.argmax(sims))
        src_photo = valid_paths[best_idx]

        # Copy the representative photo into the static directory
        dest_dir = static_dir / celeb_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_photo = dest_dir / src_photo.name
        if not dest_photo.exists():
            shutil.copy2(src_photo, dest_photo)

        idx = len(vectors)
        # Celebrity name: underscores → spaces (matches LFW naming convention)
        name = celeb_dir.name.replace("_", " ")
        photo_url = f"/static/{celeb_dir.name}/{src_photo.name}"
        meta[str(idx)] = {"name": name, "photo_url": photo_url}
        vectors.append(mean_emb)

    if not vectors:
        raise ValueError(
            "No celebrities with valid face embeddings were found. "
            "Check that the dataset root contains subdirectories of face images."
        )

    if skipped:
        log.info("Skipped %d celebrities (too few images or no detectable faces)", skipped)

    vectors_np = np.stack(vectors)
    # Normalise again after stacking to guard against floating-point drift
    faiss.normalize_L2(vectors_np)
    index = faiss.IndexFlatIP(vectors_np.shape[1])
    index.add(vectors_np)

    log.info("Index built: %d celebrities, embedding dim %d", len(vectors), vectors_np.shape[1])
    return index, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Dataset root directory")
    parser.add_argument("--out_index", default="data/celebs.faiss",
                        help="Output path for the FAISS index")
    parser.add_argument("--out_meta", default="data/celebs_meta.json",
                        help="Output path for the metadata JSON")
    parser.add_argument("--static_dir", type=Path, default=Path("static"),
                        help="Directory where representative photos are copied (served at /static)")
    parser.add_argument("--min_images", type=int, default=1,
                        help="Skip celebrities with fewer than this many images (default: 1)")
    args = parser.parse_args()

    index, meta = process_dataset(args.root, args.static_dir, args.min_images)

    os.makedirs(Path(args.out_index).parent, exist_ok=True)
    faiss.write_index(index, args.out_index)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Saved → %s", args.out_index)
    log.info("Saved → %s", args.out_meta)


if __name__ == "__main__":
    main()
