"""Build a FAISS index and metadata from a celebrity face dataset.

Expected dataset layout (e.g. celebrity-1000, LFW, VGGFace2):
    <root>/
        Celebrity_Name/
            image1.jpg
            image2.jpg
            ...

One index entry is created *per celebrity* using an outlier-filtered mean of
their 5-point landmark-aligned ArcFace embeddings (re-normalised to unit length).
The photo closest to the cleaned centroid is saved as an optimised 256x256 JPEG
thumbnail in <static_dir> so the web UI can serve it rapidly.

Usage (run from the project root):
    python -m backend.scripts.build_vectors data/celebs \\
        [--out_index  data/celebs.faiss] \\
        [--out_meta   data/celebs_meta.json] \\
        [--static_dir static/] \\
        [--min_images 2]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import faiss
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # graceful fallback
        return it

from ..app.face import get_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def _collect_images(celeb_dir: Path) -> List[Path]:
    images: List[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        images.extend(celeb_dir.glob(pattern))
    return sorted(set(images))


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


def _clean_centroid(embeddings: List[np.ndarray]) -> Tuple[Optional[np.ndarray], int]:
    """Compute an outlier-rejected L2-normalised centroid and best representative index.

    When a celebrity has >= 3 embeddings, finds the medoid (embedding with highest
    average pairwise cosine similarity) and rejects any outlier embeddings (e.g.
    bystander faces or mislabelled photos with similarity < 0.28 to the medoid).
    """
    if not embeddings:
        return None, 0
    if len(embeddings) == 1:
        emb = embeddings[0].astype("float32")
        norm = float(np.linalg.norm(emb))
        return (emb / norm if norm > 0 else None), 0

    mat = np.stack(embeddings).astype("float32")
    if len(embeddings) >= 3:
        sim_matrix = mat @ mat.T
        medoid_idx = int(np.argmax(sim_matrix.mean(axis=1)))
        medoid_sims = sim_matrix[medoid_idx]
        inlier_mask = medoid_sims >= 0.28
        if int(inlier_mask.sum()) >= 2:
            inliers = mat[inlier_mask]
        else:
            # Keep top half most consistent with the medoid
            k = max(2, (len(embeddings) + 1) // 2)
            top_indices = np.argsort(medoid_sims)[-k:]
            inliers = mat[top_indices]
    else:
        inliers = mat

    mean_emb = np.mean(inliers, axis=0).astype("float32")
    mean_norm = float(np.linalg.norm(mean_emb))
    if mean_norm == 0.0:
        return None, 0
    mean_emb /= mean_norm

    sims = [float(np.dot(e, mean_emb)) for e in embeddings]
    best_idx = int(np.argmax(sims))
    return mean_emb, best_idx


def _save_thumbnail(src_photo: Path, dest_photo: Path, size: int = 256) -> None:
    """Save an optimised square JPEG thumbnail for fast static serving."""
    img = cv2.imread(str(src_photo))
    if img is None:
        shutil.copy2(src_photo, dest_photo)
        return
    h, w = img.shape[:2]
    if h != size or w != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest_photo), img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])


def process_dataset(
    root: Path,
    static_dir: Path,
    min_images: int = 1,
    max_images_per_celeb: int = 8,
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
        if max_images_per_celeb > 0:
            images = images[:max_images_per_celeb]

        embeddings: List[np.ndarray] = []
        valid_paths: List[Path] = []
        if hasattr(pipeline, "_detect_crop") and hasattr(pipeline, "_embed_crops_batch"):
            crops: List[np.ndarray] = []
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                crop = pipeline._detect_crop(img)
                if crop is not None:
                    crops.append(crop)
                    valid_paths.append(img_path)
            if crops:
                raw_embs = pipeline._embed_crops_batch(crops)
                for emb in raw_embs:
                    norm = float(np.linalg.norm(emb))
                    if norm > 0:
                        embeddings.append((emb / norm).astype("float32"))
        else:
            for img_path in images:
                emb = _embed_image(pipeline, img_path)
                if emb is not None:
                    embeddings.append(emb)
                    valid_paths.append(img_path)

        if not embeddings:
            log.warning("No valid face embeddings for %s — skipping", celeb_dir.name)
            skipped += 1
            continue

        mean_emb, best_idx = _clean_centroid(embeddings)
        if mean_emb is None:
            skipped += 1
            continue

        src_photo = valid_paths[best_idx]

        # Copy/compress representative photo into static directory
        dest_dir = static_dir / celeb_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_photo = dest_dir / src_photo.name
        if not dest_photo.exists():
            _save_thumbnail(src_photo, dest_photo)

        idx = len(vectors)
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
    faiss.normalize_L2(vectors_np)
    index = faiss.IndexFlatIP(vectors_np.shape[1])
    index.add(vectors_np)

    log.info("Index built: %d celebrities, embedding dim %d", len(vectors), vectors_np.shape[1])
    return index, meta


def build_head_index_from_meta(
    meta_path: str = "data/celebs_meta.json",
    out_head_path: str = "data/celebs_head.npz",
    project_root: Path = Path("."),
) -> None:
    """Compute Full-Head (hair color, hair style, head vibe) vectors & traits for all indexed celebrities."""
    pipeline = get_pipeline()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n = len(meta)
    color_vecs = []
    style_vecs = []
    vibe_vecs = []

    for i in tqdm(range(n), desc="Extracting full-head & hair features"):
        entry = meta[str(i)]
        rel_url = entry["photo_url"].lstrip("/")
        img_path = project_root / rel_url
        img = cv2.imread(str(img_path))
        _, det_meta = pipeline._detect_crop_with_meta(img) if img is not None else (None, {})
        hf = det_meta.get("head_features")
        if hf is None and img is not None:
            from ..app.face import _extract_full_head_features
            hf = _extract_full_head_features(img)
        color_vecs.append(hf["hair_color_vec"])
        style_vecs.append(hf["hair_style_vec"])
        vibe_vecs.append(hf["head_vibe_vec"])
        entry["traits"] = hf["traits"]

    np.savez_compressed(
        out_head_path,
        hair_color_vecs=np.stack(color_vecs).astype(np.float32),
        hair_style_vecs=np.stack(style_vecs).astype(np.float32),
        head_vibe_vecs=np.stack(vibe_vecs).astype(np.float32),
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Saved full-head index (%d celebs) → %s", n, out_head_path)


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
    parser.add_argument("--max_per_celeb", type=int, default=6,
                        help="Max images to embed per celebrity (default: 6)")
    args = parser.parse_args()

    index, meta = process_dataset(
        args.root,
        args.static_dir,
        args.min_images,
        args.max_per_celeb,
    )

    os.makedirs(Path(args.out_index).parent, exist_ok=True)
    faiss.write_index(index, args.out_index)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Saved → %s", args.out_index)
    log.info("Saved → %s", args.out_meta)
    build_head_index_from_meta(args.out_meta)


if __name__ == "__main__":
    main()
