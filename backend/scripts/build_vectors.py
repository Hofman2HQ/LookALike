"""Build FAISS index and metadata from a celeb dataset."""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import faiss
from PIL import Image
from ..app.face import get_pipeline


def process_dataset(root: Path):
    pipeline = get_pipeline()
    vectors = []
    meta = {}
    idx = 0
    for celeb_dir in root.iterdir():
        if not celeb_dir.is_dir():
            continue
        name = celeb_dir.name
        for img_path in celeb_dir.glob("*.jpg"):
            img = np.asarray(Image.open(img_path))
            face = pipeline.detect_and_align(img)
            emb = pipeline.embed(face).astype('float32')
            vectors.append(emb)
            meta[idx] = {
                "name": name,
                "photo_url": f"/static/{name}/{img_path.name}"
            }
            idx += 1
    vectors_np = np.stack(vectors)
    index = faiss.IndexFlatIP(vectors_np.shape[1])
    faiss.normalize_L2(vectors_np)
    index.add(vectors_np)
    return index, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out_index", default="data/celebs.faiss")
    parser.add_argument("--out_meta", default="data/celebs_meta.json")
    args = parser.parse_args()

    index, meta = process_dataset(args.root)
    os.makedirs(Path(args.out_index).parent, exist_ok=True)
    faiss.write_index(index, args.out_index)
    with open(args.out_meta, 'w') as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()

