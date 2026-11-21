"""Download a celebrity face dataset via HuggingFace Hub (no login required).

By default, downloads the curated modern celebrity dataset ``tonyassi/celebrity-1000``
(~990 modern actors, musicians, and public figures, 256x256 face portraits).
Also supports ``--dataset lfw`` (``bitmind/lfw``) for legacy LFW benchmarks.

Images are saved to disk in the directory layout that ``build_vectors.py`` expects:

    <root>/
        Person_Name/
            Person_Name_0001.jpg
            Person_Name_0002.jpg
            ...

Usage (run from the project root):
    python -m backend.scripts.download_celeba [--root data/celebs] [--max_per_celeb 5]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import unicodedata

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # graceful fallback
        return it


_CELEB_1000_DATASET = "tonyassi/celebrity-1000"
_LFW_DATASET = "bitmind/lfw"
_PARQUET_FILE = "data/train-00000-of-00001.parquet"


def _person_dir(filename: str) -> str:
    """Return the directory name for a person given their LFW filename.

    Examples:
        'Salma_Hayek_0012.jpg' -> 'Salma_Hayek'
        'George_W_Bush_0001.jpg' -> 'George_W_Bush'
    """
    stem = Path(filename).stem
    return stem.rsplit("_", 1)[0]


def _canonical_celeb_dir(raw_name: str) -> str:
    """Normalize a celebrity name into a clean, deduplicated directory name.

    Merges duplicate accented variants (e.g. 'Timothée Chalamet' and
    'Timothee Chalamet', 'Zoë Kravitz' and 'Zoe Kravitz').
    """
    ascii_name = (
        unicodedata.normalize("NFKD", raw_name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    cleaned = re.sub(r"[^\w\s-]", "", ascii_name).strip()
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    return cleaned or "Unknown"


def download_celebrities(
    root: Path = Path("data/celebs"),
    dataset_kind: str = "celebrity-1000",
    max_per_celeb: int = 5,
) -> Path:
    """Download the chosen celebrity dataset from HuggingFace Hub and save under *root*."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    if dataset_kind == "lfw":
        return download_lfw(root, max_per_celeb=max_per_celeb)

    print(f"Downloading '{_CELEB_1000_DATASET}' parquet from HuggingFace Hub …")
    parquet_path = hf_hub_download(
        repo_id=_CELEB_1000_DATASET,
        filename=_PARQUET_FILE,
        repo_type="dataset",
    )

    pf = pq.ParquetFile(parquet_path)
    hf_meta = json.loads(pf.schema_arrow.metadata[b"huggingface"].decode("utf-8"))
    label_names = hf_meta["info"]["features"]["label"]["names"]

    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    saved = skipped = 0

    for batch in tqdm(pf.iter_batches(batch_size=512), desc="Extracting celebrity images"):
        images_col = batch.column("image").to_pylist()
        labels_col = batch.column("label").to_pylist()
        for img_entry, label_idx in zip(images_col, labels_col):
            raw_name = label_names[int(label_idx)]
            person_slug = _canonical_celeb_dir(raw_name)

            if max_per_celeb > 0 and counts[person_slug] >= max_per_celeb:
                continue

            counts[person_slug] += 1
            seq = counts[person_slug]
            dest_dir = root / person_slug
            dest_dir.mkdir(exist_ok=True)

            filename = f"{person_slug}_{seq:04d}.jpg"
            dest_file = dest_dir / filename
            if dest_file.exists():
                skipped += 1
            else:
                img_bytes = img_entry["bytes"] if isinstance(img_entry, dict) else img_entry
                dest_file.write_bytes(img_bytes)
                saved += 1

    n_people = sum(1 for p in root.iterdir() if p.is_dir())
    n_images = sum(1 for _ in root.rglob("*.jpg"))
    print(f"\nDataset ready at {root}")
    print(
        f"  {n_people:,} celebrities  |  {n_images:,} images  |  "
        f"{saved} saved, {skipped} already present"
    )
    return root


def download_lfw(root: Path = Path("data/lfw"), max_per_celeb: int = 0) -> Path:
    """Download LFW from HuggingFace Hub and save it under *root*."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    print(f"Downloading '{_LFW_DATASET}' parquet from HuggingFace Hub …")
    parquet_path = hf_hub_download(
        repo_id=_LFW_DATASET,
        filename=_PARQUET_FILE,
        repo_type="dataset",
    )

    pf = pq.ParquetFile(parquet_path)
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    saved = skipped = 0

    for batch in tqdm(pf.iter_batches(batch_size=512), desc="Extracting LFW images"):
        filenames_col = batch.column("filename").to_pylist()
        images_col = batch.column("image").to_pylist()
        for filename, img_entry in zip(filenames_col, images_col):
            person = _person_dir(filename)
            if max_per_celeb > 0 and counts[person] >= max_per_celeb:
                continue
            counts[person] += 1

            dest_dir = root / person
            dest_dir.mkdir(exist_ok=True)
            dest_file = dest_dir / filename
            if dest_file.exists():
                skipped += 1
            else:
                img_bytes = img_entry["bytes"] if isinstance(img_entry, dict) else img_entry
                dest_file.write_bytes(img_bytes)
                saved += 1

    n_people = sum(1 for p in root.iterdir() if p.is_dir())
    n_images = sum(1 for _ in root.rglob("*.jpg"))
    print(f"\nDataset ready at {root}")
    print(
        f"  {n_people:,} people  |  {n_images:,} images  |  "
        f"{saved} saved, {skipped} already present"
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/celebs"),
        help="Where to save the dataset (default: data/celebs)",
    )
    parser.add_argument(
        "--dataset",
        choices=["celebrity-1000", "lfw"],
        default="celebrity-1000",
        help="Which HuggingFace celebrity dataset to download (default: celebrity-1000)",
    )
    parser.add_argument(
        "--max_per_celeb",
        type=int,
        default=5,
        help="Max images to save per celebrity (0 = unlimited, default: 5)",
    )
    args = parser.parse_args()

    try:
        download_celebrities(args.root, args.dataset, args.max_per_celeb)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
