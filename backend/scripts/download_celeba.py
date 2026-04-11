"""Download LFW via the HuggingFace Hub (no login required).

Uses the ``bitmind/lfw`` mirror (~188 MB) from HuggingFace Hub.  Images are
saved to disk in the directory layout that ``build_vectors.py`` expects:

    <root>/
        Person_Name/
            Person_Name_0001.jpg
            Person_Name_0002.jpg
            ...

No Kaggle account or manual download is needed.

Usage (run from the project root):
    python -m backend.scripts.download_celeba [--root data/lfw]

After downloading, build the FAISS index with:
    python -m backend.scripts.build_vectors data/lfw --min_images 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print(
        "ERROR: 'datasets' package is not installed.\n"
        "Install it with:  pip install datasets",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):   # graceful fallback
        return it


_HF_DATASET = "bitmind/lfw"


def _person_dir(filename: str) -> str:
    """Return the directory name for a person given their LFW filename.

    LFW filenames follow the pattern ``First_Last_NNNN.jpg``.  We strip the
    trailing ``_NNNN`` sequence to recover the per-person directory name.

    Examples:
        'Salma_Hayek_0012.jpg' → 'Salma_Hayek'
        'George_W_Bush_0001.jpg' → 'George_W_Bush'
    """
    stem = Path(filename).stem        # e.g. 'Salma_Hayek_0012'
    return stem.rsplit("_", 1)[0]     # e.g. 'Salma_Hayek'


def download_lfw(root: Path = Path("data/lfw")) -> Path:
    """Download LFW from HuggingFace Hub and save it under *root*."""
    print(f"Loading '{_HF_DATASET}' from HuggingFace Hub …")
    print("(The first run will download ~188 MB; subsequent runs use the cache.)\n")

    ds = load_dataset(_HF_DATASET, split="train")

    root.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(ds):,} images to {root} …")

    saved = skipped = 0
    for item in tqdm(ds, desc="Saving images", unit="img"):
        filename: str = item["filename"]
        image = item["image"]  # PIL.Image.Image

        dest_dir = root / _person_dir(filename)
        dest_dir.mkdir(exist_ok=True)

        dest_file = dest_dir / filename
        if dest_file.exists():
            skipped += 1
        else:
            image.save(dest_file)
            saved += 1

    n_people = sum(1 for p in root.iterdir() if p.is_dir())
    n_images = sum(1 for _ in root.rglob("*.jpg"))
    print(f"\nDataset ready at {root}")
    print(f"  {n_people:,} people  |  {n_images:,} images  |  {saved} saved, {skipped} already present")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/lfw"),
        help="Where to save the dataset (default: data/lfw)",
    )
    args = parser.parse_args()

    try:
        download_lfw(args.root)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
