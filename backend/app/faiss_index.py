from functools import lru_cache
import os
from pathlib import Path
from typing import List
import faiss
import numpy as np
import json

from .face import get_pipeline

# Project root is two levels above this file (backend/app/faiss_index.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INDEX = str(_PROJECT_ROOT / "data" / "celebs.faiss")
_DEFAULT_META  = str(_PROJECT_ROOT / "data" / "celebs_meta.json")

class FaissIndex:
    def __init__(self, index_path: str, meta_path: str):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            # create an empty index for development/testing
            dim = get_pipeline().embedding_size
            self.index = faiss.IndexFlatIP(dim)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

    def search(self, vector: np.ndarray, top_k: int = 3, score_threshold: float = 0.0):
        scores, indices = self.index.search(vector.reshape(1, -1), top_k)
        matches = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score < score_threshold:
                continue
            celeb = self.meta.get(str(idx))
            if celeb is None:
                continue
            matches.append({
                "name": celeb["name"],
                "score": float(score),
                "photo_url": celeb["photo_url"]
            })
        return matches

@lru_cache()
def get_index(index_path: str = _DEFAULT_INDEX, meta_path: str = _DEFAULT_META) -> FaissIndex:
    return FaissIndex(index_path, meta_path)

