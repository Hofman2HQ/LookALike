from functools import lru_cache
import os
from typing import List
import faiss
import numpy as np
import json

class FaissIndex:
    def __init__(self, index_path: str, meta_path: str):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            # create an empty index for development/testing
            self.index = faiss.IndexFlatIP(512)
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
            celeb = self.meta[str(idx)]
            matches.append({
                "name": celeb["name"],
                "score": float(score),
                "photo_url": celeb["photo_url"]
            })
        return matches

@lru_cache()
def get_index(index_path: str = "data/celebs.faiss", meta_path: str = "data/celebs_meta.json") -> FaissIndex:
    return FaissIndex(index_path, meta_path)

