from functools import lru_cache
import json
import os
from pathlib import Path
import threading
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

from .face import get_pipeline

# Project root is two levels above this file (backend/app/faiss_index.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INDEX = str(_PROJECT_ROOT / "data" / "celebs.faiss")
_DEFAULT_META = str(_PROJECT_ROOT / "data" / "celebs_meta.json")
_DEFAULT_HEAD = str(_PROJECT_ROOT / "data" / "celebs_head.npz")

MODE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "face": {
        "face": 1.0,
        "hair_style": 0.0,
        "hair_color": 0.0,
        "head_vibe": 0.0,
    },
    "full_head": {
        "face": 0.46,
        "hair_style": 0.22,
        "hair_color": 0.20,
        "head_vibe": 0.12,
    },
    "hair_twin": {
        "face": 0.20,
        "hair_style": 0.40,
        "hair_color": 0.30,
        "head_vibe": 0.10,
    },
}


def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size <= 1:
        return np.zeros_like(arr)
    std = float(arr.std())
    if std < 1e-7:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def _pct_from_z(z: float, raw_cos: float) -> float:
    """Convert a feature z-score & raw cosine similarity into an intuitive 35..99% sub-score."""
    val = 68.0 + z * 11.5 + (raw_cos - 0.5) * 18.0
    return round(float(max(32.0, min(99.0, val))), 1)


class FaissIndex:
    def __init__(
        self,
        index_path: str,
        meta_path: str,
        head_path: str = _DEFAULT_HEAD,
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.head_path = head_path
        self._lock = threading.Lock()
        self._mtime: float = 0.0
        self.index = None
        self.meta: Dict[str, Any] = {}
        self.csls_baseline: Optional[np.ndarray] = None
        self.hair_color_vecs: Optional[np.ndarray] = None
        self.hair_style_vecs: Optional[np.ndarray] = None
        self.head_vibe_vecs: Optional[np.ndarray] = None
        self._load()

    def _current_max_mtime(self) -> float:
        mtimes = [
            os.path.getmtime(p)
            for p in (self.index_path, self.meta_path, self.head_path)
            if os.path.exists(p)
        ]
        return max(mtimes) if mtimes else 0.0

    def _load(self) -> None:
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            dim = get_pipeline().embedding_size
            self.index = faiss.IndexFlatIP(dim)

        # Precompute CSLS (Cross-Domain Similarity Local Scaling) hubness baseline
        # so "hub" centroids near the global face mean do not unfairly dominate top-K matches.
        self.csls_baseline = None
        if self.index is not None and self.index.ntotal > 12:
            try:
                ntotal = int(self.index.ntotal)
                all_vecs = self.index.reconstruct_n(0, ntotal)
                k_nbr = min(11, ntotal)
                nbr_scores, _ = self.index.search(all_vecs, k_nbr)
                r_t = nbr_scores[:, 1:].mean(axis=1).astype(np.float32)
                self.csls_baseline = r_t - float(r_t.mean())
            except Exception:
                self.csls_baseline = None

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

        if os.path.exists(self.head_path):
            try:
                with np.load(self.head_path) as data:
                    self.hair_color_vecs = data["hair_color_vecs"].astype(np.float32)
                    self.hair_style_vecs = data["hair_style_vecs"].astype(np.float32)
                    self.head_vibe_vecs = data["head_vibe_vecs"].astype(np.float32)
            except Exception:
                self.hair_color_vecs = None
                self.hair_style_vecs = None
                self.head_vibe_vecs = None
        else:
            self.hair_color_vecs = None
            self.hair_style_vecs = None
            self.head_vibe_vecs = None

        self._mtime = self._current_max_mtime()

    def _maybe_reload(self) -> None:
        cur = self._current_max_mtime()
        if cur != self._mtime:
            with self._lock:
                if self._current_max_mtime() != self._mtime:
                    self._load()

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 8,
        score_threshold: float = -1.0,
        mode: str = "face",
        head_features: Optional[Dict[str, Any]] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        self._maybe_reload()
        if self.index is None or self.index.ntotal == 0:
            return []

        ntotal = int(self.index.ntotal)
        q = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)

        has_head_index = (
            head_features is not None
            and self.hair_color_vecs is not None
            and self.hair_style_vecs is not None
            and self.head_vibe_vecs is not None
            and len(self.hair_color_vecs) == ntotal
        )

        # Resolve mode weights
        weights = dict(MODE_WEIGHTS.get(mode, MODE_WEIGHTS["face"]))
        if custom_weights:
            for k in ("face", "hair_style", "hair_color", "head_vibe"):
                if k in custom_weights and custom_weights[k] is not None:
                    weights[k] = max(0.0, float(custom_weights[k]))
        w_sum = sum(weights.values())
        if w_sum <= 0:
            weights = dict(MODE_WEIGHTS["face"])
            w_sum = 1.0
        for k in weights:
            weights[k] /= w_sum

        if not has_head_index:
            query_k = min(max(top_k * 3, 15), ntotal)
            scores, indices = self.index.search(q, query_k)
            matches: List[Dict[str, Any]] = []
            seen_names = set()
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or float(score) < score_threshold:
                    continue
                celeb = self.meta.get(str(idx))
                if celeb is None:
                    continue
                name = celeb["name"]
                norm_name = name.strip().lower()
                if norm_name in seen_names:
                    continue
                seen_names.add(norm_name)
                matches.append(
                    {
                        "name": name,
                        "score": float(score),
                        "face_score": float(score),
                        "photo_url": celeb["photo_url"],
                        "traits": celeb.get("traits"),
                    }
                )
                if len(matches) >= top_k:
                    break
            return matches

        # Full multi-modal scoring across all N celebrities
        raw_face_scores, raw_face_indices = self.index.search(q, ntotal)
        raw_face_sims = np.zeros(ntotal, dtype=np.float32)
        for s, idx in zip(raw_face_scores[0], raw_face_indices[0]):
            if 0 <= idx < ntotal:
                raw_face_sims[idx] = float(s)

        # Apply CSLS hubness correction to penalize generic "hub" centroids
        if self.csls_baseline is not None and len(self.csls_baseline) == ntotal:
            face_sims = raw_face_sims - 0.35 * self.csls_baseline
        else:
            face_sims = raw_face_sims

        q_color = head_features["hair_color_vec"].astype(np.float32)
        q_style = head_features["hair_style_vec"].astype(np.float32)
        q_vibe = head_features["head_vibe_vec"].astype(np.float32)

        color_sims = self.hair_color_vecs @ q_color
        style_sims = self.hair_style_vecs @ q_style
        vibe_sims = self.head_vibe_vecs @ q_vibe

        z_face = _zscore(face_sims)
        z_style = _zscore(style_sims)
        z_color = _zscore(color_sims)
        z_vibe = _zscore(vibe_sims)

        # Soft facial plausibility penalty so full-head/hair matches still share plausible facial geometry
        z_face_eff = (
            np.where(z_face < 0.0, z_face * 1.65, z_face)
            if weights["face"] >= 0.15
            else z_face
        )

        raw_combined_z = (
            weights["face"] * z_face_eff
            + weights["hair_style"] * z_style
            + weights["hair_color"] * z_color
            + weights["head_vibe"] * z_vibe
        )
        combined_z = _zscore(raw_combined_z)

        face_mean = float(raw_face_sims.mean())
        face_std = max(float(raw_face_sims.std()), 0.06)
        effective_scores = np.clip(face_mean + combined_z * face_std, -1.0, 0.99)

        if weights["face"] >= 0.999:
            order = np.argsort(-face_sims)
            effective_scores = np.clip(face_sims, -1.0, 0.99)
        else:
            order = np.argsort(-combined_z)

        matches = []
        seen_names = set()
        for idx in order:
            idx_int = int(idx)
            eff_score = float(effective_scores[idx_int])
            if eff_score < score_threshold:
                continue
            celeb = self.meta.get(str(idx_int))
            if celeb is None:
                continue
            name = celeb["name"]
            norm_name = name.strip().lower()
            if norm_name in seen_names:
                continue
            seen_names.add(norm_name)

            matches.append(
                {
                    "name": name,
                    "score": eff_score,
                    "face_score": float(raw_face_sims[idx_int]),
                    "hair_style_score": _pct_from_z(
                        float(z_style[idx_int]), float(style_sims[idx_int])
                    ),
                    "hair_color_score": _pct_from_z(
                        float(z_color[idx_int]), float(color_sims[idx_int])
                    ),
                    "head_vibe_score": _pct_from_z(
                        float(z_vibe[idx_int]), float(vibe_sims[idx_int])
                    ),
                    "photo_url": celeb["photo_url"],
                    "traits": celeb.get("traits"),
                }
            )
            if len(matches) >= top_k:
                break

        return matches


@lru_cache()
def get_index(
    index_path: str = _DEFAULT_INDEX,
    meta_path: str = _DEFAULT_META,
    head_path: str = _DEFAULT_HEAD,
) -> FaissIndex:
    return FaissIndex(index_path, meta_path, head_path)
