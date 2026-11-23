import math
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field, field_validator

AllowedMode = Literal["face", "full_head", "hair_twin", "custom"]
_ALLOWED_WEIGHT_KEYS = {"face", "hair_style", "hair_color", "head_vibe"}


class MatchRequest(BaseModel):
    image_base64: str = Field(..., min_length=16, max_length=7_000_000)
    mode: AllowedMode = "face"
    weights: Optional[Dict[str, float]] = None

    @field_validator("weights")
    @classmethod
    def validate_weights(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        if v is None:
            return None
        cleaned: Dict[str, float] = {}
        for k, val in v.items():
            if k not in _ALLOWED_WEIGHT_KEYS:
                raise ValueError(f"Unsupported weight key: {k}")
            fval = float(val)
            if not math.isfinite(fval) or fval < 0.0 or fval > 1000.0:
                raise ValueError(f"Weight for {k} must be a finite number in [0, 1000]")
            cleaned[k] = fval
        return cleaned


class Match(BaseModel):
    name: str
    score: float
    photo_url: str
    face_score: Optional[float] = None
    hair_style_score: Optional[float] = None
    hair_color_score: Optional[float] = None
    head_vibe_score: Optional[float] = None
    traits: Optional[Dict[str, str]] = None
    angular_distance_deg: Optional[float] = None
    l2_distance: Optional[float] = None
    percentile: Optional[float] = None


class ComparisonDetails(BaseModel):
    total_ms: float
    decode_ms: float
    detect_align_ms: float
    embed_ms: float
    search_ms: float
    celebrities_compared: int
    comparison_rate_per_sec: float
    vector_search_rate_per_sec: float
    embedding_dim: int = 512
    detector: str = "YuNet 5-Point Landmark"
    model: str = "ArcFace ResNet-50 (w600k_r50, 512-D)"
    face_confidence: float = 0.0
    face_bbox: Optional[List[float]] = None
    landmarks: Optional[List[List[float]]] = None
    image_resolution: Optional[List[int]] = None
    mode: AllowedMode = "face"
    weights_used: Optional[Dict[str, float]] = None
    subject_traits: Optional[Dict[str, str]] = None


class MatchResponse(BaseModel):
    query_id: str
    timestamp: str
    matches: List[Match]
    best_guess: bool = False  # True when no match cleared the threshold
    details: Optional[ComparisonDetails] = None
