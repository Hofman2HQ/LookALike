from pydantic import BaseModel
from typing import List

class MatchRequest(BaseModel):
    image_base64: str

class Match(BaseModel):
    name: str
    score: float
    photo_url: str

class MatchResponse(BaseModel):
    query_id: str
    timestamp: str
    matches: List[Match]
    best_guess: bool = False  # True when no match cleared the threshold

