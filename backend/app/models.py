from pydantic import BaseModel, HttpUrl
from typing import List

class MatchRequest(BaseModel):
    image_base64: str

class Match(BaseModel):
    name: str
    score: float
    photo_url: HttpUrl

class MatchResponse(BaseModel):
    query_id: str
    timestamp: str
    matches: List[Match]

