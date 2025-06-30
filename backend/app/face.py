from functools import lru_cache
from typing import Tuple
import numpy as np

class FacePipeline:
    def __init__(self):
        # TODO: load models lazily
        self.detector = None
        self.embedder = None

    def detect_and_align(self, image: np.ndarray) -> np.ndarray:
        # TODO: implement face detection and alignment
        return image

    def embed(self, face: np.ndarray) -> np.ndarray:
        # TODO: implement embedding extraction
        return np.zeros(512, dtype=np.float32)

@lru_cache()
def get_pipeline() -> FacePipeline:
    return FacePipeline()

