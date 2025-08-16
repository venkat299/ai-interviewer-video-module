"""Face preprocessing utilities."""
from __future__ import annotations
import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_face(frame_bgr: np.ndarray, bbox: list[int], landmarks: Optional[list[tuple[float, float]]] = None,
                    size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Crop and normalize face region."""
    x, y, w, h = bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_bgr.shape[1], x + w), min(frame_bgr.shape[0], y + h)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    crop = cv2.resize(crop, size)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return rgb
