"""Face detection and tracking."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Face:
    track_id: int
    bbox: List[int]
    landmarks: Optional[List[tuple[float, float]]] = field(default_factory=list)


class FaceTracker:
    """Very simple tracker that assigns incremental IDs to detections."""

    def __init__(self, detector):
        self.detector = detector
        self._next_id = 0

    def detect_and_track(self, frame_bgr: np.ndarray) -> List[Face]:
        detections = self.detector(frame_bgr)
        faces: List[Face] = []
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            lmk = det.get("landmarks", [])
            faces.append(Face(track_id=self._next_id, bbox=bbox, landmarks=lmk))
            self._next_id += 1
        return faces
