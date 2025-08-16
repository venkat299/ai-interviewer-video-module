"""Wrapper for Deepface detector backends."""
from __future__ import annotations
from typing import Callable, List

try:
    from deepface import DeepFace
except Exception:  # pragma: no cover - optional
    DeepFace = None

_cached: dict[str, Callable] = {}


def load_detector(name: str) -> Callable:
    """Load and cache deepface detector backend."""
    if name in _cached:
        return _cached[name]

    if DeepFace is None:
        def dummy(frame):
            h, w = frame.shape[:2]
            return [{"bbox": [0, 0, w, h], "landmarks": []}]
        _cached[name] = dummy
        return dummy

    def _detect(frame):
        objs = DeepFace.extract_faces(frame, detector_backend=name, enforce_detection=False)
        out: List[dict] = []
        for obj in objs:
            box = obj["facial_area"]
            out.append({"bbox": [box["x"], box["y"], box["w"], box["h"]], "landmarks": obj.get("landmarks", [])})
        return out

    _cached[name] = _detect
    return _detect
