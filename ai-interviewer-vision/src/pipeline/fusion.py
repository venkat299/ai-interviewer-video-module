"""Score fusion and temporal smoothing."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class FusionConfig:
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    temporal_window: int = 25


class FusionHead:
    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg

    def combine(self, liveness: float, deepfake: float, track_id: int) -> float:
        w = self.cfg.weights
        return w.get("liveness", 0.5) * liveness + w.get("deepfake", 0.5) * deepfake


class TemporalWindow:
    def __init__(self, window: int):
        self.alpha = 2 / float(window + 1)
        self.state: Dict[int, float] = {}

    def update_and_get(self, track_id: int, value: float) -> float:
        prev = self.state.get(track_id, value)
        new = self.alpha * value + (1 - self.alpha) * prev
        self.state[track_id] = new
        return new


def threshold_with_hysteresis(prob: float, thresholds: Dict[str, float]) -> str:
    if prob < thresholds["real"]:
        return "REAL"
    if prob < thresholds["uncertain"]:
        return "UNCERTAIN"
    if prob < thresholds["fake"]:
        return "FAKE"
    return "FAKE"
