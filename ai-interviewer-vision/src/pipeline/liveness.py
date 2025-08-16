"""Simplified liveness head using geometric cues."""
from __future__ import annotations
import random
from collections import deque
from typing import Dict

import numpy as np


class LivenessHead:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.buffers: Dict[int, deque] = {}

    def fake_prob(self, crop_rgb: np.ndarray, track_id: int) -> float:
        """Return a dummy fake probability based on random blink heuristic."""
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.cfg.get("blink_window", 15))
        self.buffers[track_id].append(random.random())
        # simple average
        val = sum(self.buffers[track_id]) / len(self.buffers[track_id])
        return 1.0 - val
