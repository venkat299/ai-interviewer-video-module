"""Deepfake detection head using PyTorch Hub models."""
from __future__ import annotations
import numpy as np
import torch


class DeepfakeHead:
    def __init__(self, repo: str, model_name: str, device: str = "cpu"):
        self.device = device
        if repo and model_name:
            try:
                self.model = torch.hub.load(repo, model_name, pretrained=True).to(device).eval()
            except Exception:
                self.model = None
        else:
            self.model = None

    def preprocess(self, crop_rgb: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0)

    def fake_prob(self, crop_rgb: np.ndarray) -> float:
        if self.model is None:
            return 0.5
        with torch.no_grad():
            inp = self.preprocess(crop_rgb).to(self.device)
            out = self.model(inp)
            if isinstance(out, (list, tuple)):
                out = out[0]
            prob = torch.sigmoid(out).mean().item()
        return float(prob)

    def batch_fake_prob(self, crops: list[np.ndarray]) -> list[float]:
        return [self.fake_prob(c) for c in crops]
