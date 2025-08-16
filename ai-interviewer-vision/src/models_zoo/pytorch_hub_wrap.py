"""Utility to load PyTorch Hub models."""
from __future__ import annotations
from typing import Callable

import torch


def load_model(repo: str, model_name: str, device: str) -> torch.nn.Module:
    if not repo or not model_name:
        class Dummy(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((x.shape[0], 1))
        return Dummy()
    model = torch.hub.load(repo, model_name, pretrained=True)
    model.to(device).eval()
    if device == "cuda":
        model.half()
    return model
