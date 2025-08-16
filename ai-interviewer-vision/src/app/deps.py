"""Dependency utilities for the FastAPI app."""
from __future__ import annotations
import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict

from . import schemas
from ..models_zoo import deepface_wrap, pytorch_hub_wrap

CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "../../configs/default.yaml"))
THRESH_PATH = os.path.join(os.path.dirname(__file__), "../../configs/thresholds.yaml")


def load_config() -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelRegistry:
    detector: Any
    deepfake: Any
    device: str


def init_models(cfg: Dict[str, Any]) -> ModelRegistry:
    """Instantiate models based on configuration."""
    device = "cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu"
    detector = deepface_wrap.load_detector(cfg["models"]["face_detector"])
    df_cfg = cfg["models"].get("deepfake", {})
    deepfake = None
    if df_cfg.get("enable", True):
        deepfake = pytorch_hub_wrap.load_model(df_cfg.get("hub_repo"), df_cfg.get("model_name"), device)
    return ModelRegistry(detector=detector, deepfake=deepfake, device=device)


_config = load_config()
_registry = init_models(_config)


def get_config() -> Dict[str, Any]:
    return _config


def get_registry() -> ModelRegistry:
    return _registry
