"""Pydantic models for API inputs and outputs."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class FrameIn(BaseModel):
    """Frame input, expects base64 encoded image data."""
    image_base64: Optional[str] = None


class FaceOut(BaseModel):
    bbox: List[int]
    track_id: int
    liveness: float
    deepfake: float
    fused: float
    decision: str


class DetectOut(BaseModel):
    faces: List[FaceOut] = Field(default_factory=list)
    ts: float


class HealthOut(BaseModel):
    status: str
    fps: float
    device: str
    models: List[str]
