"""FastAPI application entry point."""
from __future__ import annotations
import base64
import time
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter

from . import schemas, deps
from ..pipeline import preprocess, face_track, deepfake, liveness, fusion, overlays
from ..utils import metrics

app = FastAPI(title="AI Interviewer Vision")
router = APIRouter()


@router.get("/health", response_model=schemas.HealthOut)
async def health(cfg = Depends(deps.get_config), registry: deps.ModelRegistry = Depends(deps.get_registry)):
    return schemas.HealthOut(status="ok", fps=0.0, device=registry.device, models=["detector", "deepfake"])


@router.post("/detect", response_model=schemas.DetectOut)
async def detect(frame: UploadFile | None = File(None), body: schemas.FrameIn | None = None,
                 cfg = Depends(deps.get_config), registry: deps.ModelRegistry = Depends(deps.get_registry)):
    data = None
    if frame is not None:
        data = await frame.read()
    elif body and body.image_base64:
        data = base64.b64decode(body.image_base64)
    else:
        return JSONResponse(status_code=400, content={"error": "no image"})

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # Fake pipeline: just run detector
    faces = registry.detector(img)
    out_faces: List[schemas.FaceOut] = []
    for i, f in enumerate(faces):
        bbox = f.get("bbox", [0, 0, 0, 0])
        out_faces.append(schemas.FaceOut(
            bbox=bbox,
            track_id=i,
            liveness=0.5,
            deepfake=0.5,
            fused=0.5,
            decision="UNCERTAIN",
        ))
    return schemas.DetectOut(faces=out_faces, ts=time.time())


app.include_router(router)


@app.websocket("/stream")
async def stream(ws: WebSocket, registry: deps.ModelRegistry = Depends(deps.get_registry)):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            faces = registry.detector(img)
            resp = schemas.DetectOut(
                faces=[schemas.FaceOut(bbox=f.get("bbox", [0, 0, 0, 0]), track_id=i,
                                        liveness=0.5, deepfake=0.5, fused=0.5,
                                        decision="UNCERTAIN")
                        for i, f in enumerate(faces)],
                ts=time.time(),
            )
            await ws.send_json(resp.model_dump())
    except WebSocketDisconnect:
        return


@app.get("/metrics")
async def metrics_endpoint():
    from prometheus_client import generate_latest
    return JSONResponse(content=generate_latest().decode("utf-8"))
