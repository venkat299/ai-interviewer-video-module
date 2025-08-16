"""Drawing helpers for visualisation."""
from __future__ import annotations
import cv2

COLORS = {
    "REAL": (0, 255, 0),
    "UNCERTAIN": (0, 255, 255),
    "FAKE": (0, 0, 255),
}


def draw(frame_bgr, faces_with_scores):
    for f in faces_with_scores:
        x, y, w, h = f["bbox"]
        decision = f.get("decision", "UNCERTAIN")
        color = COLORS.get(decision, (255, 255, 255))
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        text = f"{f.get('track_id', 0)}:{f.get('fused',0):.2f}:{decision}"
        cv2.putText(frame_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame_bgr
