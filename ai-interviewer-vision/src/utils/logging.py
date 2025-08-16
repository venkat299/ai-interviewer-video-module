"""Simple JSON logging helpers."""
from __future__ import annotations
import json
import sys
import time


def _log(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def log_frame_result(session_id: str, decision: str, scores: dict, track_ids: list[int]):
    _log({
        "ts": time.time(),
        "session": session_id,
        "decision": decision,
        "scores": scores,
        "tracks": track_ids,
    })


def log_perf(stage: str, latency_ms: float):
    _log({"ts": time.time(), "stage": stage, "latency_ms": latency_ms})
