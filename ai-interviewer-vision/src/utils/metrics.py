"""Prometheus metrics helpers."""
from __future__ import annotations
from contextlib import contextmanager
from time import time

from prometheus_client import Gauge, Histogram

fps_gauge = Gauge("vision_fps", "Frames per second")
queue_size_gauge = Gauge("vision_queue_size", "Capture queue size")
latency_histogram = Histogram("vision_stage_latency_ms", "Stage latency", ["stage"])


@contextmanager
def record_latency(stage: str):
    start = time()
    try:
        yield
    finally:
        elapsed = (time() - start) * 1000.0
        latency_histogram.labels(stage=stage).observe(elapsed)
