"""Benchmark pipeline performance."""
from __future__ import annotations
import argparse
import time

import cv2

from src.app import deps
from src.pipeline import video_io


def main(source: str, seconds: int):
    cfg = deps.get_config()
    src = video_io.AsyncVideoSource(source, target_fps=cfg["video"]["target_fps"])
    src.start()
    start = time.time()
    frames = 0
    async def run():
        nonlocal frames
        async for ts, frame in src.frames():
            frames += 1
            if time.time() - start > seconds:
                break
        await src.stop()
    import asyncio
    asyncio.run(run())
    print("Captured", frames, "frames in", seconds, "seconds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--seconds", type=int, default=5)
    args = ap.parse_args()
    main(args.source, args.seconds)
