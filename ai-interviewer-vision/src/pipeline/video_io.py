"""Async video capture utilities."""
from __future__ import annotations
import asyncio
import time
from typing import AsyncGenerator

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class AsyncVideoSource:
    """Asynchronous wrapper around OpenCV VideoCapture."""

    def __init__(self, source: str | int, target_fps: int = 20, max_queue: int = 5):
        if cv2 is None:
            raise ImportError("cv2 is required for AsyncVideoSource")
        self.source = source
        self.target_fps = target_fps
        self.max_queue = max_queue
        self._queue: asyncio.Queue = asyncio.Queue(max_queue)
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self._reader())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            await self._task
            self._task = None

    async def frames(self) -> AsyncGenerator[tuple[float, any], None]:
        while self._running or not self._queue.empty():
            ts, frame = await self._queue.get()
            yield ts, frame

    async def _reader(self) -> None:
        cap = cv2.VideoCapture(self.source)
        delay = 1.0 / float(self.target_fps)
        while self._running:
            ts = time.time()
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self._queue.put((ts, frame))
            await asyncio.sleep(delay)
        cap.release()
