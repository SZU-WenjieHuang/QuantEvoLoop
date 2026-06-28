"""Lane queue — manages per-lane work queues for parallel mutation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class LaneTask:
    """A unit of work for a lane to execute."""
    gen_id: int
    lane_id: int
    hypothesis: str
    mutation_type: str
    champion_strategy: str  # path as string
    knowledge_context: str = ""


class LaneQueue:
    """Async queue for a single lane's mutation tasks."""

    def __init__(self, lane_id: int):
        self.lane_id = lane_id
        self._queue: asyncio.Queue[LaneTask] = asyncio.Queue()
        self._running = False

    async def put(self, task: LaneTask) -> None:
        await self._queue.put(task)

    async def get(self) -> LaneTask:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()
