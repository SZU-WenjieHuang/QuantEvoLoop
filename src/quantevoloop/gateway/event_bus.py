"""Simple event bus for inter-component communication."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class Event:
    """An event in the evolution pipeline."""
    event_type: str  # mutation_started | mutation_done | evaluation_done | promoted | rejected
    gen_id: int = 0
    lane_id: int = 0
    payload: dict[str, Any] = field(default_factory=dict)


# Type alias for event handlers
EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Publish/subscribe event bus for pipeline coordination."""

    def __init__(self):
        self._handlers: dict[str, list[EventHandler]] = {}

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def publish(self, event: Event) -> None:
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception:
                pass  # event handlers should not crash the pipeline

    def unsubscribe_all(self) -> None:
        self._handlers.clear()
