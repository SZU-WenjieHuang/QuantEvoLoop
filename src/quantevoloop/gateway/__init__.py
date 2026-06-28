"""Gateway layer — parallel execution coordinator.

Manages concurrent lane execution via asyncio, coordinates
mutation → evaluation → selection pipeline.
"""

from .event_bus import EventBus, Event
from .lane_queue import LaneQueue
from .coordinator import EvolutionCoordinator

__all__ = ["EventBus", "Event", "LaneQueue", "EvolutionCoordinator"]
