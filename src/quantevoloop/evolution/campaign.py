"""Campaign lifecycle management.

A Campaign groups a set of hypotheses/mutations targeting a specific
weakness direction. Campaigns run for a bounded number of iterations
and produce lessons for future campaigns.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class CampaignStatus(str, Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    ACHIEVED = "achieved"
    STALLED = "stalled"
    ABANDONED = "abandoned"


@dataclass
class CampaignLesson:
    """A lesson learned during a campaign iteration."""
    generation: int
    hypothesis: str
    verdict: str  # promoted | rejected | dead_end
    score: float | None = None
    insight: str = ""
    timestamp: str = ""


@dataclass
class Campaign:
    """A campaign targeting a specific weakness direction."""
    campaign_id: int
    name: str
    weakness_direction: str
    status: CampaignStatus = CampaignStatus.PLANNING
    max_iterations: int = 20
    current_iteration: int = 0
    best_score: float = 0.0
    best_generation: int = 0
    lessons: list[CampaignLesson] = field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def advance(self) -> int:
        self.current_iteration += 1
        if self.status == CampaignStatus.PLANNING:
            self.status = CampaignStatus.ACTIVE
            if not self.started_at:
                self.started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        return self.current_iteration

    def is_exhausted(self) -> bool:
        return self.current_iteration >= self.max_iterations

    def add_lesson(self, lesson: CampaignLesson) -> None:
        self.lessons.append(lesson)

    def record_best(self, gen_id: int, score: float) -> None:
        if score > self.best_score:
            self.best_score = score
            self.best_generation = gen_id

    def finalize(self, status: CampaignStatus) -> None:
        self.status = status
        self.ended_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["status"] = self.status.value
        data["lessons"] = [asdict(l) for l in self.lessons]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> Campaign:
        data = json.loads(path.read_text())
        lessons = [CampaignLesson(**l) for l in data.pop("lessons", [])]
        data["status"] = CampaignStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
                   lessons=lessons)

    def to_summary(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "direction": self.weakness_direction,
            "status": self.status.value,
            "iteration": f"{self.current_iteration}/{self.max_iterations}",
            "best_score": f"{self.best_score:+.4f}",
            "lessons": len(self.lessons),
        }
