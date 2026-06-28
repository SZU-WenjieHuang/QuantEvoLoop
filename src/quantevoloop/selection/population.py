"""Candidate population management.

Tracks the current population of strategy candidates across lanes,
with support for pruning and archival.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    gen_id: int
    lane_id: int
    score: float
    sharpe_test: float
    sharpe_train: float
    verdict: str = "pending"
    mutation_type: str = ""
    hypothesis_tag: str = ""
    cost_usd: float = 0.0
    duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Population:
    """Manages the candidate population for the current evolution run."""

    def __init__(self):
        self._candidates: list[Candidate] = []

    def add(self, candidate: Candidate) -> None:
        self._candidates.append(candidate)

    def get_by_gen(self, gen_id: int) -> Candidate | None:
        for c in self._candidates:
            if c.gen_id == gen_id:
                return c
        return None

    def get_by_lane(self, lane_id: int) -> list[Candidate]:
        return [c for c in self._candidates if c.lane_id == lane_id]

    def get_promoted(self) -> list[Candidate]:
        return [c for c in self._candidates if c.verdict == "promoted"]

    def get_rejected(self) -> list[Candidate]:
        return [c for c in self._candidates if c.verdict == "rejected"]

    def get_dead_ends(self) -> list[Candidate]:
        return [c for c in self._candidates if c.verdict in ("dead_end", "regime-fragile")]

    def best(self) -> Candidate | None:
        promotable = self.get_promoted()
        return max(promotable, key=lambda c: c.score) if promotable else None

    def size(self) -> int:
        return len(self._candidates)

    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self._candidates)

    def to_summary(self) -> dict:
        return {
            "total": self.size(),
            "promoted": len(self.get_promoted()),
            "rejected": len(self.get_rejected()),
            "dead_ends": len(self.get_dead_ends()),
            "total_cost_usd": round(self.total_cost(), 4),
        }
