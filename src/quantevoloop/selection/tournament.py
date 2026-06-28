"""Multi-lane tournament selection.

Manages N parallel SubAgent lanes. Each lane independently evolves
the strategy. The best candidate across all lanes competes in a
tournament to become the next champion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LaneCandidate:
    """A candidate produced by one lane."""
    lane_id: int
    gen_id: int
    score: float
    sharpe_test: float
    sharpe_train: float
    verdict: str  # pending | promoted | rejected | dead_end
    metadata: dict[str, Any] = field(default_factory=dict)


class Tournament:
    """Tournament selection across multiple lanes.

    Each generation round, all lanes produce candidates.
    The candidate with the highest score (above promotion threshold)
    is selected for promotion.
    """

    def __init__(self, n_lanes: int = 3):
        self.n_lanes = n_lanes
        self.rounds: list[list[LaneCandidate]] = []

    def add_round(self, candidates: list[LaneCandidate]) -> LaneCandidate | None:
        """Add a round of candidates and select the best."""
        self.rounds.append(candidates)
        # Filter to only promotable candidates (score > 0, verdict=promoted)
        promotable = [c for c in candidates if c.verdict == "promoted" and c.score > 0]
        if not promotable:
            return None
        return max(promotable, key=lambda c: c.score)

    def get_lane_scores(self) -> dict[int, dict]:
        """Aggregate per-lane performance across all rounds."""
        lane_stats: dict[int, dict] = {}
        for round_cands in self.rounds:
            for c in round_cands:
                if c.lane_id not in lane_stats:
                    lane_stats[c.lane_id] = {"rounds": 0, "promotions": 0, "total_score": 0.0}
                stats = lane_stats[c.lane_id]
                stats["rounds"] += 1
                stats["total_score"] += c.score
                if c.verdict == "promoted":
                    stats["promotions"] += 1
        for stats in lane_stats.values():
            stats["avg_score"] = stats["total_score"] / max(1, stats["rounds"])
        return lane_stats
