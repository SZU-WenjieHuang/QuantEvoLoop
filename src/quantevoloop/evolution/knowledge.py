"""Cross-campaign knowledge accumulation.

Maintains a structured knowledge base of what mutations work (High-EV)
and what doesn't (Low-EV / dead-ends) for future campaign planning.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class MutationRecord:
    """Record of a single mutation attempt."""
    mutation_type: str
    hypothesis_tag: str
    weakness_direction: str
    verdict: str  # promoted | rejected | dead_end
    score: float = 0.0
    generation: int = 0
    campaign_id: int = 0
    insight: str = ""
    timestamp: str = ""


@dataclass
class KnowledgeEntry:
    """Aggregated knowledge about a mutation type."""
    mutation_type: str
    weakness_direction: str
    total_attempts: int = 0
    promotions: int = 0
    rejects: int = 0
    dead_ends: int = 0
    avg_score: float = 0.0
    best_score: float = 0.0
    ev_rating: str = "unknown"  # high-ev | neutral | low-ev | dead
    notes: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.promotions / max(1, self.total_attempts)


class KnowledgeBase:
    """Persistent knowledge base tracking mutation effectiveness."""

    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, KnowledgeEntry] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for k, v in data.get("entries", {}).items():
                self._entries[k] = KnowledgeEntry(**v)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": {k: asdict(v) for k, v in self._entries.items()},
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        self.path.write_text(json.dumps(data, indent=2))

    def _key(self, mutation_type: str, direction: str) -> str:
        return f"{mutation_type}::{direction}"

    def record(self, record: MutationRecord) -> None:
        key = self._key(record.mutation_type, record.weakness_direction)
        if key not in self._entries:
            self._entries[key] = KnowledgeEntry(
                mutation_type=record.mutation_type,
                weakness_direction=record.weakness_direction,
            )
        entry = self._entries[key]
        entry.total_attempts += 1
        if record.verdict == "promoted":
            entry.promotions += 1
        elif record.verdict == "rejected":
            entry.rejects += 1
        elif record.verdict in ("dead_end", "regime-fragile"):
            entry.dead_ends += 1

        entry.avg_score = (
            (entry.avg_score * (entry.total_attempts - 1) + record.score)
            / entry.total_attempts
        )
        if record.score > entry.best_score:
            entry.best_score = record.score
        if record.insight:
            entry.notes.append(record.insight)

        # Classify EV rating
        if entry.total_attempts >= 3:
            if entry.success_rate >= 0.30:
                entry.ev_rating = "high-ev"
            elif entry.success_rate <= 0.05 and entry.dead_ends >= 2:
                entry.ev_rating = "dead"
            elif entry.success_rate <= 0.10:
                entry.ev_rating = "low-ev"
            else:
                entry.ev_rating = "neutral"

    def get_high_ev(self) -> list[KnowledgeEntry]:
        return [e for e in self._entries.values() if e.ev_rating == "high-ev"]

    def get_dead_ends(self) -> list[KnowledgeEntry]:
        return [e for e in self._entries.values() if e.ev_rating == "dead"]

    def get_prior(self, mutation_type: str, direction: str) -> KnowledgeEntry | None:
        return self._entries.get(self._key(mutation_type, direction))

    def to_context_string(self) -> str:
        """Generate a concise context string for LLM prompts."""
        high = self.get_high_ev()
        dead = self.get_dead_ends()
        lines = []
        if high:
            lines.append("HIGH-EV mutations (prioritize):")
            for e in high[:5]:
                lines.append(f"  - {e.mutation_type} on {e.weakness_direction}: "
                             f"{e.promotions}/{e.total_attempts} promoted, best={e.best_score:+.3f}")
        if dead:
            lines.append("DEAD mutations (avoid):")
            for e in dead[:5]:
                lines.append(f"  - {e.mutation_type} on {e.weakness_direction}: "
                             f"0/{e.total_attempts} promoted")
        return "\n".join(lines) if lines else "No prior knowledge yet."
