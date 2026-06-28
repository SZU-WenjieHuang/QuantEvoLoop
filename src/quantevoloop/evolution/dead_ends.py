"""Dead-end tracking and deduplication.

Records mutations that consistently fail so future generations
can avoid repeating them.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class DeadEnd:
    gen: int
    mutation_type: str
    hypothesis_tag: str
    failure_tag: str
    reason: str
    timestamp: str = ""


class DeadEndTracker:
    """Track and deduplicate dead-end mutations."""

    def __init__(self, path: Path):
        self.path = path
        self._entries: list[DeadEnd] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            content = self.path.read_text()
            # Support both JSON array and markdown format
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    self._entries = [DeadEnd(**e) for e in data]
            except (json.JSONDecodeError, TypeError):
                pass  # markdown format, skip

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._entries]
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, entry: DeadEnd) -> bool:
        """Add a dead-end. Returns True if it's a new (non-duplicate) entry."""
        # Dedup: same mutation_type + failure_tag + hypothesis_tag
        for existing in self._entries:
            if (existing.mutation_type == entry.mutation_type
                    and existing.failure_tag == entry.failure_tag
                    and existing.hypothesis_tag == entry.hypothesis_tag):
                return False
        entry.timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self._entries.append(entry)
        return True

    def is_known_dead_end(self, mutation_type: str, hypothesis_tag: str) -> bool:
        for e in self._entries:
            if e.mutation_type == mutation_type and e.hypothesis_tag == hypothesis_tag:
                return True
        return False

    def count(self) -> int:
        return len(self._entries)

    def to_markdown(self) -> str:
        """Export as markdown for human readability."""
        lines = ["# Dead Ends", "", "| gen | mutation | hypothesis | failure | reason |",
                 "| --- | --- | --- | --- | --- |"]
        for e in self._entries:
            lines.append(f"| {e.gen} | {e.mutation_type} | {e.hypothesis_tag} | "
                         f"{e.failure_tag} | {e.reason} |")
        return "\n".join(lines)

    def to_context_string(self) -> str:
        """Generate a concise context string for LLM prompts."""
        if not self._entries:
            return "No known dead ends."
        lines = ["Known dead-end mutations (AVOID these directions):"]
        for e in self._entries[-10:]:
            lines.append(f"  - {e.mutation_type} on {e.hypothesis_tag}: "
                         f"failed with {e.failure_tag}")
        return "\n".join(lines)
