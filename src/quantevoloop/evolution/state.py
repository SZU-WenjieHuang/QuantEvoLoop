"""Evolution state management — persistent state.json.

Tracks the current generation, champion metrics, generation index, etc.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class GenerationRecord:
    """One row in the generation index (gen_index.jsonl)."""
    gen: int
    lane: int
    status: str = ""  # pending | running | promoted | rejected | dead_end | bt_error
    score: float | None = None
    sharpe_test: float | None = None
    sharpe_train: float | None = None
    hypothesis_tag: str = ""
    mutation_type: str = ""
    duration_s: float = 0.0
    cost_usd: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EvolutionState:
    """Persistent evolution state (saved to state.json)."""
    generation: int = 0
    campaign_index: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    started_at: str = ""
    last_updated: str = ""
    champion_generation: int = 0
    champion_sharpe_test: float = 0.0
    champion_sharpe_train: float = 0.0
    champion_holdout_sharpe: float = 0.0
    consecutive_rejects: int = 0
    total_promotions: int = 0
    total_dead_ends: int = 0
    total_rejects: int = 0
    status: str = "idle"  # idle | running | paused | stopped

    def save(self, path: Path) -> None:
        self.last_updated = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> EvolutionState:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def advance_generation(self) -> int:
        self.generation += 1
        return self.generation

    def record_promotion(self) -> None:
        self.total_promotions += 1
        self.consecutive_rejects = 0
        self.champion_generation = self.generation

    def record_reject(self) -> None:
        self.total_rejects += 1
        self.consecutive_rejects += 1

    def record_dead_end(self) -> None:
        self.total_dead_ends += 1
        self.consecutive_rejects += 1

    def update_cost(self, cost: float) -> None:
        self.total_cost_usd += cost
        self.total_calls += 1


class GenerationIndex:
    """Append-only generation index (gen_index.jsonl)."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, record: GenerationRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def update_status(self, gen_id: int, status: str, score: float | None = None) -> None:
        rows = self.load_all()
        for r in reversed(rows):
            if r.get("gen") == gen_id:
                r["status"] = status
                if score is not None:
                    r["score"] = score
                break
        with self.path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def count_non_baseline(self) -> int:
        return sum(1 for r in self.load_all() if r.get("gen", 0) > 0)
