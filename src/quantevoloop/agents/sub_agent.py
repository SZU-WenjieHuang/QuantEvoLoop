"""SubAgent — executes mutations in parallel lanes.

Each SubAgent:
  1. Receives a hypothesis from LeadAgent
  2. Calls the backend to mutate the strategy
  3. Returns the mutation result for evaluation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backends.base import CodeAgentBackend


@dataclass
class SubAgentResult:
    """Result from a SubAgent mutation execution."""
    lane_id: int
    gen_id: int
    success: bool
    strategy_path: Path | None = None
    duration_s: float = 0.0
    cost_usd: float = 0.0
    error: str = ""
    metadata: dict[str, Any] | None = None


class SubAgent:
    """Executes a single mutation in a dedicated lane."""

    def __init__(self, lane_id: int, backend: CodeAgentBackend):
        self.lane_id = lane_id
        self.backend = backend

    async def execute(
        self,
        gen_id: int,
        champion_strategy: Path,
        hypothesis: str,
        mutation_type: str,
        gen_dir: Path,
        knowledge_context: str = "",
    ) -> SubAgentResult:
        """Execute the mutation and return result."""
        t0 = time.time()
        try:
            result = await self.backend.mutate_strategy(
                strategy_path=champion_strategy,
                hypothesis=hypothesis,
                mutation_type=mutation_type,
                output_dir=gen_dir,
                knowledge_context=knowledge_context,
            )
            duration = time.time() - t0
            cost = result.get("cost_usd", 0.0)

            # Check if the mutated strategy file exists
            mutated = gen_dir / "strategy.py"
            if not mutated.exists():
                return SubAgentResult(
                    lane_id=self.lane_id, gen_id=gen_id,
                    success=False, duration_s=duration,
                    cost_usd=cost, error="mutated strategy.py not found",
                )

            return SubAgentResult(
                lane_id=self.lane_id, gen_id=gen_id,
                success=True, strategy_path=mutated,
                duration_s=duration, cost_usd=cost,
                metadata=result,
            )
        except Exception as e:
            return SubAgentResult(
                lane_id=self.lane_id, gen_id=gen_id,
                success=False, duration_s=time.time() - t0,
                error=str(e),
            )
