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

from ..backends.base import CodeAgentBackend, BackendMutationContext


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
            # Copy champion strategy to lane dir before mutating
            gen_dir.mkdir(parents=True, exist_ok=True)
            mutated = gen_dir / "strategy.py"
            if champion_strategy.exists():
                mutated.write_text(champion_strategy.read_text(encoding="utf-8"), encoding="utf-8")
            else:
                mutated.write_text("# placeholder\nclass Strategy:\n    pass\n")

            ctx = BackendMutationContext(
                hypothesis=hypothesis,
                mutation_type=mutation_type,
                output_dir=gen_dir,
                knowledge_context=knowledge_context,
            )
            result = await self.backend.mutate_strategy(
                strategy_path=mutated,
                hypothesis=hypothesis,
                context=ctx,
            )
            duration = time.time() - t0

            if not result.success:
                return SubAgentResult(
                    lane_id=self.lane_id, gen_id=gen_id,
                    success=False, duration_s=duration,
                    cost_usd=result.cost_usd, error=result.error,
                )

            return SubAgentResult(
                lane_id=self.lane_id, gen_id=gen_id,
                success=True, strategy_path=mutated,
                duration_s=duration, cost_usd=result.cost_usd,
                metadata={"session_id": result.session_id, "raw": result.raw_output},
            )
        except Exception as e:
            return SubAgentResult(
                lane_id=self.lane_id, gen_id=gen_id,
                success=False, duration_s=time.time() - t0,
                error=str(e),
            )
