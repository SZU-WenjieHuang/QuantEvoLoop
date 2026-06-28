"""LeadAgent — diagnoses weakness, proposes hypotheses, launches campaigns.

The LeadAgent:
  1. Runs diagnostics on the champion's backtest results
  2. Identifies the top weakness directions
  3. Proposes structured hypotheses for each direction
  4. Delegates mutation work to SubAgents
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..backends.base import CodeAgentBackend
from ..evaluation.diagnostics import DiagnosticReport, diagnose


@dataclass
class Hypothesis:
    """A structured mutation hypothesis proposed by the LeadAgent."""
    weakness_tag: str
    mutation_type: str
    direction: str
    description: str
    hint_param: str = ""
    expected_improvement: str = ""
    confidence: str = "medium"  # low | medium | high


class LeadAgent:
    """Orchestrates the evolution process: diagnose → hypothesize → delegate."""

    def __init__(self, backend: CodeAgentBackend):
        self.backend = backend

    async def diagnose_champion(
        self,
        trades: list[dict],
        strategy_summary: dict[str, Any],
        segment: str = "test",
    ) -> DiagnosticReport:
        """Run diagnostic analysis on champion's backtest trades."""
        return diagnose(trades, strategy_summary, segment)

    def propose_hypotheses(
        self,
        diagnostic: DiagnosticReport,
        knowledge_context: str = "",
        dead_ends_context: str = "",
    ) -> list[Hypothesis]:
        """Generate hypotheses from diagnostic weaknesses.

        Each weakness gets a corresponding hypothesis with mutation direction.
        Filters out hypotheses that match known dead-ends.
        """
        hypotheses: list[Hypothesis] = []
        for w in diagnostic.weaknesses:
            h = Hypothesis(
                weakness_tag=w.issue,
                mutation_type=w.hint_mutation,
                direction=w.issue,
                description=w.detail,
                hint_param=w.hint_param,
                expected_improvement=f"reduce {w.issue} loss share from {w.loss_share_pct}%",
                confidence="high" if w.loss_share_pct >= 30 else "medium",
            )
            hypotheses.append(h)

        # If no weaknesses detected, fall back to generic improvement
        if not hypotheses:
            hypotheses.append(Hypothesis(
                weakness_tag="no-specific-weakness",
                mutation_type="GENERAL_TUNE",
                direction="overall-optimization",
                description="No dominant weakness detected. Try general parameter tuning.",
                hint_param="various parameters",
            ))

        return hypotheses

    async def mutate(
        self,
        strategy_path: Path,
        hypothesis: Hypothesis,
        gen_dir: Path,
        knowledge_context: str = "",
    ) -> dict[str, Any]:
        """Delegate a mutation to the backend."""
        return await self.backend.mutate_strategy(
            strategy_path=strategy_path,
            hypothesis=hypothesis.description,
            mutation_type=hypothesis.mutation_type,
            output_dir=gen_dir,
            knowledge_context=knowledge_context,
        )
