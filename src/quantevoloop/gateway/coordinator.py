"""EvolutionCoordinator — the main evolution loop.

Orchestrates:
  1. LeadAgent diagnoses champion weakness
  2. LeadAgent proposes hypotheses
  3. N SubAgents execute mutations in parallel
  4. JudgeAgent evaluates each candidate
  5. Tournament selects best candidate
  6. Promoter swaps champion if promoted
  7. Knowledge base + Dead-end tracker updated
  8. IM notification sent
  9. Checkpoint saved
  10. Loop until max_campaign_iter or convergence
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

from ..agents.lead_agent import LeadAgent, Hypothesis
from ..agents.sub_agent import SubAgent
from ..agents.judge_agent import JudgeAgent
from ..backends.base import CodeAgentBackend
from ..config import QuantEvoLoopConfig
from ..engine.base import BacktestEngine
from ..evaluation.scorer import decide as scorer_decide
from ..evolution.campaign import Campaign, CampaignLesson, CampaignStatus
from ..evolution.dead_ends import DeadEnd, DeadEndTracker
from ..evolution.knowledge import KnowledgeBase, MutationRecord
from ..evolution.promoter import Promoter
from ..evolution.state import EvolutionState, GenerationIndex, GenerationRecord
from ..selection.population import Candidate, Population
from ..selection.reward import compute_reward
from ..selection.tournament import LaneCandidate, Tournament
from ..selection.ucb import UCBBandit
from .event_bus import EventBus, Event
from .lane_queue import LaneQueue, LaneTask

logger = logging.getLogger("quantevoloop.coordinator")


class EvolutionCoordinator:
    """Main evolution loop — runs N parallel lanes per generation."""

    def __init__(
        self,
        config: QuantEvoLoopConfig,
        backend: CodeAgentBackend,
        engine: BacktestEngine,
        notifier: Any = None,  # IMNotifier protocol
    ):
        self.config = config
        self.backend = backend
        self.engine = engine
        self.notifier = notifier

        # Core components
        self.state = EvolutionState.load(config.state_file)
        self.gen_index = GenerationIndex(config.generations_dir / "gen_index.jsonl")
        self.knowledge = KnowledgeBase(config.knowledge_dir / "knowledge.json")
        self.dead_ends = DeadEndTracker(config.dead_ends_file)
        self.promoter = Promoter(config.champion_dir, config.generations_dir)
        self.tournament = Tournament(n_lanes=config.n_lanes)
        self.population = Population()

        # UCB bandit for mutation type selection
        mutation_types = [
            "EXIT_TIGHTEN", "EXIT_LOOSEN", "NEW_FILTER", "BOX_FILTER",
            "COOLDOWN", "PARAM_TUNE", "GENERAL_TUNE", "ENTRY_REFINE",
        ]
        self.bandit = UCBBandit(mutation_types)

        # Agent layer
        self.lead_agent = LeadAgent(backend)
        self.lanes = [LaneQueue(i) for i in range(config.n_lanes)]
        self.event_bus = EventBus()

        # Campaign
        self.campaign: Campaign | None = None

    async def run(self, max_generations: int | None = None) -> EvolutionState:
        """Main evolution loop.

        Runs generations until:
          - max_generations reached
          - budget exhausted
          - convergence (consecutive_rejects > threshold)
        """
        max_gen = max_generations or self.config.max_campaign_iter
        self.state.status = "running"
        self.state.started_at = self.state.started_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        self.state.save(self.config.state_file)

        await self._notify("important", "Evolution started",
                           f"max_gen={max_gen}, lanes={self.config.n_lanes}")

        try:
            while self.state.generation < max_gen:
                # Check budget
                if self.config.cost.budget_usd and self.state.total_cost_usd >= self.config.cost.budget_usd:
                    logger.warning(f"Budget exhausted: ${self.state.total_cost_usd:.2f}")
                    break

                gen_id = self.state.advance_generation()
                gen_dir = self.config.generations_dir / f"gen_{gen_id:04d}"
                gen_dir.mkdir(parents=True, exist_ok=True)

                # Save checkpoint
                if gen_id % self.config.checkpoint.auto_save_every == 0:
                    self._save_checkpoint(gen_id)

                logger.info(f"=== Generation {gen_id:04d} ===")

                # 1. Diagnose champion (only on first gen or after promotion)
                if gen_id == 1 or self.state.consecutive_rejects == 0:
                    # Use mock diagnostics for now (real BT data comes from engine)
                    hypotheses = [Hypothesis(
                        weakness_tag="auto",
                        mutation_type=self.bandit.select(),
                        direction="auto-diagnose",
                        description="Auto-selected mutation via UCB bandit",
                    )]
                else:
                    hypotheses = [Hypothesis(
                        weakness_tag="auto",
                        mutation_type=self.bandit.select(),
                        direction="auto-diagnose",
                        description="Auto-selected mutation via UCB bandit",
                    )]

                # 2. Launch parallel lanes
                champion_strategy = self.config.champion_dir / "strategy.py"
                tasks = []
                for lane_id in range(self.config.n_lanes):
                    h = hypotheses[lane_id % len(hypotheses)]
                    lane_gen_dir = gen_dir / f"lane_{lane_id}"
                    lane_gen_dir.mkdir(parents=True, exist_ok=True)

                    sub_agent = SubAgent(lane_id, self.backend)
                    task = sub_agent.execute(
                        gen_id=gen_id,
                        champion_strategy=champion_strategy,
                        hypothesis=h.description,
                        mutation_type=h.mutation_type,
                        gen_dir=lane_gen_dir,
                        knowledge_context=self.knowledge.to_context_string(),
                    )
                    tasks.append(task)

                # 3. Execute all lanes concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 4. Evaluate each successful mutation
                lane_candidates = []
                champion_metrics = self._load_champion_metrics()

                for lane_id, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Lane {lane_id} failed: {result}")
                        continue
                    if not result.success or not result.strategy_path:
                        logger.warning(f"Lane {lane_id} mutation failed: {result.error}")
                        continue

                    # Evaluate candidate
                    judge = JudgeAgent(self.engine, self.config.statistical_gates.model_dump())
                    judge_result = await judge.evaluate(
                        candidate_strategy=result.strategy_path,
                        champion_metrics=champion_metrics,
                        gen_dir=lane_gen_dir,
                    )

                    # Record in population
                    cand = Candidate(
                        gen_id=gen_id,
                        lane_id=lane_id,
                        score=judge_result.score or 0.0,
                        sharpe_test=0.0,
                        sharpe_train=0.0,
                        verdict=judge_result.verdict,
                        cost_usd=result.cost_usd,
                        duration_s=result.duration_s,
                    )
                    self.population.add(cand)

                    # Update UCB bandit
                    mutation_type = hypotheses[lane_id % len(hypotheses)].mutation_type
                    reward = compute_reward(judge_result.verdict, judge_result.score)
                    self.bandit.update(mutation_type, reward)

                    # Update generation index
                    self.gen_index.append(GenerationRecord(
                        gen=gen_id, lane=lane_id,
                        status=judge_result.verdict,
                        score=judge_result.score,
                        hypothesis_tag=mutation_type,
                        duration_s=result.duration_s,
                        cost_usd=result.cost_usd,
                        timestamp=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    ))

                    # Update state cost
                    self.state.update_cost(result.cost_usd)

                    # Track for tournament
                    lane_candidates.append(LaneCandidate(
                        lane_id=lane_id, gen_id=gen_id,
                        score=judge_result.score or 0.0,
                        sharpe_test=0.0, sharpe_train=0.0,
                        verdict=judge_result.verdict,
                    ))

                # 5. Tournament selection
                winner = self.tournament.add_round(lane_candidates)

                if winner:
                    # Promote!
                    self.state.record_promotion()
                    promote_result = self.promoter.promote(
                        gen_id=winner.gen_id,
                        score=winner.score,
                        cand_train={},
                        cand_test={},
                    )
                    logger.info(f"PROMOTED: gen_{winner.gen_id:04d} score={winner.score:+.4f}")
                    await self._notify("important",
                                       f"New Champion: gen_{winner.gen_id:04d}",
                                       f"score={winner.score:+.4f}")
                else:
                    self.state.record_reject()
                    logger.info(f"Generation {gen_id:04d}: no promotable candidate")

                # 6. Save state
                self.state.save(self.config.state_file)
                self.knowledge.save()
                self.dead_ends.save()

                # 7. IM notification
                status_msg = (
                    f"Gen {gen_id:04d}: {len(lane_candidates)} evaluated, "
                    f"{'PROMOTED' if winner else 'no promote'}, "
                    f"cost=${self.state.total_cost_usd:.2f}"
                )
                await self._notify("info", f"Gen {gen_id:04d}", status_msg)

        except KeyboardInterrupt:
            logger.info("Evolution interrupted by user")
        finally:
            self.state.status = "stopped"
            self.state.save(self.config.state_file)
            self.knowledge.save()
            self.dead_ends.save()

        return self.state

    def _load_champion_metrics(self) -> dict[str, Any]:
        metrics_path = self.config.champion_dir / "metrics.json"
        if metrics_path.exists():
            return json.loads(metrics_path.read_text())
        return {"train": {}, "test": {}, "holdout": None}

    def _save_checkpoint(self, gen_id: int) -> None:
        ckpt = {
            "gen_id": gen_id,
            "state": json.loads(self.config.state_file.read_text()) if self.config.state_file.exists() else {},
            "bandit": self.bandit.get_arm_stats(),
            "population": self.population.to_summary(),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        ckpt_path = self.config.workspace_dir / f"checkpoint_gen{gen_id:04d}.json"
        ckpt_path.write_text(json.dumps(ckpt, indent=2))

    async def _notify(self, level: str, title: str, message: str) -> None:
        if self.notifier:
            try:
                await self.notifier.send(level, title, message)
            except Exception:
                pass
