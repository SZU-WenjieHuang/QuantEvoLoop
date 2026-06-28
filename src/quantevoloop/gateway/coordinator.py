"""EvolutionCoordinator — the main evolution loop.

Orchestrates:
  1. Bootstrap champion: run initial train/test/holdout backtests
  2. LeadAgent diagnoses champion weakness
  3. LeadAgent proposes structured hypotheses
  4. N SubAgents execute mutations in parallel
  5. JudgeAgent evaluates each candidate (5-gate pipeline)
  6. Tournament selects best candidate
  7. Promoter swaps champion if promoted
  8. Knowledge base + Dead-end tracker updated
  9. IM notification sent
  10. Checkpoint saved
  11. Loop until max_campaign_iter or convergence
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from ..agents.lead_agent import LeadAgent, Hypothesis
from ..agents.sub_agent import SubAgent
from ..agents.judge_agent import JudgeAgent
from ..backends.base import CodeAgentBackend
from ..config import QuantEvoLoopConfig
from ..engine.base import BacktestEngine
from ..evaluation.diagnostics import diagnose
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

# Maximum consecutive rejections before stopping
MAX_CONSECUTIVE_REJECTS = 10


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

        # Champion baseline (populated by _bootstrap_champion)
        self.champion_metrics: dict[str, Any] = {"train": {}, "test": {}, "holdout": None}
        self.champion_trades: dict[str, list[dict]] = {"train": [], "test": []}

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap_champion(self) -> None:
        """Run initial backtests on the champion strategy to establish baseline.

        Saves metrics to champion/metrics.json and champion/trades.json.
        If metrics.json already exists, loads it instead of re-running.
        """
        metrics_path = self.config.champion_dir / "metrics.json"
        trades_path = self.config.champion_dir / "trades.json"
        strategy_path = self.config.strategy_path

        # If baseline already exists, load it
        if metrics_path.exists():
            self.champion_metrics = json.loads(metrics_path.read_text())
            if trades_path.exists():
                self.champion_trades = json.loads(trades_path.read_text())
            logger.info("Loaded existing champion baseline from %s", metrics_path)
            return

        logger.info("Bootstrapping champion baseline...")
        await self._notify("info", "Bootstrap", "Running initial champion backtests...")

        # Copy strategy to champion dir
        self.config.champion_dir.mkdir(parents=True, exist_ok=True)
        champion_strategy = self.config.champion_dir / "strategy.py"
        if not champion_strategy.exists() and strategy_path.exists():
            shutil.copy2(strategy_path, champion_strategy)

        # Run train backtest
        train_bt = await self.engine.run_backtest(
            champion_strategy, self.engine.train_timerange
        )
        if train_bt.error:
            logger.warning("Champion train backtest error: %s", train_bt.error)

        # Run test backtest
        test_bt = await self.engine.run_backtest(
            champion_strategy, self.engine.test_timerange
        )
        if test_bt.error:
            logger.warning("Champion test backtest error: %s", test_bt.error)

        # Run holdout backtest
        holdout_bt = await self.engine.run_backtest(
            champion_strategy, self.engine.holdout_timerange
        )

        # Build metrics
        self.champion_metrics = {
            "generation": 0,
            "train": train_bt.to_dict(),
            "test": test_bt.to_dict(),
            "holdout": holdout_bt.to_dict() if not holdout_bt.error else None,
        }
        self.champion_trades = {
            "train": train_bt.trades,
            "test": test_bt.trades,
        }

        # Save
        metrics_path.write_text(json.dumps(self.champion_metrics, indent=2, default=str))
        trades_path.write_text(json.dumps(self.champion_trades, indent=2, default=str))

        logger.info(
            "Champion baseline: train_sharpe=%.3f test_sharpe=%.3f trades=%d/%d",
            train_bt.sharpe, test_bt.sharpe, train_bt.total_trades, test_bt.total_trades,
        )
        await self._notify(
            "important", "Champion Baseline",
            f"train_sharpe={train_bt.sharpe:.3f} test_sharpe={test_bt.sharpe:.3f}",
        )

    # ------------------------------------------------------------------
    # Diagnosis + Hypothesis
    # ------------------------------------------------------------------

    def _diagnose_and_hypothesize(self) -> list[Hypothesis]:
        """Run diagnostics on champion trades and generate hypotheses."""
        # Prefer test-segment trades for diagnosis
        trades = self.champion_trades.get("test", []) or self.champion_trades.get("train", [])
        test_metrics = self.champion_metrics.get("test", {})

        if not trades:
            # No trades available — fall back to UCB-selected generic mutation
            logger.warning("No champion trades for diagnosis, using UCB fallback")
            return [Hypothesis(
                weakness_tag="no-data",
                mutation_type=self.bandit.select(),
                direction="auto-diagnose",
                description="No trade data for diagnosis. Auto-selected mutation via UCB bandit.",
            )]

        # Run diagnostics
        strategy_summary = {
            "sharpe": test_metrics.get("sharpe", 0.0),
            "cagr": test_metrics.get("cagr", 0.0),
            "max_drawdown_account": test_metrics.get("max_drawdown_account", 0.0),
            "total_trades": test_metrics.get("total_trades", 0),
            "winrate": test_metrics.get("winrate", 0.0),
        }
        diagnostic = diagnose(trades, strategy_summary, segment="test")

        # Generate hypotheses from weaknesses
        hypotheses = self.lead_agent.propose_hypotheses(
            diagnostic,
            knowledge_context=self.knowledge.to_context_string(),
            dead_ends_context=self.dead_ends.to_context_string(),
        )

        logger.info(
            "Diagnosed %d weaknesses, proposed %d hypotheses",
            len(diagnostic.weaknesses), len(hypotheses),
        )

        # If no specific hypotheses, use UCB fallback
        if not hypotheses:
            return [Hypothesis(
                weakness_tag="no-specific-weakness",
                mutation_type=self.bandit.select(),
                direction="auto-diagnose",
                description="No specific weakness detected. Auto-selected mutation via UCB bandit.",
            )]

        return hypotheses

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, max_generations: int | None = None) -> EvolutionState:
        """Main evolution loop.

        Runs generations until:
          - max_generations reached
          - budget exhausted
          - convergence (consecutive_rejects > threshold)
        """
        max_gen = max_generations or self.config.max_campaign_iter
        self.config.ensure_dirs()

        # Bootstrap champion baseline
        await self._bootstrap_champion()

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

                # Check convergence
                if self.state.consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                    logger.info("Convergence: %d consecutive rejections", self.state.consecutive_rejects)
                    break

                gen_id = self.state.advance_generation()
                gen_dir = self.config.generations_dir / f"gen_{gen_id:04d}"
                gen_dir.mkdir(parents=True, exist_ok=True)

                # Save checkpoint
                if gen_id % self.config.checkpoint.auto_save_every == 0:
                    self._save_checkpoint(gen_id)

                logger.info(f"=== Generation {gen_id:04d} ===")

                # 1. Diagnose champion and generate hypotheses
                if gen_id == 1 or self.state.consecutive_rejects == 0:
                    # Fresh diagnosis after promotion or first gen
                    hypotheses = self._diagnose_and_hypothesize()
                else:
                    # Re-use diagnosis but vary mutation types via UCB
                    hypotheses = [Hypothesis(
                        weakness_tag="auto",
                        mutation_type=self.bandit.select(),
                        direction="auto-diagnose",
                        description="Auto-selected mutation via UCB bandit (no new diagnosis).",
                    )]

                # 2. Launch parallel lanes
                champion_strategy = self.config.champion_dir / "strategy.py"
                tasks = []
                lane_gen_dirs = []
                for lane_id in range(self.config.n_lanes):
                    h = hypotheses[lane_id % len(hypotheses)]
                    lane_gen_dir = gen_dir / f"lane_{lane_id}"
                    lane_gen_dir.mkdir(parents=True, exist_ok=True)
                    lane_gen_dirs.append((lane_id, h, lane_gen_dir))

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

                for lane_id, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Lane {lane_id} failed: {result}")
                        continue
                    if not result.success or not result.strategy_path:
                        logger.warning(f"Lane {lane_id} mutation failed: {result.error}")
                        continue

                    # Find the hypothesis for this lane
                    _, hypothesis, lane_gen_dir = lane_gen_dirs[lane_id]

                    # Evaluate candidate
                    judge = JudgeAgent(
                        self.engine,
                        self.config.statistical_gates.model_dump(),
                        train_start=self.config.data_splits.train_start,
                        train_end=self.config.data_splits.train_end,
                    )
                    judge_result = await judge.evaluate(
                        candidate_strategy=result.strategy_path,
                        champion_metrics=self.champion_metrics,
                        gen_dir=lane_gen_dir,
                    )

                    # Record in population
                    cand = Candidate(
                        gen_id=gen_id,
                        lane_id=lane_id,
                        score=judge_result.score or 0.0,
                        sharpe_test=judge_result.details.get("cand_test", {}).get("sharpe", 0.0) if judge_result.details else 0.0,
                        sharpe_train=judge_result.details.get("cand_train", {}).get("sharpe", 0.0) if judge_result.details else 0.0,
                        verdict=judge_result.verdict,
                        cost_usd=result.cost_usd,
                        duration_s=result.duration_s,
                    )
                    self.population.add(cand)

                    # Update UCB bandit
                    mutation_type = hypothesis.mutation_type
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
                    cand_test = judge_result.details.get("cand_test", {}) if judge_result.details else {}
                    cand_train = judge_result.details.get("cand_train", {}) if judge_result.details else {}
                    lane_candidates.append(LaneCandidate(
                        lane_id=lane_id, gen_id=gen_id,
                        score=judge_result.score or 0.0,
                        sharpe_test=cand_test.get("sharpe", 0.0),
                        sharpe_train=cand_train.get("sharpe", 0.0),
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

                    # Update champion baseline with promoted candidate
                    await self._update_champion_after_promotion(winner.gen_id)
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

    # ------------------------------------------------------------------
    # Post-promotion
    # ------------------------------------------------------------------

    async def _update_champion_after_promotion(self, gen_id: int) -> None:
        """After promotion, update champion metrics from the promoted candidate."""
        # Look for the promoted candidate's metrics in the generation dir
        gen_dir = self.config.generations_dir / f"gen_{gen_id:04d}"
        if not gen_dir.exists():
            return

        # Find the promoted lane's metrics
        for lane_dir in sorted(gen_dir.iterdir()):
            if not lane_dir.is_dir():
                continue
            test_metrics_file = lane_dir / "test_metrics.json"
            train_metrics_file = lane_dir / "train_metrics.json"
            if test_metrics_file.exists() and train_metrics_file.exists():
                cand_train = json.loads(train_metrics_file.read_text())
                cand_test = json.loads(test_metrics_file.read_text())
                self.champion_metrics["train"] = cand_train
                self.champion_metrics["test"] = cand_test
                self.champion_metrics["generation"] = gen_id

                # Also update the champion metrics.json
                metrics_path = self.config.champion_dir / "metrics.json"
                metrics_path.write_text(json.dumps(self.champion_metrics, indent=2, default=str))

                # Copy promoted strategy to champion dir
                promoted_strategy = lane_dir / "strategy.py"
                champion_strategy = self.config.champion_dir / "strategy.py"
                if promoted_strategy.exists():
                    shutil.copy2(promoted_strategy, champion_strategy)

                logger.info("Updated champion baseline from gen_%04d", gen_id)
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
