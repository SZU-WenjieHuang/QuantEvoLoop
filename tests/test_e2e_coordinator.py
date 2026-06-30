"""End-to-end integration test for EvolutionCoordinator with MockEngine."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from quantevoloop.config import QuantEvoLoopConfig, BackendConfig, DataSplits
from quantevoloop.engine.mock_engine import MockBacktestEngine
from quantevoloop.gateway.coordinator import EvolutionCoordinator


class MockBackend:
    """Mock backend that simulates successful mutations."""

    def __init__(self):
        self._total_cost = 0.0
        self._total_calls = 0

    @property
    def total_cost(self):
        return self._total_cost

    @property
    def total_calls(self):
        return self._total_calls

    async def mutate_strategy(self, strategy_path, hypothesis, context=None):
        from quantevoloop.backends.base import MutationResult
        self._total_calls += 1
        cost = 0.01  # $0.01 per call
        self._total_cost += cost

        # Simulate: copy strategy to output_dir if context has output_dir
        if context and context.output_dir:
            output = context.output_dir / "strategy.py"
            output.write_text(
                f"# Mutated: {hypothesis}\nclass MutatedStrategy:\n    pass\n"
            )

        return MutationResult(
            success=True,
            modified_files=[str(strategy_path)],
            raw_output="mock mutation successful",
            cost_usd=cost,
            latency_seconds=0.5,
            strategy_path=strategy_path,
        )

    async def analyze_weakness(self, strategy_path, diagnostic):
        from quantevoloop.backends.base import AnalysisResult
        return AnalysisResult(directions=[], raw_output="mock analysis")

    async def judge_candidates(self, candidates, champion_metrics):
        from quantevoloop.backends.base import JudgeResult
        return JudgeResult(decision="reject", reasoning="mock judge")

    async def check_health(self):
        return True, "Mock backend: always healthy"


@pytest.fixture
def e2e_workspace(tmp_path):
    """Create a full workspace with champion strategy."""
    ws = tmp_path / "evo_workspace"
    for d in ["champion", "generations", "campaigns", "knowledge", "logs"]:
        (ws / d).mkdir(parents=True)

    # Create champion strategy
    strategy = ws / "champion" / "strategy.py"
    strategy.write_text(
        "class ChampionStrategy:\n"
        "    def populate_indicators(self, dataframe, metadata):\n"
        "        return dataframe\n"
    )

    # Create original strategy (same content)
    original = ws / "original_strategy.py"
    original.write_text(strategy.read_text())

    return ws


@pytest.fixture
def e2e_config(e2e_workspace):
    """Create config for E2E test."""
    config = QuantEvoLoopConfig(
        workspace_dir=e2e_workspace,
        strategy_path=e2e_workspace / "original_strategy.py",
        n_lanes=2,
        max_campaign_iter=3,
        data_splits=DataSplits(
            train_start="20220101",
            train_end="20240701",
            test_start="20240701",
            test_end="20260101",
            holdout_start="20210101",
            holdout_end="20220101",
        ),
    )
    config.ensure_dirs()
    return config


@pytest.mark.asyncio
async def test_coordinator_bootstrap(e2e_config):
    """Test that coordinator bootstraps champion baseline."""
    backend = MockBackend()
    engine = MockBacktestEngine(e2e_config, seed=42)
    coordinator = EvolutionCoordinator(e2e_config, backend, engine)

    # Run bootstrap
    await coordinator._bootstrap_champion()

    # Verify champion metrics were saved
    metrics_path = e2e_config.champion_dir / "metrics.json"
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "train" in metrics
    assert "test" in metrics
    assert metrics["train"].get("sharpe") is not None
    assert metrics["test"].get("sharpe") is not None


@pytest.mark.asyncio
async def test_coordinator_full_loop(e2e_config):
    """Test that coordinator runs a full evolution loop."""
    backend = MockBackend()
    engine = MockBacktestEngine(e2e_config, seed=42)
    coordinator = EvolutionCoordinator(e2e_config, backend, engine)

    # Run 2 generations
    state = await coordinator.run(max_generations=2)

    # Verify state
    assert state.generation >= 2
    assert state.status == "stopped"

    # Verify generations were created
    gen_dirs = list(e2e_config.generations_dir.glob("gen_*"))
    assert len(gen_dirs) >= 2

    # Verify state.json was saved
    state_file = e2e_config.state_file
    assert state_file.exists()


@pytest.mark.asyncio
async def test_coordinator_diagnose_and_hypothesize(e2e_config):
    """Test diagnosis + hypothesis generation."""
    backend = MockBackend()
    engine = MockBacktestEngine(e2e_config, seed=42)
    coordinator = EvolutionCoordinator(e2e_config, backend, engine)

    # Bootstrap first to populate trades
    await coordinator._bootstrap_champion()

    # Run diagnosis
    hypotheses = coordinator._diagnose_and_hypothesize()
    assert len(hypotheses) >= 1
    assert hypotheses[0].mutation_type  # Should have a mutation type


def test_generate_folds():
    """Test dynamic fold generation from config."""
    from quantevoloop.evaluation.walkforward import generate_folds

    folds = generate_folds("20220101", "20240701", n_folds=3)
    assert len(folds) == 3
    for start, end in folds:
        assert len(start) == 8  # YYYYMMDD format
        assert len(end) == 8
        assert start < end


def test_generate_folds_short_period():
    """Test fold generation with short period returns single fold."""
    from quantevoloop.evaluation.walkforward import generate_folds

    folds = generate_folds("20240101", "20240301", n_folds=3)
    assert len(folds) == 1  # Too short, falls back to single fold


def test_dead_ends_context_string():
    """Test DeadEndTracker.to_context_string()."""
    import tempfile
    from quantevoloop.evolution.dead_ends import DeadEndTracker, DeadEnd

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    tracker = DeadEndTracker(path)
    assert "No known dead ends" in tracker.to_context_string()

    tracker.add(DeadEnd(
        gen=1, mutation_type="EXIT_TIGHTEN",
        hypothesis_tag="MFE_giveback", failure_tag="risk-deteriorate",
        reason="max drawdown increased 40%",
    ))
    ctx = tracker.to_context_string()
    assert "EXIT_TIGHTEN" in ctx
    assert "MFE_giveback" in ctx

    path.unlink(missing_ok=True)
