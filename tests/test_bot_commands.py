"""Tests for IM bot command processor."""

import json
import pytest
from pathlib import Path

from quantevoloop.channels.bot_commands import CommandProcessor, CommandResult


@pytest.fixture
def cmd_workspace(tmp_path):
    """Create a workspace with state, metrics, trades, and gen_index."""
    workspace = tmp_path / "evo_workspace"
    workspace.mkdir()
    (workspace / "champion").mkdir()
    (workspace / "generations").mkdir()

    # state.json
    state = {
        "generation": 5,
        "status": "running",
        "champion_generation": 3,
        "champion_sharpe_test": 0.85,
        "total_promotions": 2,
        "total_rejects": 3,
        "total_dead_ends": 0,
        "consecutive_rejects": 1,
        "total_cost_usd": 12.50,
        "total_calls": 15,
        "last_updated": "2025-01-15T10:00:00+00:00",
    }
    (workspace / "state.json").write_text(json.dumps(state))

    # champion/metrics.json
    metrics = {
        "generation": 3,
        "train": {"sharpe": 0.9, "cagr": 0.18, "max_drawdown_account": 0.10,
                  "total_trades": 150, "winrate": 0.55},
        "test": {"sharpe": 0.85, "cagr": 0.14, "max_drawdown_account": 0.12,
                 "total_trades": 80, "winrate": 0.50},
        "holdout": {"sharpe": 0.45},
    }
    (workspace / "champion" / "metrics.json").write_text(json.dumps(metrics))

    # champion/trades.json
    trades = {
        "train": [{"profit_ratio": 0.01, "exit_reason": "roi"} for _ in range(50)],
        "test": [{"profit_ratio": 0.005, "exit_reason": "take_profit"} for _ in range(30)],
    }
    (workspace / "champion" / "trades.json").write_text(json.dumps(trades))

    # generations/gen_index.jsonl
    records = [
        {"gen": 1, "lane": 0, "status": "promoted", "score": 0.15, "hypothesis_tag": "EXIT_TIGHTEN"},
        {"gen": 1, "lane": 1, "status": "rejected", "score": -0.05, "hypothesis_tag": "NEW_FILTER"},
        {"gen": 2, "lane": 0, "status": "rejected", "score": -0.02, "hypothesis_tag": "PARAM_TUNE"},
        {"gen": 2, "lane": 1, "status": "rejected", "score": 0.01, "hypothesis_tag": "BOX_FILTER"},
        {"gen": 3, "lane": 0, "status": "promoted", "score": 0.10, "hypothesis_tag": "COOLDOWN"},
    ]
    gen_index = workspace / "generations" / "gen_index.jsonl"
    gen_index.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    return workspace


@pytest.fixture
def processor(cmd_workspace):
    return CommandProcessor(coordinator=None, workspace=cmd_workspace)


class TestHelpCommand:
    def test_help_lists_all_commands(self, processor):
        result = processor.process("/help")
        assert result.command == "help"
        assert "/status" in result.text
        assert "/pause" in result.text
        assert "/diagnose" in result.text
        assert "/history" in result.text


class TestStatusCommand:
    def test_status_shows_state(self, processor):
        result = processor.process("/status")
        assert result.command == "status"
        assert "running" in result.text
        assert "`5`" in result.text  # generation
        assert "$12.50" in result.text

    def test_status_no_state(self, tmp_path):
        processor = CommandProcessor(workspace=tmp_path / "nonexistent")
        result = processor.process("/status")
        assert "No state.json found" in result.text


class TestChampionCommand:
    def test_champion_metrics(self, processor):
        result = processor.process("/champion")
        assert result.command == "champion"
        assert "Train:" in result.text
        assert "Test:" in result.text
        assert "Holdout:" in result.text

    def test_champion_no_data(self, tmp_path):
        processor = CommandProcessor(workspace=tmp_path / "nonexistent")
        result = processor.process("/champion")
        assert "No champion metrics" in result.text


class TestHistoryCommand:
    def test_history_default(self, processor):
        result = processor.process("/history")
        assert result.command == "history"
        assert "Last 5 generations" in result.text
        assert "gen0001" in result.text

    def test_history_with_limit(self, processor):
        result = processor.process("/history 2")
        assert "Last 2 generations" in result.text

    def test_history_no_data(self, tmp_path):
        processor = CommandProcessor(workspace=tmp_path / "nonexistent")
        result = processor.process("/history")
        assert "No generation history" in result.text


class TestDiagnoseCommand:
    def test_diagnose_runs(self, processor):
        result = processor.process("/diagnose")
        assert result.command == "diagnose"
        # Should have some diagnosis output (may find weaknesses or not)
        assert "Champion Diagnosis" in result.text
        assert "30 trades" in result.text

    def test_diagnose_no_trades(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "champion").mkdir()
        processor = CommandProcessor(workspace=workspace)
        result = processor.process("/diagnose")
        assert "No champion trades found" in result.text


class TestPauseResume:
    def test_pause_sets_flag(self, processor):
        assert not processor.pause_requested
        result = processor.process("/pause")
        assert result.command == "pause"
        assert processor.pause_requested
        assert "Pause requested" in result.text

    def test_resume_clears_flag(self, processor):
        processor.process("/pause")
        assert processor.pause_requested
        result = processor.process("/resume")
        assert result.command == "resume"
        assert not processor.pause_requested


class TestUnknownCommand:
    def test_unknown_returns_error(self, processor):
        result = processor.process("/foobar")
        assert "Unknown command" in result.text
        assert "/help" in result.text


class TestBanditCommand:
    def test_bandit_no_data(self, processor):
        result = processor.process("/bandit")
        assert "No bandit statistics" in result.text or "UCB1 Bandit" in result.text
