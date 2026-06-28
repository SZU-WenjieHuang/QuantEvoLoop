"""Tests for evolution + selection + integration layers."""

import json
import pytest
from pathlib import Path

from quantevoloop.evolution.state import EvolutionState, GenerationRecord, GenerationIndex
from quantevoloop.evolution.campaign import Campaign, CampaignStatus, CampaignLesson
from quantevoloop.evolution.knowledge import KnowledgeBase, MutationRecord
from quantevoloop.evolution.dead_ends import DeadEndTracker, DeadEnd
from quantevoloop.evolution.promoter import Promoter
from quantevoloop.selection.ucb import UCBBandit
from quantevoloop.selection.tournament import Tournament, LaneCandidate
from quantevoloop.selection.population import Population, Candidate
from quantevoloop.selection.reward import compute_reward


class TestEvolutionState:
    def test_save_load(self, tmp_path):
        state = EvolutionState(generation=5, total_cost_usd=1.23)
        state.save(tmp_path / "state.json")
        loaded = EvolutionState.load(tmp_path / "state.json")
        assert loaded.generation == 5
        assert loaded.total_cost_usd == 1.23

    def test_advance_generation(self):
        state = EvolutionState()
        assert state.advance_generation() == 1
        assert state.advance_generation() == 2

    def test_record_promotion(self):
        state = EvolutionState(consecutive_rejects=5)
        state.record_promotion()
        assert state.total_promotions == 1
        assert state.consecutive_rejects == 0

    def test_gen_index(self, tmp_path):
        idx = GenerationIndex(tmp_path / "gen_index.jsonl")
        idx.append(GenerationRecord(gen=1, lane=0, status="promoted", score=0.5))
        idx.append(GenerationRecord(gen=2, lane=0, status="rejected"))
        rows = idx.load_all()
        assert len(rows) == 2
        assert rows[0]["status"] == "promoted"


class TestCampaign:
    def test_lifecycle(self, tmp_path):
        campaign = Campaign(campaign_id=1, name="test", weakness_direction="exit")
        assert campaign.status == CampaignStatus.PLANNING
        campaign.advance()
        assert campaign.status == CampaignStatus.ACTIVE
        assert campaign.current_iteration == 1
        assert not campaign.is_exhausted()

        campaign.max_iterations = 1
        assert campaign.is_exhausted()

        campaign.save(tmp_path / "campaign.json")
        loaded = Campaign.load(tmp_path / "campaign.json")
        assert loaded.campaign_id == 1


class TestKnowledge:
    def test_record_and_query(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "knowledge.json")
        kb.record(MutationRecord(
            mutation_type="EXIT_TIGHTEN", hypothesis_tag="test",
            weakness_direction="MFE_giveback", verdict="promoted",
            score=0.5, generation=1,
        ))
        kb.save()
        assert len(kb.get_high_ev()) == 0  # needs >= 3 attempts
        prior = kb.get_prior("EXIT_TIGHTEN", "MFE_giveback")
        assert prior is not None
        assert prior.total_attempts == 1

    def test_context_string(self, tmp_path):
        kb = KnowledgeBase(tmp_path / "knowledge.json")
        ctx = kb.to_context_string()
        assert "No prior knowledge" in ctx


class TestDeadEnds:
    def test_add_and_dedup(self, tmp_path):
        tracker = DeadEndTracker(tmp_path / "dead_ends.json")
        de = DeadEnd(gen=1, mutation_type="NEW_FILTER", hypothesis_tag="adx",
                     failure_tag="too-sparse", reason="trades < 50")
        assert tracker.add(de)
        assert not tracker.add(de)  # duplicate
        assert tracker.count() == 1
        assert tracker.is_known_dead_end("NEW_FILTER", "adx")


class TestUCBBandit:
    def test_explores_all_arms(self):
        bandit = UCBBandit(["A", "B", "C"])
        selected = set()
        for _ in range(3):
            arm = bandit.select()
            bandit.update(arm, 0.5)
            selected.add(arm)
        assert len(selected) == 3

    def test_exploits_best_arm(self):
        bandit = UCBBandit(["A", "B"])
        # Pre-fill: A is much better than B
        for _ in range(10):
            bandit.update("A", 1.0)
            bandit.update("B", -0.5)
        # Now A should be selected most of the time
        a_count = sum(1 for _ in range(100) if bandit.select() == "A")
        assert a_count > 50


class TestTournament:
    def test_selects_best(self):
        t = Tournament(n_lanes=3)
        candidates = [
            LaneCandidate(lane_id=0, gen_id=1, score=0.5, sharpe_test=0.8, sharpe_train=0.7, verdict="promoted"),
            LaneCandidate(lane_id=1, gen_id=1, score=0.3, sharpe_test=0.6, sharpe_train=0.5, verdict="promoted"),
            LaneCandidate(lane_id=2, gen_id=1, score=-0.1, sharpe_test=0.3, sharpe_train=0.2, verdict="rejected"),
        ]
        winner = t.add_round(candidates)
        assert winner is not None
        assert winner.lane_id == 0

    def test_no_winner(self):
        t = Tournament(n_lanes=2)
        candidates = [
            LaneCandidate(lane_id=0, gen_id=1, score=-0.1, sharpe_test=0.3, sharpe_train=0.2, verdict="rejected"),
            LaneCandidate(lane_id=1, gen_id=1, score=-0.2, sharpe_test=0.2, sharpe_train=0.1, verdict="rejected"),
        ]
        assert t.add_round(candidates) is None


class TestReward:
    def test_promoted(self):
        assert compute_reward("promoted", 0.5) > 0

    def test_dead_end(self):
        assert compute_reward("dead_end") < 0

    def test_bt_error(self):
        assert compute_reward("bt-error") == -1.0


class TestPopulation:
    def test_basic(self):
        pop = Population()
        pop.add(Candidate(gen_id=1, lane_id=0, score=0.5, sharpe_test=0.8, sharpe_train=0.7, verdict="promoted"))
        pop.add(Candidate(gen_id=2, lane_id=0, score=-0.1, sharpe_test=0.3, sharpe_train=0.2, verdict="rejected"))
        assert pop.size() == 2
        assert pop.best() is not None
        assert pop.best().gen_id == 1


class TestPromoter:
    def test_promote(self, tmp_workspace):
        gen_dir = tmp_workspace / "generations" / "gen_0001"
        gen_dir.mkdir(parents=True)
        (gen_dir / "strategy.py").write_text("class S:\n    pass\n")

        promoter = Promoter(tmp_workspace / "champion", tmp_workspace / "generations")
        result = promoter.promote(
            gen_id=1, score=0.5,
            cand_train={"sharpe": 0.9}, cand_test={"sharpe": 0.8},
        )
        assert result.success
        assert result.new_champion_gen == 1
        assert (tmp_workspace / "champion" / "strategy.py").exists()
        assert (tmp_workspace / "champion" / "metrics.json").exists()


class TestIntegration:
    """End-to-end integration test with mock engine."""

    @pytest.mark.asyncio
    async def test_mock_engine(self):
        from quantevoloop.engine.mock_engine import MockBacktestEngine
        engine = MockBacktestEngine(seed=42)
        result = await engine.run_backtest(
            strategy_path=Path("/tmp/dummy.py"),
            timerange="20220101-20240701",
        )
        assert result.is_valid
        assert result.total_trades > 0
        assert len(result.trade_returns) == result.total_trades
