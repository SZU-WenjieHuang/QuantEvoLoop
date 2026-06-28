"""Tests for the evaluation layer."""

import numpy as np
import pytest

from quantevoloop.evaluation.significance import (
    trade_sharpe, probabilistic_sharpe, bootstrap_sharpe_ci, deflated_sharpe,
)
from quantevoloop.evaluation.scorer import compute_score, check_hard_constraints, decide
from quantevoloop.evaluation.robustness import drop_top_k_sharpe, check_robustness
from quantevoloop.evaluation.diagnostics import diagnose, mfe_pct, mae_pct
from quantevoloop.evaluation.holdout import check_holdout
from quantevoloop.evaluation.walkforward import summarize_folds, validate_walkforward, FoldResult, WalkForwardSummary


class TestSignificance:
    def test_trade_sharpe_positive(self, sample_returns):
        sr = trade_sharpe(sample_returns)
        assert sr > 0, f"Expected positive Sharpe, got {sr}"

    def test_trade_sharpe_negative(self, negative_returns):
        sr = trade_sharpe(negative_returns)
        assert sr < 0, f"Expected negative Sharpe, got {sr}"

    def test_trade_sharpe_empty(self):
        sr = trade_sharpe(np.array([]))
        assert sr == 0.0

    def test_psr_positive_returns(self, sample_returns):
        result = probabilistic_sharpe(sample_returns)
        assert 0.0 <= result["psr"] <= 1.0
        assert result["n_trades"] == 100
        assert "sharpe_observed" in result

    def test_psr_insufficient_trades(self):
        result = probabilistic_sharpe(np.array([0.01]))
        assert result["psr"] == 0.0
        assert result.get("note") == "insufficient_trades"

    def test_bootstrap_ci(self, sample_returns):
        result = bootstrap_sharpe_ci(sample_returns, n_resamples=100, ci=0.90)
        assert result["sharpe_ci_lo"] <= result["sharpe_ci_hi"]
        assert result["n_resamples"] == 100

    def test_dsr(self, sample_returns):
        result = deflated_sharpe(sample_returns, n_trials=50)
        assert 0.0 <= result["dsr"] <= 1.0
        assert result["n_trials"] == 50


class TestScorer:
    def test_compute_score_positive(self):
        cand_train = {"sharpe": 1.0, "cagr": 0.2}
        cand_test = {"sharpe": 0.9, "cagr": 0.18, "max_drawdown_account": 0.10}
        champ_train = {"sharpe": 0.8, "cagr": 0.15}
        champ_test = {"sharpe": 0.7, "cagr": 0.12, "max_drawdown_account": 0.12}
        result = compute_score(cand_train, cand_test, champ_train, champ_test)
        assert result.score > 0

    def test_compute_score_negative(self):
        cand_train = {"sharpe": 0.5, "cagr": 0.05}
        cand_test = {"sharpe": 0.4, "cagr": 0.03, "max_drawdown_account": 0.25}
        champ_train = {"sharpe": 0.8, "cagr": 0.15}
        champ_test = {"sharpe": 0.7, "cagr": 0.12, "max_drawdown_account": 0.12}
        result = compute_score(cand_train, cand_test, champ_train, champ_test)
        assert result.score < 0

    def test_hard_constraints_pass(self):
        cand_train = {"sharpe": 0.8}
        cand_test = {"sharpe": 0.7, "total_trades": 100, "max_drawdown_account": 0.10}
        champ_train = {"sharpe": 0.7}
        champ_test = {"sharpe": 0.6, "max_drawdown_account": 0.12}
        result = check_hard_constraints(cand_train, cand_test, champ_train, champ_test)
        assert result.passes

    def test_hard_constraints_risk(self):
        cand_train = {"sharpe": 0.8}
        cand_test = {"sharpe": 0.7, "total_trades": 100, "max_drawdown_account": 0.30}
        champ_train = {"sharpe": 0.7}
        champ_test = {"sharpe": 0.6, "max_drawdown_account": 0.12}
        result = check_hard_constraints(cand_train, cand_test, champ_train, champ_test)
        assert not result.passes
        assert result.failure_tag == "risk-deteriorate"

    def test_hard_constraints_too_sparse(self):
        cand_test = {"sharpe": 0.7, "total_trades": 10, "max_drawdown_account": 0.10}
        result = check_hard_constraints({"sharpe": 0.7}, cand_test, {"sharpe": 0.6}, {"sharpe": 0.5, "max_drawdown_account": 0.12})
        assert not result.passes
        assert result.failure_tag == "too-sparse"


class TestRobustness:
    def test_drop_top_k(self, sample_returns):
        result = drop_top_k_sharpe(sample_returns, k=2, backtest_days=365.0)
        assert "retained_ratio" in result
        assert result["drop_k"] == 2
        assert result["kept_n"] == 98

    def test_drop_top_k_insufficient(self):
        result = drop_top_k_sharpe(np.array([0.01, 0.02]), k=2, backtest_days=365.0)
        assert result["note"] == "insufficient_trades"


class TestDiagnostics:
    def test_diagnose(self, sample_trades):
        report = diagnose(sample_trades, {"sharpe": 0.5, "cagr": 0.1}, "test")
        assert report.n_trades == 50
        assert report.segment == "test"
        assert report.by_exit_reason is not None
        assert report.by_direction is not None

    def test_mfe_pct(self):
        trade = {"open_rate": 100.0, "max_rate": 105.0, "is_short": False}
        assert mfe_pct(trade) == pytest.approx(0.05)

    def test_mae_pct(self):
        trade = {"open_rate": 100.0, "min_rate": 95.0, "is_short": False}
        assert mae_pct(trade) == pytest.approx(0.05)


class TestHoldout:
    def test_holdout_pass(self):
        result = check_holdout({"sharpe": 0.5}, champion_holdout_sharpe=0.4)
        assert result.passes

    def test_holdout_fail(self):
        result = check_holdout({"sharpe": 0.1}, champion_holdout_sharpe=0.4)
        assert not result.passes


class TestWalkForward:
    def test_validate_pass(self):
        folds = [
            FoldResult("fold1", "20220101-20230101", sharpe=0.8, total_trades=100),
            FoldResult("fold2", "20220701-20230701", sharpe=0.7, total_trades=90),
            FoldResult("fold3", "20230101-20240101", sharpe=0.9, total_trades=110),
        ]
        champ_summary = WalkForwardSummary(mean_sharpe=0.75, std_sharpe=0.1)
        result = validate_walkforward(folds, champ_summary)
        assert result.verdict == "pass"

    def test_validate_fail(self):
        folds = [
            FoldResult("fold1", "20220101-20230101", sharpe=0.3, total_trades=100),
            FoldResult("fold2", "20220701-20230701", sharpe=0.2, total_trades=90),
            FoldResult("fold3", "20230101-20240101", sharpe=0.1, total_trades=110),
        ]
        champ_summary = WalkForwardSummary(mean_sharpe=0.75, std_sharpe=0.1)
        result = validate_walkforward(folds, champ_summary)
        assert result.verdict == "fail"
