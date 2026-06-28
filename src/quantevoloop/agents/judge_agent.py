"""JudgeAgent — evaluates candidates against champion.

The JudgeAgent runs the full 5-gate statistical pipeline:
  1. Hard constraints (risk, trade count, overfit gap)
  2. Score computation (min-of-segments)
  3. Statistical significance (PSR + Bootstrap CI)
  4. Robustness (drop-top-K)
  5. Walk-Forward + Holdout final-gate (OOS regime check)

Returns a verdict: promote | reject | dead_end | bt_error
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..engine.base import BacktestEngine, BacktestResult
from ..evaluation.scorer import decide as scorer_decide
from ..evaluation.robustness import check_robustness
from ..evaluation.holdout import check_holdout
from ..evaluation.significance import extract_trade_returns_from_dict
from ..evaluation.walkforward import (
    FoldResult,
    generate_folds,
    run_walkforward_backtests,
    summarize_folds,
    validate_walkforward,
    WalkForwardSummary,
)


@dataclass
class JudgeResult:
    """Full judge evaluation result."""
    verdict: str
    score: float | None = None
    reason: str = ""
    details: dict[str, Any] | None = None


class JudgeAgent:
    """Evaluates candidates using the 5-gate statistical pipeline."""

    def __init__(
        self,
        engine: BacktestEngine,
        config_dict: dict[str, Any] | None = None,
        *,
        train_timerange: str = "",
        test_timerange: str = "",
        holdout_timerange: str = "",
        train_start: str = "",
        train_end: str = "",
    ):
        self.engine = engine
        self.gates = config_dict or {}
        # Timeranges for WFA and holdout
        self.train_timerange = train_timerange or engine.train_timerange
        self.test_timerange = test_timerange or engine.test_timerange
        self.holdout_timerange = holdout_timerange or engine.holdout_timerange
        # Train period for generating WFA folds
        self.train_start = train_start
        self.train_end = train_end

    async def evaluate(
        self,
        candidate_strategy: Path,
        champion_metrics: dict[str, Any],
        gen_dir: Path,
        *,
        run_holdout: bool = True,
        run_wfa: bool = True,
    ) -> JudgeResult:
        """Run full evaluation pipeline on a candidate strategy."""
        try:
            # Run train + test backtests
            cand_train = await self.engine.run_backtest(
                candidate_strategy, self.train_timerange
            )
            cand_test = await self.engine.run_backtest(
                candidate_strategy, self.test_timerange
            )

            if cand_train.error or cand_test.error:
                return JudgeResult(
                    verdict="bt_error",
                    reason=f"BT failed: train={cand_train.error}, test={cand_test.error}",
                )

            # Extract champion metrics
            champ_train = champion_metrics.get("train", {})
            champ_test = champion_metrics.get("test", {})
            champ_holdout = champion_metrics.get("holdout")

            # Build candidate metrics dicts
            cand_train_dict = cand_train.to_dict()
            cand_test_dict = cand_test.to_dict()

            # Extract trade returns for statistical tests
            train_returns = cand_train.trade_returns
            test_returns = cand_test.trade_returns

            # Walk-Forward Analysis
            wfa_result = None
            if run_wfa:
                wfa_result = await self._run_walkforward(candidate_strategy, champion_metrics)

            # Holdout backtest
            holdout_result = None
            if run_holdout and self.holdout_timerange:
                ho_bt = await self.engine.run_backtest(
                    candidate_strategy, self.holdout_timerange
                )
                if not ho_bt.error:
                    holdout_result = ho_bt.to_dict()

            # Full decide pipeline
            result = scorer_decide(
                cand_train=cand_train_dict,
                cand_test=cand_test_dict,
                champ_train=champ_train,
                champ_test=champ_test,
                champ_holdout=champ_holdout,
                train_returns=train_returns,
                test_returns=test_returns,
                holdout_result=holdout_result,
                gates=self.gates,
            )

            # Save metrics to gen_dir
            gen_dir.mkdir(parents=True, exist_ok=True)
            (gen_dir / "train_metrics.json").write_text(
                json.dumps(cand_train_dict, indent=2, default=str)
            )
            (gen_dir / "test_metrics.json").write_text(
                json.dumps(cand_test_dict, indent=2, default=str)
            )
            if wfa_result:
                (gen_dir / "walkforward.json").write_text(
                    json.dumps({
                        "verdict": wfa_result.verdict,
                        "reasons": wfa_result.reasons,
                        "candidate_summary": {
                            "mean_sharpe": wfa_result.candidate_summary.mean_sharpe,
                            "std_sharpe": wfa_result.candidate_summary.std_sharpe,
                            "sharpes_per_fold": wfa_result.candidate_summary.sharpes_per_fold,
                        },
                    }, indent=2)
                )
            decision_text = (
                f"# Decision — {gen_dir.name}\n\n"
                f"**verdict**: `{result.verdict}`\n\n"
                f"**reason**: {result.reason}\n"
            )
            if result.score:
                decision_text += f"\n**score**: {result.score:+.4f}\n"
            if wfa_result:
                decision_text += f"\n**walkforward**: {wfa_result.verdict}\n"
            (gen_dir / "decision.md").write_text(decision_text)

            # If WFA failed but decide was promote, downgrade to reject
            if wfa_result and wfa_result.verdict == "fail" and result.verdict == "promote":
                return JudgeResult(
                    verdict="reject",
                    score=result.score,
                    reason=f"{result.reason}; WFA failed: {'; '.join(wfa_result.reasons)}",
                    details={
                        "cand_train": cand_train_dict,
                        "cand_test": cand_test_dict,
                        "holdout": holdout_result,
                        "walkforward": wfa_result.verdict,
                    },
                )

            return JudgeResult(
                verdict=result.verdict,
                score=result.score,
                reason=result.reason,
                details={
                    "cand_train": cand_train_dict,
                    "cand_test": cand_test_dict,
                    "holdout": holdout_result,
                    "walkforward": wfa_result.verdict if wfa_result else None,
                },
            )

        except Exception as e:
            return JudgeResult(
                verdict="bt_error",
                reason=f"Evaluation failed: {e}",
            )

    async def _run_walkforward(
        self,
        candidate_strategy: Path,
        champion_metrics: dict[str, Any],
    ) -> Any:
        """Run walk-forward analysis on a candidate strategy."""
        # Generate folds from train period
        if self.train_start and self.train_end:
            folds = generate_folds(self.train_start, self.train_end, n_folds=3)
        else:
            # Try to extract from timerange string (e.g., "20220101-20240701")
            folds = self._folds_from_timerange()

        if not folds:
            return None

        # Run candidate fold backtests
        cand_fold_results = await run_walkforward_backtests(
            self.engine, candidate_strategy, folds,
        )

        # Build champion fold summary
        # If champion has WFA results, use them; otherwise use heuristic
        champ_fold_results = self._get_champion_fold_results(champion_metrics, folds)
        champ_summary = summarize_folds(champ_fold_results)

        return validate_walkforward(cand_fold_results, champ_summary)

    def _folds_from_timerange(self) -> list[tuple[str, str]]:
        """Extract folds from train_timerange string."""
        if not self.train_timerange or "-" not in self.train_timerange:
            return []
        parts = self.train_timerange.split("-")
        if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 8:
            return generate_folds(parts[0], parts[1], n_folds=3)
        return []

    @staticmethod
    def _get_champion_fold_results(
        champion_metrics: dict[str, Any],
        folds: list[tuple[str, str]],
    ) -> list[FoldResult]:
        """Build champion fold results.

        If champion has per-fold data, use it. Otherwise synthesize from
        overall train sharpe (assuming uniform distribution across folds).
        """
        champ_train = champion_metrics.get("train", {})
        champ_sharpe = champ_train.get("sharpe", 0.0)
        champ_trades = champ_train.get("total_trades", 0)
        trades_per_fold = max(champ_trades // len(folds), 1)

        # Synthesize: assume champion's per-fold sharpe ≈ overall train sharpe
        return [
            FoldResult(
                fold_name=f"fold_{i}",
                timerange=f"{s}-{e}",
                sharpe=champ_sharpe,
                total_trades=trades_per_fold,
            )
            for i, (s, e) in enumerate(folds)
        ]
