"""JudgeAgent — evaluates candidates against champion.

The JudgeAgent runs the full 5-gate statistical pipeline:
  1. Hard constraints (risk, trade count, overfit gap)
  2. Score computation (min-of-segments)
  3. Statistical significance (PSR + Bootstrap CI)
  4. Robustness (drop-top-K)
  5. Holdout final-gate (OOS regime check)

Returns a verdict: promote | reject | dead_end | bt_error
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine.base import BacktestEngine, BacktestResult
from ..evaluation.scorer import decide as scorer_decide
from ..evaluation.robustness import check_robustness
from ..evaluation.holdout import check_holdout
from ..evaluation.significance import extract_trade_returns_from_dict


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
    ):
        self.engine = engine
        self.gates = config_dict or {}

    async def evaluate(
        self,
        candidate_strategy: Path,
        champion_metrics: dict[str, Any],
        gen_dir: Path,
        *,
        run_holdout: bool = False,
    ) -> JudgeResult:
        """Run full evaluation pipeline on a candidate strategy."""
        try:
            # Run train + test backtests
            cand_train = await self.engine.run_backtest(
                candidate_strategy, self.engine.train_timerange
            )
            cand_test = await self.engine.run_backtest(
                candidate_strategy, self.engine.test_timerange
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

            # Full decide pipeline
            holdout_result = None
            if run_holdout:
                ho_bt = await self.engine.run_backtest(
                    candidate_strategy, self.engine.holdout_timerange
                )
                if not ho_bt.error:
                    holdout_result = ho_bt.to_dict()

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
            (gen_dir / "decision.md").write_text(
                f"# Decision — {gen_dir.name}\n\n"
                f"**verdict**: `{result.verdict}`\n\n"
                f"**reason**: {result.reason}\n\n"
                f"**score**: {result.score:+.4f}\n" if result.score else ""
            )

            return JudgeResult(
                verdict=result.verdict,
                score=result.score,
                reason=result.reason,
                details={
                    "cand_train": cand_train_dict,
                    "cand_test": cand_test_dict,
                    "holdout": holdout_result,
                },
            )

        except Exception as e:
            return JudgeResult(
                verdict="bt_error",
                reason=f"Evaluation failed: {e}",
            )
