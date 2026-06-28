"""Walk-forward validation — three-fold rolling walk-forward gate.

Replaces: auto_evolve/scripts/walkforward.py
Config-driven: no hardcoded paths. Uses BacktestEngine abstraction.
"""

from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FoldResult:
    fold_name: str
    timerange: str
    sharpe: float = 0.0
    total_trades: int = 0
    cagr: float = 0.0
    error: str = ""
    duration_s: float = 0.0


@dataclass
class WalkForwardSummary:
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    sharpes_per_fold: list[float] = field(default_factory=list)
    n_trades_per_fold: list[int] = field(default_factory=list)
    n_failed_folds: int = 0


# Default three folds (within train range 2022-01-01 → 2024-07-01)
DEFAULT_FOLDS = [
    ("fold1", "20220101-20230101"),
    ("fold2", "20220701-20230701"),
    ("fold3", "20230101-20240101"),
]

PASS_STD_MULTIPLIER = 1.5
PASS_MEAN_TOLERANCE = 0.95  # allows 5% cross-fold loss


def generate_folds(
    train_start: str,
    train_end: str,
    n_folds: int = 3,
    overlap_months: int = 6,
) -> list[tuple[str, str]]:
    """Generate overlapping rolling-window fold timeranges from train period.

    Example (train 20220101-20240701, 3 folds, 6-month overlap):
      fold1: 20220101-20230101
      fold2: 20220701-20230701
      fold3: 20230101-20240101
    """
    start = dt.datetime.strptime(train_start, "%Y%m%d")
    end = dt.datetime.strptime(train_end, "%Y%m%d")
    total_months = (end.year - start.year) * 12 + (end.month - start.month)

    if n_folds <= 1 or total_months < 6:
        return [(train_start, train_end)]

    # Each fold covers ~half the train period
    fold_span_months = max(total_months // 2, 6)
    step_months = max((total_months - fold_span_months) // max(n_folds - 1, 1), 1)

    folds = []
    for i in range(n_folds):
        fold_start = start + dt.timedelta(days=30 * step_months * i)
        fold_end = fold_start + dt.timedelta(days=30 * fold_span_months)
        if fold_end > end:
            fold_end = end
        if fold_start >= end:
            break
        folds.append((
            fold_start.strftime("%Y%m%d"),
            fold_end.strftime("%Y%m%d"),
        ))

    return folds or [(train_start, train_end)]


@dataclass
class WalkForwardResult:
    verdict: str  # pass | fail | skipped
    reasons: list[str] = field(default_factory=list)
    candidate_folds: list[FoldResult] = field(default_factory=list)
    candidate_summary: WalkForwardSummary = field(default_factory=WalkForwardSummary)
    champion_summary: WalkForwardSummary | None = None


def summarize_folds(folds: list[FoldResult]) -> WalkForwardSummary:
    sharpes = [f.sharpe for f in folds if not f.error]
    n_trades = [f.total_trades for f in folds if not f.error]
    if not sharpes:
        return WalkForwardSummary(n_failed_folds=len(folds))
    return WalkForwardSummary(
        mean_sharpe=statistics.mean(sharpes),
        std_sharpe=statistics.stdev(sharpes) if len(sharpes) > 1 else 0.0,
        sharpes_per_fold=sharpes,
        n_trades_per_fold=n_trades,
        n_failed_folds=len(folds) - len(sharpes),
    )


def validate_walkforward(
    candidate_folds: list[FoldResult],
    champion_summary: WalkForwardSummary,
    *,
    pass_mean_tolerance: float = PASS_MEAN_TOLERANCE,
    pass_std_multiplier: float = PASS_STD_MULTIPLIER,
) -> WalkForwardResult:
    """Compare candidate's per-fold sharpe distribution to champion's.

    Pass criteria:
      mean(cand_sharpe) >= mean(champ_sharpe) × tolerance
      AND std(cand_sharpe) <= std(champ_sharpe) × multiplier (with ABS floor 0.05)
      AND no failed folds
    """
    cand_summary = summarize_folds(candidate_folds)

    mean_floor = champion_summary.mean_sharpe * pass_mean_tolerance
    mean_ok = cand_summary.mean_sharpe >= mean_floor

    std_ceiling = max(champion_summary.std_sharpe * pass_std_multiplier, 0.05)
    std_ok = cand_summary.std_sharpe <= std_ceiling

    n_failed_ok = cand_summary.n_failed_folds == 0

    reasons = []
    if not mean_ok:
        reasons.append(
            f"mean_sharpe {cand_summary.mean_sharpe:.3f} < {mean_floor:.3f} "
            f"(champion {champion_summary.mean_sharpe:.3f} × {pass_mean_tolerance})"
        )
    if not std_ok:
        reasons.append(
            f"std_sharpe {cand_summary.std_sharpe:.3f} > {std_ceiling:.3f} "
            f"(champion×{pass_std_multiplier})"
        )
    if not n_failed_ok:
        reasons.append(f"{cand_summary.n_failed_folds} folds errored")

    pass_all = mean_ok and std_ok and n_failed_ok
    return WalkForwardResult(
        verdict="pass" if pass_all else "fail",
        reasons=reasons,
        candidate_folds=candidate_folds,
        candidate_summary=cand_summary,
        champion_summary=champion_summary,
    )


async def run_walkforward_backtests(
    engine: Any,  # BacktestEngine
    strategy_path: Path,
    folds: list[tuple[str, str]],
) -> list[FoldResult]:
    """Run backtests for each fold and return FoldResults.

    Args:
        engine: BacktestEngine instance.
        strategy_path: Path to the strategy .py file.
        folds: List of (fold_start, fold_end) timerange tuples.

    Returns:
        List of FoldResult for each fold.
    """
    results = []
    for fold_idx, (fold_start, fold_end) in enumerate(folds):
        timerange = f"{fold_start}-{fold_end}"
        fold_name = f"fold_{fold_idx}"
        try:
            bt = await engine.run_backtest(strategy_path, timerange)
            results.append(FoldResult(
                fold_name=fold_name,
                timerange=timerange,
                sharpe=bt.sharpe,
                total_trades=bt.total_trades,
                cagr=bt.cagr,
                error=bt.error,
            ))
        except Exception as e:
            results.append(FoldResult(
                fold_name=fold_name,
                timerange=timerange,
                error=str(e),
            ))
    return results
