"""Holdout validation — run a LOCKED out-of-sample backtest for regime robustness.

Replaces: auto_evolve/scripts/holdout_validate.py
Config-driven: no hardcoded paths. Delegates BT execution to BacktestEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HoldoutResult:
    """Out-of-sample holdout validation result."""
    sharpe: float = 0.0
    cagr: float = 0.0
    max_drawdown_account: float = 0.0
    total_trades: int = 0
    winrate: float = 0.0
    timerange: str = ""
    passes: bool = False
    reason: str = ""


def check_holdout(
    holdout_metrics: dict[str, Any],
    champion_holdout_sharpe: float = 0.0,
    *,
    floor_ratio: float = 0.70,
    floor_slack: float = 0.05,
    timerange: str = "",
) -> HoldoutResult:
    """Validate holdout result against champion baseline.

    Pass criteria: holdout_sharpe >= champ_holdout_sharpe × floor_ratio − floor_slack
    """
    ho_sharpe = holdout_metrics.get("sharpe") or 0.0
    ho_cagr = holdout_metrics.get("cagr") or 0.0
    ho_dd = holdout_metrics.get("max_drawdown_account") or 0.0
    ho_trades = holdout_metrics.get("total_trades") or 0
    ho_wr = holdout_metrics.get("winrate") or 0.0

    holdout_floor = champion_holdout_sharpe * floor_ratio - floor_slack
    passes = ho_sharpe >= holdout_floor

    if passes:
        reason = f"holdout sharpe {ho_sharpe:+.3f} >= floor {holdout_floor:+.3f}"
    else:
        reason = f"holdout sharpe {ho_sharpe:+.3f} < floor {holdout_floor:+.3f} (regime-fragile)"

    return HoldoutResult(
        sharpe=ho_sharpe,
        cagr=ho_cagr,
        max_drawdown_account=ho_dd,
        total_trades=ho_trades,
        winrate=ho_wr,
        timerange=timerange,
        passes=passes,
        reason=reason,
    )
