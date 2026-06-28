"""Robustness checks — drop-top-K concentration risk gate.

Replaces: significance.py drop_top_k_sharpe (extracted for clarity).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .significance import annualized_sharpe, trade_sharpe


def drop_top_k_sharpe(
    returns: np.ndarray,
    k: int,
    backtest_days: float,
) -> dict[str, Any]:
    """Sort trade returns descending, drop the top-k largest winners,
    recompute annualized Sharpe on the remainder.

    A healthy strategy survives losing its 2 biggest winners with most of
    its Sharpe intact. If the strategy collapses, PnL was carried by a few
    tail events and the OOS expectation is fragile.

    Returns dict with: drop_k, kept_n, drop_top_k_sharpe, pre_drop_sharpe,
    retained_ratio.
    """
    n = len(returns)
    if n <= k + 1:
        return {
            "drop_k": k,
            "kept_n": max(0, n - k),
            "drop_top_k_sharpe": 0.0,
            "pre_drop_sharpe": 0.0,
            "retained_ratio": 0.0,
            "note": "insufficient_trades",
        }

    pre = annualized_sharpe(returns, backtest_days)
    sorted_desc = np.sort(returns)[::-1]
    kept = sorted_desc[k:]
    # Re-derive trades_per_year from ORIGINAL N (preserves time density)
    trades_per_year = n / float(backtest_days) * 365.0
    post = trade_sharpe(kept) * np.sqrt(trades_per_year)

    return {
        "drop_k": int(k),
        "kept_n": int(len(kept)),
        "drop_top_k_sharpe": float(post),
        "pre_drop_sharpe": float(pre),
        "retained_ratio": float(post / pre) if abs(pre) > 1e-9 else 0.0,
    }


def check_robustness(
    train_returns: np.ndarray,
    test_returns: np.ndarray,
    train_days: float,
    test_days: float,
    k: int = 2,
    *,
    train_retained_min: float = 0.40,
    test_retained_min: float = 0.45,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool, str]:
    """Run drop-top-K on both segments. Returns (train_result, test_result, passes, reason)."""
    train_robust = drop_top_k_sharpe(train_returns, k, train_days) if len(train_returns) > k + 1 else None
    test_robust = drop_top_k_sharpe(test_returns, k, test_days) if len(test_returns) > k + 1 else None

    if train_robust and train_robust["retained_ratio"] < train_retained_min:
        return train_robust, test_robust, False, (
            f"drop_top_{k}_train_retained={train_robust['retained_ratio']:.3f} "
            f"< {train_retained_min}"
        )
    if test_robust and test_robust["retained_ratio"] < test_retained_min:
        return train_robust, test_robust, False, (
            f"drop_top_{k}_test_retained={test_robust['retained_ratio']:.3f} "
            f"< {test_retained_min}"
        )
    return train_robust, test_robust, True, "robust"
