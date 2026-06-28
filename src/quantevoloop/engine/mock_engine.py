"""Mock backtesting engine for testing without Freqtrade."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from quantevoloop.engine.base import BacktestEngine, BacktestResult


class MockBacktestEngine(BacktestEngine):
    """Mock engine that returns synthetic backtest results.

    Useful for testing the evolution loop without requiring a real
    Freqtrade installation or market data.
    """

    def __init__(self, config=None, seed: int = 42):
        self._rng = random.Random(seed)
        if config:
            self.train_timerange = config.data_splits.train_timerange
            self.test_timerange = config.data_splits.test_timerange
            self.holdout_timerange = config.data_splits.holdout_timerange

    async def run_backtest(
        self,
        strategy_path: Path,
        timerange: str,
        config_path: Path | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        # Generate synthetic metrics with some variance
        sharpe = self._rng.gauss(0.5, 0.3)
        cagr = self._rng.gauss(0.15, 0.1)
        max_dd = abs(self._rng.gauss(0.15, 0.05))
        trades = self._rng.randint(30, 200)
        win_rate = self._rng.uniform(0.35, 0.65)

        # Generate synthetic trade returns for statistical tests
        trade_returns_list = [self._rng.gauss(0.002, 0.015) for _ in range(trades)]

        return BacktestResult(
            sharpe=sharpe,
            cagr=cagr,
            max_drawdown=max_dd,
            max_drawdown_account=max_dd,
            total_trades=trades,
            win_rate=win_rate,
            profit_factor=self._rng.uniform(0.8, 2.0),
            sortino=sharpe * self._rng.uniform(0.8, 1.5),
            calmar=cagr / max_dd if max_dd > 0 else 0.0,
            total_profit_pct=cagr * 100 * self._rng.uniform(0.5, 1.5),
            backtest_days=365.0,
            timerange=timerange,
            strategy_name=strategy_path.stem,
            trades=[{"profit_ratio": r} for r in trade_returns_list],
        )

    async def check_health(self) -> tuple[bool, str]:
        return True, "Mock engine: always available"

    def extract_metrics(self, result_dir: Path) -> dict[str, Any]:
        return {"mock": True}


# Alias for backward compatibility
MockEngine = MockBacktestEngine
