"""Abstract base class for backtesting engines."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BacktestResult:
    """Standardized result from a backtest run."""

    # Core metrics
    sharpe: float = 0.0
    cagr: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_account: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Additional metrics
    avg_trade_duration: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    total_profit_pct: float = 0.0
    backtest_days: float = 0.0

    # Raw data
    trades: list[dict[str, Any]] = field(default_factory=list)
    raw_output: dict[str, Any] = field(default_factory=dict)

    # Metadata
    timerange: str = ""
    strategy_name: str = ""
    config_path: str = ""
    exit_code: int = 0
    error: str = ""

    @property
    def is_valid(self) -> bool:
        return self.exit_code == 0 and self.total_trades > 0 and not self.error

    @property
    def trade_returns(self) -> np.ndarray:
        """Per-trade profit_ratio as numpy array for statistical tests."""
        return np.array(
            [t.get("profit_ratio", 0.0) for t in self.trades],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to metrics dict compatible with scorer."""
        return {
            "sharpe": self.sharpe,
            "cagr": self.cagr,
            "max_drawdown_account": self.max_drawdown_account,
            "total_trades": self.total_trades,
            "winrate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "profit_total": self.total_profit_pct,
            "backtest_days": self.backtest_days,
        }


class BacktestEngine(ABC):
    """Abstract interface for backtesting engines.

    Freqtrade is the default implementation. Other engines (Backtrader, etc.)
    can be added by implementing this interface.
    """

    # Subclasses should set these from config
    train_timerange: str = ""
    test_timerange: str = ""
    holdout_timerange: str = ""

    @abstractmethod
    async def run_backtest(
        self,
        strategy_path: Path,
        timerange: str,
        config_path: Path | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        """Run a backtest and return standardized results.

        Args:
            strategy_path: Path to the strategy .py file.
            timerange: Timerange string (e.g., "20220101-20240701").
            config_path: Optional path to engine-specific config file.
            **kwargs: Engine-specific options.

        Returns:
            BacktestResult with standardized metrics.
        """

    @abstractmethod
    async def check_health(self) -> tuple[bool, str]:
        """Check if the engine is available and properly configured."""

    @abstractmethod
    def extract_metrics(self, result_dir: Path) -> dict[str, Any]:
        """Extract raw metrics from a backtest result directory/zip.

        Args:
            result_dir: Path to the backtest output directory or zip file.

        Returns:
            Dict of raw metrics as produced by the engine.
        """
