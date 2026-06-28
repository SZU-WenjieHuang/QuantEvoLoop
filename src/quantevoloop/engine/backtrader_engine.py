"""Backtrader backtesting engine adapter (skeleton).

Usage:
    engine = BacktraderEngine(config_path="bt_config.json")
    result = await engine.run_backtest(strategy_path, timerange="20220101-20240701")

Requires: pip install backtrader
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from quantevoloop.engine.base import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class BacktraderEngine(BacktestEngine):
    """Backtesting engine using Backtrader.

    Note: This is a skeleton implementation. Full integration requires:
    1. Converting Freqtrade IStrategy to Backtrader Strategy format
    2. Implementing data feed from local CSV/Parquet files
    3. Mapping Backtrader analytics to BacktestResult fields
    """

    def __init__(self, config_path: Path | None = None, cash: float = 100_000.0):
        self.config_path = config_path
        self.cash = cash

    async def run_backtest(
        self,
        strategy_path: Path,
        timerange: str,
        config_path: Path | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        """Run a Backtrader backtest.

        TODO: Implement full Backtrader execution:
        1. Load strategy class from strategy_path
        2. Parse timerange into start/end dates
        3. Create Cerebro engine with data feeds
        4. Add analyzers (SharpeRatio, DrawDown, TradeAnalyzer)
        5. Run and extract metrics
        """
        logger.warning("BacktraderEngine is a skeleton — not yet implemented")
        return BacktestResult(
            error="BacktraderEngine not yet implemented. Use FreqtradeEngine.",
            timerange=timerange,
        )

    async def check_health(self) -> tuple[bool, str]:
        try:
            import backtrader
            return True, f"Backtrader: {backtrader.__version__}"
        except ImportError:
            return False, "Backtrader not installed (pip install backtrader)"

    def extract_metrics(self, result_dir: Path) -> dict[str, Any]:
        if result_dir.suffix == ".json" and result_dir.exists():
            return json.loads(result_dir.read_text())
        return {}
