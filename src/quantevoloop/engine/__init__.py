"""Backtesting engine factory and base classes."""

from quantevoloop.engine.base import BacktestEngine, BacktestResult
from quantevoloop.engine.freqtrade_engine import FreqtradeEngine
from quantevoloop.engine.mock_engine import MockEngine
from quantevoloop.engine.backtrader_engine import BacktraderEngine
from quantevoloop.engine.zipline_engine import ZiplineEngine

__all__ = [
    "BacktestEngine", "BacktestResult",
    "FreqtradeEngine", "MockEngine",
    "BacktraderEngine", "ZiplineEngine",
]
