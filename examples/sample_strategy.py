"""Sample Freqtrade strategy for QuantEvoLoop e2e testing.

This is a minimal trend-following strategy using RSI + EMA crossover.
It is intentionally simple so that the evolutionary loop can improve it.

Usage with QuantEvoLoop:
    quantevoloop init --strategy examples/sample_strategy.py --backend claude-code
    quantevoloop run --engine freqtrade --max-gens 20
"""

from datetime import datetime

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class SampleStrategy(IStrategy):
    """Minimal EMA + RSI trend-following strategy."""

    INTERFACE_VERSION = 3

    # Strategy parameters (evolvable)
    minimal_roi = {
        "0": 0.05,
        "360": 0.02,
        "720": 0.01,
    }

    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    timeframe = "1h"

    # EMA periods
    ema_fast = 20
    ema_slow = 50

    # RSI thresholds
    rsi_period = 14
    rsi_oversold = 35
    rsi_overbought = 70

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add technical indicators."""
        # EMA crossover
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)

        # ATR for volatility filter
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Volume filter
        dataframe["volume_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry conditions."""
        dataframe.loc[
            (
                (dataframe["ema_fast"] > dataframe["ema_slow"]) &
                (dataframe["rsi"] > self.rsi_oversold) &
                (dataframe["rsi"] < self.rsi_overbought) &
                (dataframe["volume"] > dataframe["volume_sma"] * 0.8) &
                (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions."""
        dataframe.loc[
            (
                (dataframe["ema_fast"] < dataframe["ema_slow"]) |
                (dataframe["rsi"] > self.rsi_overbought)
            ),
            "exit_long",
        ] = 1
        return dataframe
