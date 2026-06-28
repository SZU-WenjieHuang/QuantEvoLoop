"""Shared test fixtures."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json


@pytest.fixture
def sample_returns():
    """100 synthetic trade returns with positive Sharpe."""
    rng = np.random.default_rng(42)
    return rng.normal(0.002, 0.015, size=100)


@pytest.fixture
def negative_returns():
    """100 synthetic trade returns with negative Sharpe."""
    rng = np.random.default_rng(42)
    return rng.normal(-0.005, 0.02, size=100)


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace directory structure."""
    dirs = ["champion", "generations", "campaigns", "knowledge", "logs"]
    for d in dirs:
        (tmp_path / d).mkdir()

    # Create a dummy champion strategy
    strategy = tmp_path / "champion" / "strategy.py"
    strategy.write_text("class MyStrategy:\n    pass\n")

    # Create dummy champion metrics
    metrics = {
        "generation": 0,
        "train": {"sharpe": 0.8, "cagr": 0.15, "max_drawdown_account": 0.12,
                  "total_trades": 150, "winrate": 0.52},
        "test": {"sharpe": 0.6, "cagr": 0.10, "max_drawdown_account": 0.15,
                 "total_trades": 80, "winrate": 0.48},
        "holdout": {"sharpe": 0.4},
    }
    (tmp_path / "champion" / "metrics.json").write_text(json.dumps(metrics))

    return tmp_path


@pytest.fixture
def sample_trades():
    """Generate 50 sample trades for diagnostics."""
    rng = np.random.default_rng(42)
    trades = []
    for i in range(50):
        profit_ratio = float(rng.normal(0.002, 0.02))
        trades.append({
            "open_date": f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "close_date": f"2023-{(i % 12) + 1:02d}-{min((i % 28) + 3, 28):02d}",
            "is_short": i % 3 == 0,
            "profit_abs": profit_ratio * 1000,
            "profit_ratio": profit_ratio,
            "exit_reason": ["take_profit", "stop_loss", "trailing_stop", "roi"][i % 4],
            "trade_duration": int(rng.integers(60, 10080)),
            "open_rate": 50000.0,
            "max_rate": 50000.0 * (1 + abs(profit_ratio)),
            "min_rate": 50000.0 * (1 - abs(profit_ratio)),
            "funding_fees": float(rng.normal(-1.0, 0.5)),
        })
    return trades
