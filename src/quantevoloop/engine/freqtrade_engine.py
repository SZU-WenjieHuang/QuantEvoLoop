"""Freqtrade backtesting engine implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from quantevoloop.engine.base import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class FreqtradeEngine(BacktestEngine):
    """Backtesting engine using Freqtrade CLI.

    Runs: python -m freqtrade backtesting --config <config> --strategy-path <path>
          --timerange <range> --export trades
    """

    def __init__(self, python_bin: str = "python", config_path: Path | None = None,
                 data_splits=None, freqtrade_config: str | None = None):
        self.python_bin = python_bin
        self.default_config = Path(config_path) if config_path else (Path(freqtrade_config) if freqtrade_config else None)
        if data_splits:
            self.train_timerange = data_splits.train_timerange
            self.test_timerange = data_splits.test_timerange
            self.holdout_timerange = data_splits.holdout_timerange

    async def run_backtest(
        self,
        strategy_path: Path,
        timerange: str,
        config_path: Path | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        cfg = config_path or self.default_config
        if cfg is None:
            return BacktestResult(error="No config path provided")

        strategy_name = kwargs.get("strategy_name", self._detect_strategy_class(strategy_path))
        export_dir = kwargs.get("export_dir", strategy_path.parent / "bt_results")
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python_bin, "-m", "freqtrade", "backtesting",
            "--config", str(cfg),
            "--strategy-path", str(strategy_path.parent),
            "--strategy", strategy_name,
            "--timerange", timerange,
            "--export", "trades",
            "--export-filename", str(export_dir / f"bt_{timerange}.json"),
            "--breakdown", "month",
        ]

        # Set offline mode if configured
        env = kwargs.get("env", {})

        logger.info("Running freqtrade backtest: %s", " ".join(cmd))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**dict(__import__("os").environ), **env},
            )
            stdout, stderr = await proc.communicate()
            rc = proc.returncode or 0
        except FileNotFoundError:
            return BacktestResult(error=f"Python not found: {self.python_bin}")

        if rc != 0:
            return BacktestResult(
                exit_code=rc,
                error=stderr.decode("utf-8", errors="replace")[:500],
                timerange=timerange,
            )

        # Parse results
        result_file = export_dir / f"bt_{timerange}.json"
        if not result_file.exists():
            return BacktestResult(
                exit_code=rc,
                error="Backtest completed but result file not found",
                timerange=timerange,
            )

        return self._parse_result(result_file, timerange, strategy_name)

    async def check_health(self) -> tuple[bool, str]:
        """Check if freqtrade is installed and accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.python_bin, "-m", "freqtrade", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                return True, f"Freqtrade: {stdout.decode().strip()}"
            return False, f"Freqtrade error: {stderr.decode().strip()[:200]}"
        except FileNotFoundError:
            return False, f"Python not found: {self.python_bin}"

    def extract_metrics(self, result_dir: Path) -> dict[str, Any]:
        """Extract metrics from a freqtrade backtest result JSON."""
        if result_dir.suffix == ".zip":
            return self._extract_from_zip(result_dir)
        if result_dir.suffix == ".json":
            with open(result_dir) as f:
                return json.load(f)
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_strategy_class(path: Path) -> str:
        """Detect the strategy class name from the .py file."""
        import ast

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it inherits from IStrategy
                for base in node.bases:
                    base_name = getattr(base, "id", "") or getattr(base, "attr", "")
                    if base_name == "IStrategy":
                        return node.name
        # Fallback: use filename
        return path.stem

    @staticmethod
    def _parse_result(result_file: Path, timerange: str, strategy_name: str) -> BacktestResult:
        """Parse freqtrade backtest result JSON into BacktestResult.

        Handles multiple freqtrade output formats:
        - Modern: {"strategy": {"StrategyName": {...}, "metadata": {...}}}
        - Flat:   {"strategy": {"sharpe": ..., "trades": [...]}}
        - Array:  {"strategy": [{"key": "StrategyName", ...}]}
        """
        try:
            with open(result_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return BacktestResult(error=f"Failed to parse result: {e}", timerange=timerange)

        # Locate strategy data — handle multiple freqtrade output formats
        strategy_data = data.get("strategy", {})

        if isinstance(strategy_data, dict):
            if strategy_name in strategy_data:
                strategy_data = strategy_data[strategy_name]
            elif "metadata" in strategy_data and len(strategy_data) == 2:
                for k, v in strategy_data.items():
                    if k != "metadata" and isinstance(v, dict):
                        strategy_data = v
                        break
        elif isinstance(strategy_data, list):
            for item in strategy_data:
                if isinstance(item, dict) and item.get("key") == strategy_name:
                    strategy_data = item
                    break
            else:
                strategy_data = strategy_data[0] if strategy_data else {}

        if not isinstance(strategy_data, dict):
            strategy_data = {}

        trades = strategy_data.get("trades", [])
        total_trades = len(trades)

        # Extract key metrics — try multiple field name variants
        def _get(d: dict, *keys: str, default: float = 0.0) -> float:
            for k in keys:
                v = d.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        continue
            return default

        sharpe = _get(strategy_data, "sharpe", "sharpe_ratio", "sharpe_ratio_1d")
        cagr = _get(strategy_data, "cagr", "compound_annual_growth_ratio",
                     "compound_annual_growth_ratio_1d")
        max_dd = _get(strategy_data, "max_drawdown", "max_drawdown_account", "max_drawdown_pct")
        win_rate = _get(strategy_data, "winrate", "win_rate")
        sortino = _get(strategy_data, "sortino", "sortino_ratio", "sortino_ratio_1d")
        calmar = _get(strategy_data, "calmar", "calmar_ratio", "calmar_ratio_1d")
        profit_factor = _get(strategy_data, "profit_factor", "profit_all_ratio")
        total_profit = _get(strategy_data, "profit_total", "profit_all_pct", "profit_total_abs")
        backtest_days = _get(strategy_data, "backtest_days", default=365.0)

        return BacktestResult(
            sharpe=sharpe,
            cagr=cagr,
            max_drawdown=abs(max_dd),
            max_drawdown_account=abs(max_dd),
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sortino=sortino,
            calmar=calmar,
            total_profit_pct=total_profit * 100 if abs(total_profit) < 10 else total_profit,
            backtest_days=backtest_days,
            trades=trades,
            raw_output=data,
            timerange=timerange,
            strategy_name=strategy_name,
        )

    @staticmethod
    def _extract_from_zip(zip_path: Path) -> dict[str, Any]:
        """Extract metrics from a freqtrade backtest zip file."""
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith(".json"):
                        with zf.open(name) as f:
                            return json.loads(f.read())
        except (zipfile.BadZipFile, json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to extract from zip %s: %s", zip_path, e)
        return {}
