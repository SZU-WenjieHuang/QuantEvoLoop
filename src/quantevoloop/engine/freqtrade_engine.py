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

    def __init__(self, python_bin: str = "python", config_path: Path | None = None):
        self.python_bin = python_bin
        self.default_config = config_path

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
        """Parse freqtrade backtest result JSON into BacktestResult."""
        try:
            with open(result_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            return BacktestResult(error=f"Failed to parse result: {e}", timerange=timerange)

        # Freqtrade result structure varies by version; handle gracefully
        strategy_data = data.get("strategy", {})
        if isinstance(strategy_data, dict) and strategy_name in strategy_data:
            strategy_data = strategy_data[strategy_name]

        trades = strategy_data.get("trades", [])
        total_trades = len(trades)

        # Extract key metrics
        sharpe = strategy_data.get("sharpe", 0.0) or 0.0
        cagr = strategy_data.get("cagr", 0.0) or 0.0
        max_dd = strategy_data.get("max_drawdown", 0.0) or 0.0
        win_rate = strategy_data.get("winrate", 0.0) or 0.0
        sortino = strategy_data.get("sortino", 0.0) or 0.0
        calmar = strategy_data.get("calmar", 0.0) or 0.0
        profit_factor = strategy_data.get("profit_factor", 0.0) or 0.0
        total_profit = strategy_data.get("profit_total", 0.0) or 0.0

        return BacktestResult(
            sharpe=sharpe,
            cagr=cagr,
            max_drawdown=abs(max_dd),
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sortino=sortino,
            calmar=calmar,
            total_profit_pct=total_profit * 100,
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
