"""Workspace management — hypothesis schema and workspace initialization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


# Hypothesis taxonomy (8 categories)
HYPOTHESIS_TAXONOMY = {
    "EXIT_TIGHTEN": {
        "description": "Tighten exit conditions (trailing stop, TP, dd_ratio)",
        "examples": ["Reduce trailing offset", "Tighten stage-2 TP ratio"],
    },
    "EXIT_LOOSEN": {
        "description": "Loosen exit conditions (wider stops, longer holding)",
        "examples": ["Widen ATR stop multiplier", "Increase TIME_STOP_HOURS"],
    },
    "NEW_FILTER": {
        "description": "Add entry filter (volatility, regime, ADX)",
        "examples": ["Add Choppiness Index filter", "Add volume confirmation"],
    },
    "BOX_FILTER": {
        "description": "Directional filter (long-only, short-only, side bias)",
        "examples": ["Disable short entries in uptrend", "Add directional confirmation"],
    },
    "COOLDOWN": {
        "description": "Post-trade cooldown (reduce overtrading)",
        "examples": ["Increase COOLDOWN_BARS", "Add post-loss delay"],
    },
    "PARAM_TUNE": {
        "description": "Parameter tuning (indicators, thresholds)",
        "examples": ["Adjust EMA periods", "Tune RSI thresholds"],
    },
    "ENTRY_REFINE": {
        "description": "Entry signal refinement (timing, confirmation)",
        "examples": ["Add candle pattern filter", "Require multi-TF confirmation"],
    },
    "GENERAL_TUNE": {
        "description": "General strategy tuning (no specific weakness)",
        "examples": ["Review and optimize all parameters"],
    },
}


def init_workspace(
    workspace_dir: Path,
    strategy_path: Path,
    config_dict: dict[str, Any] | None = None,
) -> Path:
    """Initialize a new evolution workspace.

    Creates directory structure and seeds the champion strategy.
    Returns the champion strategy path.
    """
    dirs = [
        workspace_dir / "champion",
        workspace_dir / "generations",
        workspace_dir / "campaigns",
        workspace_dir / "knowledge",
        workspace_dir / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Seed champion strategy
    champion_strategy = workspace_dir / "champion" / "strategy.py"
    if strategy_path.exists():
        shutil.copy2(strategy_path, champion_strategy)

    # Write workspace metadata
    meta = {
        "workspace_version": "0.1.0",
        "strategy_source": str(strategy_path),
        "taxonomy": HYPOTHESIS_TAXONOMY,
    }
    (workspace_dir / "workspace.json").write_text(json.dumps(meta, indent=2))

    return champion_strategy
