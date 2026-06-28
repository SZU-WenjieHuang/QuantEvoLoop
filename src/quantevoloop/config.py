"""Pydantic configuration model for QuantEvoLoop.

Replaces all hardcoded paths (ROOT, EVOLVE, PY_BIN, CONFIG, etc.)
from the original auto_evolve scripts with a single, validated config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DataSplits(BaseModel):
    """Train / Test / Holdout time ranges.

    Default split follows the original auto_evolve convention:
    - Holdout:  2021 (full year, locked, used only for final gate)
    - Train:    2022-01-01 → 2024-07-01
    - Test:     2024-07-01 → 2026-01-01
    """

    holdout_start: str = "20210101"
    holdout_end: str = "20220101"
    train_start: str = "20220101"
    train_end: str = "20240701"
    test_start: str = "20240701"
    test_end: str = "20260101"

    @property
    def train_timerange(self) -> str:
        return f"{self.train_start}-{self.train_end}"

    @property
    def test_timerange(self) -> str:
        return f"{self.test_start}-{self.test_end}"

    @property
    def holdout_timerange(self) -> str:
        return f"{self.holdout_start}-{self.holdout_end}"


class ScoreWeights(BaseModel):
    """Composite score weights: 0.5·sharpe + 0.3·cagr − 0.1·max_dd."""

    sharpe: float = 0.5
    cagr: float = 0.3
    max_dd: float = 0.1


class StatisticalGates(BaseModel):
    """Thresholds for the 5-layer statistical validation pipeline."""

    # Test segment gates
    psr_min: float = 0.85
    bootstrap_ci: float = 0.70

    # Train segment gates (slightly looser)
    psr_min_train: float = 0.80
    bootstrap_ci_train: float = 0.65

    # Hard-constraint ceilings
    max_dd_ceiling_factor: float = 1.25  # candidate DD ≤ champion DD × factor
    min_trades: int = 50
    max_directional_gap: float = 0.6  # |test_sharpe − train_sharpe|
    holdout_degradation_factor: float = 0.7
    holdout_degradation_offset: float = 0.05


class BackendConfig(BaseModel):
    """Code Agent backend configuration.

    One of: claude-code, codex, qoder-cli.
    These CLIs provide full Agent Loop capabilities (ReAct, AST understanding,
    error recovery, context management) — far beyond any self-built solution.
    """

    type: Literal["claude-code", "codex", "qoder-cli"] = "claude-code"
    cli_path: str = "claude"
    max_turns: int = 15
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["Read", "Edit", "Bash"],
    )
    extra_args: list[str] = Field(default_factory=list)
    timeout_seconds: int = 300  # per-call timeout

    @field_validator("cli_path", mode="before")
    @classmethod
    def resolve_cli_path(cls, v: str, info) -> str:  # noqa: N805
        """Auto-detect CLI path based on backend type."""
        defaults = {
            "claude-code": "claude",
            "codex": "codex",
            "qoder-cli": "qodercli",
        }
        if v == "auto":
            backend_type = info.data.get("type", "claude-code")
            return defaults.get(backend_type, "claude")
        return v


class IMConfig(BaseModel):
    """Instant Messaging channel configuration."""

    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Discord
    discord_enabled: bool = False
    discord_bot_token: str = ""
    discord_channel_id: str = ""

    # Webhook (generic — works with DingTalk, Feishu, WeChat Work, etc.)
    webhook_enabled: bool = False
    webhook_url: str = ""

    # Notification levels: critical, important, info, debug
    min_notify_level: Literal["critical", "important", "info", "debug"] = "important"


class CostConfig(BaseModel):
    """Cost tracking and budget control."""

    budget_usd: float | None = None  # None = unlimited
    warn_at_pct: float = 0.80  # warn when 80% of budget consumed
    track_per_generation: bool = True
    cost_log_path: str = "cost_log.json"  # relative to workspace


class CheckpointConfig(BaseModel):
    """Checkpoint / resume configuration."""

    enabled: bool = True
    checkpoint_file: str = "checkpoint.json"  # relative to workspace
    auto_save_every: int = 1  # save every N generations


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

class QuantEvoLoopConfig(BaseModel):
    """Top-level configuration for QuantEvoLoop.

    Usage::

        config = QuantEvoLoopConfig.from_yaml("evoloop.yaml")
        # or
        config = QuantEvoLoopConfig(
            workspace_dir="./workspace",
            strategy_path="./strategies/my_strategy.py",
            ...
        )
    """

    # --- Paths ---
    workspace_dir: Path = Field(
        description="Root directory for all evolution artifacts (generations, campaigns, etc.)",
    )
    strategy_path: Path = Field(
        description="Path to the Freqtrade strategy .py file to evolve",
    )
    python_bin: Path = Field(
        default=Path("python"),
        description="Python interpreter to use for freqtrade backtesting",
    )
    backtest_config: Path | None = Field(
        default=None,
        description="Path to Freqtrade config.json for backtesting",
    )

    # --- Data ---
    data_splits: DataSplits = Field(default_factory=DataSplits)

    # --- Scoring ---
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)
    statistical_gates: StatisticalGates = Field(default_factory=StatisticalGates)

    # --- Evolution ---
    n_lanes: int = Field(default=3, ge=1, le=16)
    max_campaign_iter: int = Field(default=20, ge=1)
    max_total_generations: int | None = Field(default=None, description="None = unlimited")
    promotion_threshold: float = Field(
        default=0.05,
        description="Minimum score improvement to consider promotion",
    )

    # --- Backend ---
    backend: BackendConfig = Field(default_factory=BackendConfig)

    # --- IM ---
    im: IMConfig = Field(default_factory=IMConfig)

    # --- Cost ---
    cost: CostConfig = Field(default_factory=CostConfig)

    # --- Checkpoint ---
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # --- Derived properties ---

    @property
    def champion_dir(self) -> Path:
        return self.workspace_dir / "champion"

    @property
    def generations_dir(self) -> Path:
        return self.workspace_dir / "generations"

    @property
    def campaigns_dir(self) -> Path:
        return self.workspace_dir / "campaigns"

    @property
    def knowledge_dir(self) -> Path:
        return self.workspace_dir / "knowledge"

    @property
    def logs_dir(self) -> Path:
        return self.workspace_dir / "logs"

    @property
    def state_file(self) -> Path:
        return self.workspace_dir / "state.json"

    @property
    def dead_ends_file(self) -> Path:
        return self.workspace_dir / "dead_ends.md"

    @property
    def cost_log_path(self) -> Path:
        return self.workspace_dir / self.cost.cost_log_path

    @property
    def checkpoint_path(self) -> Path:
        return self.workspace_dir / self.checkpoint.checkpoint_file

    # --- Class methods ---

    @classmethod
    def from_yaml(cls, path: str | Path) -> QuantEvoLoopConfig:
        """Load config from a YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Expand env vars in string values
        data = _expand_env(data)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False, sort_keys=False)

    def ensure_dirs(self) -> None:
        """Create all required workspace directories."""
        for d in [
            self.workspace_dir,
            self.champion_dir,
            self.generations_dir,
            self.campaigns_dir,
            self.knowledge_dir,
            self.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_env(obj):
    """Recursively expand environment variables in string values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj
