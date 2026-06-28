"""Abstract base class for Code Agent backends.

All backends (Claude Code, Codex, Qoder CLI) implement this interface.
The orchestrator only interacts with backends through this ABC.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quantevoloop.config import BackendConfig

logger = logging.getLogger(__name__)


@dataclass
class MutationResult:
    """Result of a strategy mutation attempt."""

    success: bool
    modified_files: list[str] = field(default_factory=list)
    raw_output: str = ""
    session_id: str = ""
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    error: str = ""
    turns_used: int = 0


@dataclass
class AnalysisResult:
    """Result of a strategy weakness analysis."""

    directions: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""
    cost_usd: float = 0.0


@dataclass
class JudgeResult:
    """Result of candidate evaluation by the agent."""

    decision: str = ""  # "promote" | "merge" | "reject" | "partial"
    reasoning: str = ""
    raw_output: str = ""
    cost_usd: float = 0.0


class CodeAgentBackend(ABC):
    """Unified interface for AI Code Agent backends.

    Each backend wraps a CLI tool (Claude Code, Codex, or Qoder CLI) that
    provides full Agent Loop capabilities: ReAct reasoning, AST-level code
    understanding, automatic error recovery, and context management.
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self._total_cost: float = 0.0
        self._total_calls: int = 0

    @abstractmethod
    async def mutate_strategy(
        self,
        strategy_path: Path,
        hypothesis: dict[str, Any],
        context: dict[str, Any],
    ) -> MutationResult:
        """Modify strategy code according to the given hypothesis.

        Args:
            strategy_path: Path to the strategy .py file to modify.
            hypothesis: Structured hypothesis dict (mutation_type, change, why, etc.).
            context: Additional context (knowledge, dead_ends, champion metrics, etc.).

        Returns:
            MutationResult with success status and metadata.
        """

    @abstractmethod
    async def analyze_weakness(
        self,
        strategy_path: Path,
        diagnostic: dict[str, Any],
    ) -> AnalysisResult:
        """Analyze strategy weaknesses using the agent's code understanding.

        Args:
            strategy_path: Path to the current champion strategy.
            diagnostic: Structured diagnostic output from diagnostics.py.

        Returns:
            AnalysisResult with ranked directions.
        """

    @abstractmethod
    async def judge_candidates(
        self,
        candidates: list[dict[str, Any]],
        champion_metrics: dict[str, Any],
    ) -> JudgeResult:
        """Compare candidate strategies against the champion.

        Args:
            candidates: List of candidate dicts with metrics and descriptions.
            champion_metrics: Current champion's metrics.

        Returns:
            JudgeResult with decision and reasoning.
        """

    @abstractmethod
    async def check_health(self) -> tuple[bool, str]:
        """Check if the backend is available and properly authenticated.

        Returns:
            Tuple of (is_healthy, status_message).
        """

    @property
    def total_cost(self) -> float:
        """Total accumulated cost across all calls."""
        return self._total_cost

    @property
    def total_calls(self) -> int:
        """Total number of backend calls made."""
        return self._total_calls

    def reset_cost(self) -> None:
        """Reset cost tracking to zero."""
        self._total_cost = 0.0
        self._total_calls = 0

    async def _run_subprocess(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> tuple[str, str, int]:
        """Run a subprocess command asynchronously.

        Returns:
            Tuple of (stdout, stderr, return_code).
        """
        timeout = timeout or self.config.timeout_seconds

        logger.info("Running: %s", " ".join(cmd[:5]) + ("..." if len(cmd) > 5 else ""))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            return (
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
                proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            logger.error("Backend call timed out after %ds", timeout)
            proc.kill()  # type: ignore[possibly-undefined]
            return ("", f"Timeout after {timeout}s", 1)
        except FileNotFoundError:
            return ("", f"CLI not found: {cmd[0]}", 127)

    def _parse_json_output(self, stdout: str) -> dict[str, Any]:
        """Parse JSON output from CLI, handling malformed responses."""
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            # Try to find JSON block in output
            for i, char in enumerate(stdout):
                if char == "{":
                    try:
                        return json.loads(stdout[i:])
                    except json.JSONDecodeError:
                        continue
            logger.warning("Could not parse JSON from backend output")
            return {"result": stdout, "parse_error": True}
