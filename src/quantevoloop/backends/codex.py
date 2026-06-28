"""OpenAI Codex CLI backend adapter.

Invokes Codex in non-interactive exec mode:
    codex exec "<task description>" --approval-mode full-auto

Reference: https://github.com/openai/codex
"""

from __future__ import annotations

import ast
import json
import logging
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from quantevoloop.backends.base import (
    AnalysisResult,
    CodeAgentBackend,
    JudgeResult,
    MutationResult,
)
from quantevoloop.config import BackendConfig

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class CodexBackend(CodeAgentBackend):
    """Backend using OpenAI Codex CLI in exec (non-interactive) mode."""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(default=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    async def mutate_strategy(
        self,
        strategy_path: Path,
        hypothesis: dict[str, Any],
        context: dict[str, Any],
    ) -> MutationResult:
        template = self._jinja_env.get_template("mutation.j2")
        task = template.render(
            strategy_path=str(strategy_path),
            hypothesis=json.dumps(hypothesis, indent=2, ensure_ascii=False),
            context=json.dumps(context, indent=2, ensure_ascii=False),
        )

        t0 = time.monotonic()
        stdout, stderr, rc = await self._run_codex(task, strategy_path.parent)
        latency = time.monotonic() - t0

        self._total_calls += 1

        if rc != 0:
            return MutationResult(
                success=False,
                raw_output=stdout,
                error=stderr or f"Exit code {rc}",
                latency_seconds=latency,
            )

        syntax_ok = self._validate_syntax(strategy_path)
        if not syntax_ok:
            return MutationResult(
                success=False,
                raw_output=stdout,
                error="Strategy file has syntax errors after mutation",
                latency_seconds=latency,
            )

        return MutationResult(
            success=True,
            modified_files=[str(strategy_path)],
            raw_output=stdout,
            latency_seconds=latency,
        )

    async def analyze_weakness(
        self,
        strategy_path: Path,
        diagnostic: dict[str, Any],
    ) -> AnalysisResult:
        template = self._jinja_env.get_template("diagnose.j2")
        task = template.render(
            strategy_path=str(strategy_path),
            diagnostic=json.dumps(diagnostic, indent=2, ensure_ascii=False),
        )

        stdout, stderr, rc = await self._run_codex(task, strategy_path.parent)
        self._total_calls += 1

        parsed = self._parse_json_output(stdout)
        directions = parsed.get("directions", []) if isinstance(parsed, dict) else []

        return AnalysisResult(
            directions=directions,
            raw_output=stdout,
        )

    async def judge_candidates(
        self,
        candidates: list[dict[str, Any]],
        champion_metrics: dict[str, Any],
    ) -> JudgeResult:
        template = self._jinja_env.get_template("judge.j2")
        task = template.render(
            candidates=json.dumps(candidates, indent=2, ensure_ascii=False),
            champion=json.dumps(champion_metrics, indent=2, ensure_ascii=False),
        )

        stdout, stderr, rc = await self._run_codex(task, cwd=None)
        self._total_calls += 1

        parsed = self._parse_json_output(stdout)
        decision = "reject"
        reasoning = stdout

        if isinstance(parsed, dict):
            decision = parsed.get("decision", "reject")
            reasoning = parsed.get("reasoning", stdout)
        elif "promote" in stdout.lower():
            decision = "promote"

        return JudgeResult(decision=decision, reasoning=reasoning, raw_output=stdout)

    async def check_health(self) -> tuple[bool, str]:
        stdout, stderr, rc = await self._run_subprocess(
            [self.config.cli_path, "--version"],
        )
        if rc == 0:
            return True, f"Codex: {stdout.strip()}"
        return False, f"Codex not available: {stderr.strip() or 'not found'}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_codex(
        self,
        task: str,
        cwd: Path | None,
    ) -> tuple[str, str, int]:
        """Execute `codex exec <task>` with full-auto approval."""
        cmd = [
            self.config.cli_path,
            "exec",
            task,
            "--approval-mode", "full-auto",
        ]
        cmd.extend(self.config.extra_args)

        return await self._run_subprocess(cmd, cwd=cwd)

    @staticmethod
    def _validate_syntax(path: Path) -> bool:
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            return True
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning("Syntax validation failed for %s: %s", path, e)
            return False
