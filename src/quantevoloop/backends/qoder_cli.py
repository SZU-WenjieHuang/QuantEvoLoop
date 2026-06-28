"""Qoder CLI backend adapter.

Invokes Qoder CLI in non-interactive print mode:
    qodercli -p "<prompt>" --output-format json --yolo --max-turns N -w <dir>

Reference: https://docs.qoder.com/en/cli/using-cli
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


class QoderCliBackend(CodeAgentBackend):
    """Backend using Qoder CLI in print (-p) mode."""

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
        prompt = template.render(
            strategy_path=str(strategy_path),
            hypothesis=json.dumps(hypothesis, indent=2, ensure_ascii=False),
            context=json.dumps(context, indent=2, ensure_ascii=False),
        )

        workspace = strategy_path.parent
        t0 = time.monotonic()
        stdout, stderr, rc = await self._run_qoder(prompt, workspace)
        latency = time.monotonic() - t0

        parsed = self._parse_json_output(stdout)
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
                raw_output=parsed.get("result", ""),
                error="Strategy file has syntax errors after mutation",
                latency_seconds=latency,
            )

        return MutationResult(
            success=True,
            modified_files=[str(strategy_path)],
            raw_output=parsed.get("result", ""),
            session_id=parsed.get("session_id", ""),
            latency_seconds=latency,
        )

    async def analyze_weakness(
        self,
        strategy_path: Path,
        diagnostic: dict[str, Any],
    ) -> AnalysisResult:
        template = self._jinja_env.get_template("diagnose.j2")
        prompt = template.render(
            strategy_path=str(strategy_path),
            diagnostic=json.dumps(diagnostic, indent=2, ensure_ascii=False),
        )

        stdout, stderr, rc = await self._run_qoder(prompt, strategy_path.parent)
        parsed = self._parse_json_output(stdout)
        self._total_calls += 1

        directions = []
        if "structured_output" in parsed:
            directions = parsed["structured_output"].get("directions", [])
        elif isinstance(parsed.get("result"), list):
            directions = parsed["result"]

        return AnalysisResult(
            directions=directions,
            raw_output=parsed.get("result", str(parsed)),
        )

    async def judge_candidates(
        self,
        candidates: list[dict[str, Any]],
        champion_metrics: dict[str, Any],
    ) -> JudgeResult:
        template = self._jinja_env.get_template("judge.j2")
        prompt = template.render(
            candidates=json.dumps(candidates, indent=2, ensure_ascii=False),
            champion=json.dumps(champion_metrics, indent=2, ensure_ascii=False),
        )

        stdout, stderr, rc = await self._run_qoder(prompt, cwd=None)
        parsed = self._parse_json_output(stdout)
        self._total_calls += 1

        decision = "reject"
        reasoning = ""

        if "structured_output" in parsed:
            so = parsed["structured_output"]
            decision = so.get("decision", "reject")
            reasoning = so.get("reasoning", "")
        else:
            result_text = parsed.get("result", "")
            if "promote" in result_text.lower():
                decision = "promote"
            reasoning = result_text

        return JudgeResult(
            decision=decision,
            reasoning=reasoning,
            raw_output=parsed.get("result", ""),
        )

    async def check_health(self) -> tuple[bool, str]:
        stdout, stderr, rc = await self._run_subprocess(
            [self.config.cli_path, "--version"],
        )
        if rc == 0:
            return True, f"Qoder CLI: {stdout.strip()}"
        return False, f"Qoder CLI not available: {stderr.strip() or 'not found'}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_qoder(
        self,
        prompt: str,
        cwd: Path | None,
    ) -> tuple[str, str, int]:
        """Execute `qodercli -p <prompt>` with --yolo and JSON output."""
        cmd = [
            self.config.cli_path,
            "-p", prompt,
            "--output-format", "json",
            "--yolo",
            "--max-turns", str(self.config.max_turns),
        ]

        if cwd:
            cmd.extend(["-w", str(cwd)])

        cmd.extend(self.config.extra_args)

        return await self._run_subprocess(cmd, cwd=None)

    @staticmethod
    def _validate_syntax(path: Path) -> bool:
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            return True
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning("Syntax validation failed for %s: %s", path, e)
            return False
