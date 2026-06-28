"""Claude Code CLI backend adapter.

Invokes Claude Code in headless mode:
    claude --bare -p "<prompt>" --output-format json --allowedTools ... --max-turns N -w <dir>

Reference: https://code.claude.com/docs/en/headless
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
    BackendMutationContext,
    CodeAgentBackend,
    JudgeResult,
    MutationResult,
)
from quantevoloop.config import BackendConfig

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


class ClaudeCodeBackend(CodeAgentBackend):
    """Backend using Claude Code CLI in headless (--bare -p) mode."""

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
        hypothesis: str,
        context: BackendMutationContext | None = None,
    ) -> MutationResult:
        ctx = context or BackendMutationContext()
        template = self._jinja_env.get_template("mutation.j2")
        prompt = template.render(
            strategy_path=str(strategy_path),
            hypothesis=hypothesis,
            context=json.dumps(ctx.to_dict(), indent=2, ensure_ascii=False),
        )

        workspace = strategy_path.parent
        t0 = time.monotonic()
        stdout, stderr, rc = await self._run_claude(prompt, workspace)
        latency = time.monotonic() - t0

        parsed = self._parse_json_output(stdout)
        cost = parsed.get("total_cost_usd", 0.0)
        self._total_cost += cost
        self._total_calls += 1

        if rc != 0:
            return MutationResult(
                success=False,
                raw_output=stdout,
                error=stderr or f"Exit code {rc}",
                cost_usd=cost,
                latency_seconds=latency,
            )

        # Validate that the strategy file is still syntactically valid
        syntax_ok = self._validate_syntax(strategy_path)
        if not syntax_ok:
            return MutationResult(
                success=False,
                raw_output=parsed.get("result", ""),
                error="Strategy file has syntax errors after mutation",
                cost_usd=cost,
                latency_seconds=latency,
            )

        return MutationResult(
            success=True,
            modified_files=[str(strategy_path)],
            raw_output=parsed.get("result", ""),
            session_id=parsed.get("session_id", ""),
            cost_usd=cost,
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

        stdout, stderr, rc = await self._run_claude(
            prompt, strategy_path.parent, output_json=True,
        )
        parsed = self._parse_json_output(stdout)
        cost = parsed.get("total_cost_usd", 0.0)
        self._total_cost += cost
        self._total_calls += 1

        # Try to extract structured directions from the response
        directions = []
        if "structured_output" in parsed:
            directions = parsed["structured_output"].get("directions", [])
        elif isinstance(parsed.get("result"), list):
            directions = parsed["result"]

        return AnalysisResult(
            directions=directions,
            raw_output=parsed.get("result", str(parsed)),
            cost_usd=cost,
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

        stdout, stderr, rc = await self._run_claude(prompt, cwd=None, output_json=True)
        parsed = self._parse_json_output(stdout)
        cost = parsed.get("total_cost_usd", 0.0)
        self._total_cost += cost
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
            cost_usd=cost,
        )

    async def check_health(self) -> tuple[bool, str]:
        stdout, stderr, rc = await self._run_subprocess(
            [self.config.cli_path, "--version"],
        )
        if rc == 0:
            return True, f"Claude Code: {stdout.strip()}"
        return False, f"Claude Code not available: {stderr.strip() or 'not found'}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_claude(
        self,
        prompt: str,
        cwd: Path | None,
        output_json: bool = True,
    ) -> tuple[str, str, int]:
        """Execute `claude --bare -p <prompt>` with configured options."""
        cmd = [
            self.config.cli_path,
            "--bare",
            "-p", prompt,
            "--max-turns", str(self.config.max_turns),
        ]

        if output_json:
            cmd.extend(["--output-format", "json"])

        if self.config.allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.config.allowed_tools)])

        if cwd:
            cmd.extend(["-w", str(cwd)])

        cmd.extend(self.config.extra_args)

        return await self._run_subprocess(cmd, cwd=None)  # cwd passed via -w flag

    @staticmethod
    def _validate_syntax(path: Path) -> bool:
        """Check if a Python file has valid syntax."""
        try:
            source = path.read_text(encoding="utf-8")
            ast.parse(source, filename=str(path))
            return True
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning("Syntax validation failed for %s: %s", path, e)
            return False
