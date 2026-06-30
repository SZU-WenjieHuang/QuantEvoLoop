"""IM Bot command handler — interactive commands for Telegram/Discord.

Supported commands:
  /status    — Show current evolution state (gen, sharpe, cost)
  /pause     — Pause the evolution loop (graceful stop after current gen)
  /resume    — Resume a paused evolution loop
  /diagnose  — Run diagnosis on the current champion and return weaknesses
  /history   — Show the last N generation results
  /champion  — Show champion metrics (train/test/holdout)
  /bandit    — Show UCB1 bandit arm statistics
  /help      — Show available commands

The handler uses long-polling (no webhook required) and works with
python-telegram-bot or raw HTTP via httpx/urllib.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger("quantevoloop.channels.bot")


# ---------------------------------------------------------------------------
# Command result
# ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    """Result of processing a bot command."""
    command: str
    text: str
    parse_mode: str = "Markdown"


# ---------------------------------------------------------------------------
# Coordinator protocol (to avoid circular imports)
# ---------------------------------------------------------------------------

class CoordinatorLike(Protocol):
    """Minimal protocol for the coordinator that the bot needs."""

    state: Any  # EvolutionState
    config: Any  # QuantEvoLoopConfig
    champion_metrics: dict
    bandit: Any  # UCBBandit

    def _diagnose_and_hypothesize(self) -> list: ...


# ---------------------------------------------------------------------------
# Command processor
# ---------------------------------------------------------------------------

class CommandProcessor:
    """Process IM bot commands and return formatted responses."""

    def __init__(self, coordinator: CoordinatorLike | None = None, workspace: Path | None = None):
        self.coordinator = coordinator
        self.workspace = workspace or Path("./evo_workspace")
        self._pause_requested = False

    @property
    def pause_requested(self) -> bool:
        return self._pause_requested

    def process(self, command: str, args: str = "") -> CommandResult:
        """Route a command to the appropriate handler."""
        text = command.strip()
        # Split /cmd args into cmd + args if args not provided separately
        if not args and " " in text:
            parts = text.split(maxsplit=1)
            text = parts[0]
            args = parts[1]
        cmd = text.lower().lstrip("/")
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            return CommandResult(
                command=command,
                text=f"Unknown command: `/{cmd}`\nType /help for available commands.",
            )
        try:
            return handler(args)
        except Exception as e:
            logger.error("Command /%s failed: %s", cmd, e, exc_info=True)
            return CommandResult(command=cmd, text=f"Error: {e}")

    # --- Command handlers ---

    def _cmd_help(self, args: str = "") -> CommandResult:
        return CommandResult(command="help", text=(
            "*QuantEvoLoop Bot Commands*\n\n"
            "/status — Current evolution state\n"
            "/champion — Champion metrics\n"
            "/history [N] — Last N generations (default 10)\n"
            "/diagnose — Diagnose champion weaknesses\n"
            "/bandit — UCB1 mutation type stats\n"
            "/pause — Request pause after current gen\n"
            "/resume — Clear pause flag\n"
            "/help — Show this message"
        ))

    def _cmd_status(self, args: str = "") -> CommandResult:
        state = self._load_state()
        if not state:
            return CommandResult(command="status", text="No state.json found. Has evolution started?")
        return CommandResult(command="status", text=(
            f"*Evolution Status*\n\n"
            f"Status: `{state.get('status', 'unknown')}`\n"
            f"Generation: `{state.get('generation', 0)}`\n"
            f"Champion Gen: `{state.get('champion_generation', 0)}`\n"
            f"Champion Sharpe (test): `{state.get('champion_sharpe_test', 0):.3f}`\n"
            f"Promotions: `{state.get('total_promotions', 0)}`\n"
            f"Rejects: `{state.get('total_rejects', 0)}`\n"
            f"Dead Ends: `{state.get('total_dead_ends', 0)}`\n"
            f"Consecutive Rejects: `{state.get('consecutive_rejects', 0)}`\n"
            f"Cost: `${state.get('total_cost_usd', 0):.2f}`\n"
            f"Calls: `{state.get('total_calls', 0)}`\n"
            f"Updated: {state.get('last_updated', 'N/A')}"
        ))

    def _cmd_champion(self, args: str = "") -> CommandResult:
        metrics = self._load_champion_metrics()
        if not metrics:
            return CommandResult(command="champion", text="No champion metrics found.")

        gen = metrics.get("generation", 0)
        train = metrics.get("train", {})
        test = metrics.get("test", {})
        holdout = metrics.get("holdout", {}) or {}

        lines = [
            f"*Champion Metrics* (gen {gen})\n",
            "*Train:*",
            f"  Sharpe: `{train.get('sharpe', 0):.3f}`",
            f"  CAGR: `{train.get('cagr', 0):.1%}`",
            f"  MaxDD: `{train.get('max_drawdown_account', 0):.1%}`",
            f"  Trades: `{train.get('total_trades', 0)}`",
            "",
            "*Test:*",
            f"  Sharpe: `{test.get('sharpe', 0):.3f}`",
            f"  CAGR: `{test.get('cagr', 0):.1%}`",
            f"  MaxDD: `{test.get('max_drawdown_account', 0):.1%}`",
            f"  Trades: `{test.get('total_trades', 0)}`",
        ]
        if holdout:
            lines.extend([
                "",
                "*Holdout:*",
                f"  Sharpe: `{holdout.get('sharpe', 0):.3f}`",
            ])

        return CommandResult(command="champion", text="\n".join(lines))

    def _cmd_history(self, args: str = "") -> CommandResult:
        n = 10
        if args.strip().isdigit():
            n = int(args.strip())
            n = min(n, 50)

        gen_index_path = self.workspace / "generations" / "gen_index.jsonl"
        if not gen_index_path.exists():
            return CommandResult(command="history", text="No generation history found.")

        records = [
            json.loads(line)
            for line in gen_index_path.read_text().splitlines()
            if line.strip()
        ]

        if not records:
            return CommandResult(command="history", text="No generation records.")

        recent = records[-n:]
        lines = [f"*Last {len(recent)} generations:*\n"]
        for r in recent:
            gen = r.get("gen", "?")
            lane = r.get("lane", "?")
            status = r.get("status", "?")
            score = r.get("score")
            tag = r.get("hypothesis_tag", "")
            icon = "+" if status == "promoted" else "-"
            score_str = f"{score:+.4f}" if score is not None else "N/A"
            lines.append(f"  {icon} gen{gen:04d}/L{lane} [{status}] {score_str} `{tag}`")

        return CommandResult(command="history", text="\n".join(lines))

    def _cmd_diagnose(self, args: str = "") -> CommandResult:
        # Try to load trades and run diagnosis
        trades_path = self.workspace / "champion" / "trades.json"
        metrics_path = self.workspace / "champion" / "metrics.json"

        if not trades_path.exists():
            return CommandResult(command="diagnose", text="No champion trades found. Run evolution first.")

        trades = json.loads(trades_path.read_text())
        test_trades = trades.get("test", []) or trades.get("train", [])

        if not test_trades:
            return CommandResult(command="diagnose", text="No trades available for diagnosis.")

        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        test_metrics = metrics.get("test", {})

        from ..evaluation.diagnostics import diagnose

        strategy_summary = {
            "sharpe": test_metrics.get("sharpe", 0.0),
            "cagr": test_metrics.get("cagr", 0.0),
            "max_drawdown_account": test_metrics.get("max_drawdown_account", 0.0),
            "total_trades": test_metrics.get("total_trades", 0),
            "winrate": test_metrics.get("winrate", 0.0),
        }
        diagnostic = diagnose(test_trades, strategy_summary, segment="test")

        lines = [f"*Champion Diagnosis* ({len(test_trades)} trades)\n"]
        lines.append(f"Segment: `{diagnostic.segment}`")
        lines.append(f"Trades analyzed: `{diagnostic.n_trades}`")

        # Show dominant exit reason
        if diagnostic.by_exit_reason:
            top_exit = max(diagnostic.by_exit_reason.items(), key=lambda x: x[1].get("count", 0))
            lines.append(f"Top exit: `{top_exit[0]}` ({top_exit[1].get('count', 0)} trades)")
        lines.append("")

        if diagnostic.weaknesses:
            lines.append(f"*Weaknesses ({len(diagnostic.weaknesses)}):*")
            for w in diagnostic.weaknesses[:8]:
                lines.append(f"  - [{w.tag}] {w.description}")
        else:
            lines.append("No significant weaknesses detected.")

        return CommandResult(command="diagnose", text="\n".join(lines))

    def _cmd_bandit(self, args: str = "") -> CommandResult:
        if self.coordinator and hasattr(self.coordinator, "bandit"):
            stats = self.coordinator.bandit.get_arm_stats()
        else:
            # Try to load from checkpoint
            stats = self._load_bandit_from_checkpoint()

        if not stats:
            return CommandResult(command="bandit", text="No bandit statistics available yet.")

        lines = ["*UCB1 Bandit Arm Stats:*\n"]
        sorted_arms = sorted(stats.items(), key=lambda x: x[1].get("ucb", 0), reverse=True)
        for arm_name, arm_data in sorted_arms:
            pulls = arm_data.get("pulls", 0)
            avg = arm_data.get("avg_reward", 0)
            ucb = arm_data.get("ucb", 0)
            lines.append(f"  `{arm_name}`: pulls={pulls}, avg={avg:+.3f}, ucb={ucb:.3f}")

        return CommandResult(command="bandit", text="\n".join(lines))

    def _cmd_pause(self, args: str = "") -> CommandResult:
        self._pause_requested = True
        return CommandResult(command="pause", text="Pause requested. Evolution will stop after the current generation completes.")

    def _cmd_resume(self, args: str = "") -> CommandResult:
        self._pause_requested = False
        return CommandResult(command="resume", text="Resume flag cleared. Evolution will continue.")

    # --- Helpers ---

    def _load_state(self) -> dict | None:
        path = self.workspace / "state.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _load_champion_metrics(self) -> dict | None:
        path = self.workspace / "champion" / "metrics.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _load_bandit_from_checkpoint(self) -> dict | None:
        import glob as glob_mod
        ckpts = sorted(glob_mod.glob(str(self.workspace / "checkpoint_gen*.json")))
        if ckpts:
            data = json.loads(Path(ckpts[-1]).read_text())
            return data.get("bandit", {})
        return None


# ---------------------------------------------------------------------------
# Telegram polling bot (lightweight, no external dependency beyond httpx)
# ---------------------------------------------------------------------------

class TelegramBotPoller:
    """Lightweight Telegram bot using long-polling (getUpdates).

    Listens for commands and dispatches to CommandProcessor.
    No external dependency beyond httpx (already used by telegram_adapter).
    """

    def __init__(self, bot_token: str, processor: CommandProcessor):
        self.bot_token = bot_token
        self.processor = processor
        self._running = False
        self._last_update_id = 0

    async def start(self) -> None:
        """Start long-polling loop."""
        self._running = True
        logger.info("Telegram bot poller started")

        try:
            import httpx
            client = httpx.AsyncClient(timeout=35)
        except ImportError:
            logger.error("httpx not installed, cannot start Telegram bot")
            return

        try:
            while self._running:
                updates = await self._get_updates(client)
                for update in updates:
                    await self._handle_update(client, update)
                await asyncio.sleep(1)
        finally:
            await client.aclose()

    def stop(self) -> None:
        self._running = False

    async def _get_updates(self, client: Any) -> list[dict]:
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {"timeout": 30, "offset": self._last_update_id + 1}
        try:
            resp = await client.get(url, params=params)
            data = resp.json()
            if data.get("ok"):
                return data.get("result", [])
        except Exception as e:
            logger.debug("getUpdates error: %s", e)
            await asyncio.sleep(5)
        return []

    async def _handle_update(self, client: Any, update: dict) -> None:
        self._last_update_id = update.get("update_id", self._last_update_id)
        message = update.get("message", {})
        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id")

        if not chat_id or not text.startswith("/"):
            return

        # Parse command
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].split("@")[0]  # Handle /cmd@botname format
        args = parts[1] if len(parts) > 1 else ""

        logger.info("Received command: %s from chat %s", cmd, chat_id)
        result = self.processor.process(cmd, args)

        await self._send_reply(client, chat_id, result.text)

    async def _send_reply(self, client: Any, chat_id: int, text: str) -> None:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            resp = await client.post(url, json=payload)
            if not resp.json().get("ok"):
                # Retry without Markdown
                payload["parse_mode"] = ""
                await client.post(url, json=payload)
        except Exception as e:
            logger.warning("Reply failed: %s", e)
