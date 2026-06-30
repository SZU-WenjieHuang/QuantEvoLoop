"""QuantEvoLoop CLI — main entry point.

Usage:
    quantevoloop init --strategy path/to/strategy.py [--config config.yaml]
    quantevoloop run [--config config.yaml] [--lanes N] [--backend TYPE] [--engine TYPE]
    quantevoloop diagnose [--config config.yaml]
    quantevoloop status [--config config.yaml]
    quantevoloop backend-check [--config config.yaml]
    quantevoloop batch --n 10 [--config config.yaml]
    quantevoloop gateway [--config config.yaml]  # Start with IM bot
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from quantevoloop import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="QuantEvoLoop")
def main():
    """QuantEvoLoop: Multi-Agent Evolutionary Loop for Quantitative Strategy Optimization."""


@main.command()
@click.option("--strategy", required=True, type=click.Path(exists=True),
              help="Path to the Freqtrade strategy .py file")
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Path to config YAML (optional, generates default if omitted)")
@click.option("--backend", type=click.Choice(["claude-code", "codex", "qoder-cli"]),
              default="claude-code", help="AI Code Agent backend")
@click.option("--workspace", type=click.Path(), default="./evo_workspace",
              help="Workspace directory for evolution artifacts")
def init(strategy: str, config_path: str | None, backend: str, workspace: str):
    """Initialize a new QuantEvoLoop workspace."""
    from quantevoloop.config import QuantEvoLoopConfig, BackendConfig

    if config_path and Path(config_path).exists():
        config = QuantEvoLoopConfig.from_yaml(config_path)
    else:
        config = QuantEvoLoopConfig(
            workspace_dir=Path(workspace),
            strategy_path=Path(strategy),
            backend=BackendConfig(type=backend),  # type: ignore[arg-type]
        )

    config.ensure_dirs()

    # Save config to workspace
    saved_config = config.workspace_dir / "config.yaml"
    config.to_yaml(saved_config)

    # Copy strategy to champion
    champion_strategy = config.champion_dir / "strategy.py"
    champion_strategy.parent.mkdir(parents=True, exist_ok=True)
    strategy_src = Path(strategy).read_text(encoding="utf-8")
    champion_strategy.write_text(strategy_src, encoding="utf-8")

    console.print(Panel(
        f"[green]Workspace initialized:[/green] {config.workspace_dir}\n"
        f"[green]Strategy:[/green] {strategy}\n"
        f"[green]Backend:[/green] {backend}\n"
        f"[green]Config saved:[/green] {saved_config}\n"
        f"[green]Champion seeded:[/green] {champion_strategy}",
        title="QuantEvoLoop Init",
        border_style="green",
    ))


def _create_engine(config, engine_type: str):
    """Create the backtest engine based on --engine flag."""
    if engine_type == "mock":
        from quantevoloop.engine.mock_engine import MockBacktestEngine
        return MockBacktestEngine(config)
    elif engine_type == "freqtrade":
        from quantevoloop.engine.freqtrade_engine import FreqtradeEngine
        return FreqtradeEngine(
            python_bin=str(config.python_bin),
            freqtrade_config=str(config.backtest_config) if config.backtest_config else None,
            data_splits=config.data_splits,
        )
    elif engine_type == "backtrader":
        from quantevoloop.engine.backtrader_engine import BacktraderEngine
        return BacktraderEngine(config.data_splits)
    elif engine_type == "zipline":
        from quantevoloop.engine.zipline_engine import ZiplineEngine
        return ZiplineEngine(config.data_splits)
    else:
        from quantevoloop.engine.mock_engine import MockBacktestEngine
        return MockBacktestEngine(config)


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True),
              default="evo_workspace/config.yaml", help="Path to config YAML")
@click.option("--lanes", type=int, default=None, help="Override number of lanes")
@click.option("--backend", type=click.Choice(["claude-code", "codex", "qoder-cli"]),
              default=None, help="Override backend type")
@click.option("--max-gens", type=int, default=None, help="Max generations (override config)")
@click.option("--engine", type=click.Choice(["mock", "freqtrade", "backtrader", "zipline"]),
              default="mock", help="Backtest engine to use")
@click.option("--notify", type=click.Choice(["telegram", "discord", "webhook", "none"]),
              default="none", help="Enable IM notifications")
def run(config_path: str, lanes: int | None, backend: str | None,
        max_gens: int | None, engine: str, notify: str):
    """Start the evolutionary loop."""
    from quantevoloop.config import QuantEvoLoopConfig

    config = QuantEvoLoopConfig.from_yaml(config_path)

    # Apply overrides
    if lanes is not None:
        config.n_lanes = lanes
    if backend is not None:
        config.backend.type = backend  # type: ignore[misc]
    if max_gens is not None:
        config.max_total_generations = max_gens

    console.print(Panel(
        f"[bold]QuantEvoLoop v{__version__}[/bold]\n\n"
        f"Backend: {config.backend.type}\n"
        f"Engine: {engine}\n"
        f"Lanes: {config.n_lanes}\n"
        f"Strategy: {config.strategy_path}\n"
        f"Workspace: {config.workspace_dir}",
        title="Starting Evolution Loop",
        border_style="blue",
    ))

    from quantevoloop.backends import create_backend
    from quantevoloop.channels import IMNotifier
    from quantevoloop.gateway.coordinator import EvolutionCoordinator

    bk = create_backend(config.backend)
    eng = _create_engine(config, engine)
    notifier = IMNotifier(config.im) if (config.im.telegram_enabled or config.im.discord_enabled or config.im.webhook_enabled) else None

    coordinator = EvolutionCoordinator(config, bk, eng, notifier)

    async def _run_loop():
        state = await coordinator.run(max_generations=max_gens or config.max_campaign_iter)
        return state

    state = asyncio.run(_run_loop())
    console.print(Panel(
        f"Generations: {state.generation}\n"
        f"Promotions: {state.total_promotions}\n"
        f"Dead ends: {state.total_dead_ends}\n"
        f"Total cost: ${state.total_cost_usd:.2f}\n"
        f"Status: {state.status}",
        title="Evolution Complete",
        border_style="green",
    ))


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True),
              default="evo_workspace/config.yaml")
def diagnose(config_path: str):
    """Run weakness diagnosis on the current champion."""
    from quantevoloop.config import QuantEvoLoopConfig
    from quantevoloop.evaluation.diagnostics import diagnose as run_diagnose

    config = QuantEvoLoopConfig.from_yaml(config_path)
    trades_path = config.champion_dir / "trades.json"
    metrics_path = config.champion_dir / "metrics.json"

    if not trades_path.exists():
        console.print("[yellow]No champion trades found. Run evolution first.[/yellow]")
        return

    import json
    trades = json.loads(trades_path.read_text())
    test_trades = trades.get("test", []) or trades.get("train", [])

    if not test_trades:
        console.print("[yellow]No trades available for diagnosis.[/yellow]")
        return

    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    test_metrics = metrics.get("test", {})

    strategy_summary = {
        "sharpe": test_metrics.get("sharpe", 0.0),
        "cagr": test_metrics.get("cagr", 0.0),
        "max_drawdown_account": test_metrics.get("max_drawdown_account", 0.0),
        "total_trades": test_metrics.get("total_trades", 0),
        "winrate": test_metrics.get("winrate", 0.0),
    }

    diagnostic = run_diagnose(test_trades, strategy_summary, segment="test")

    table = Table(title=f"Champion Diagnosis ({len(test_trades)} trades)")
    table.add_column("Category", style="cyan")
    table.add_column("Details", style="white")

    table.add_row("Severity", diagnostic.severity)
    table.add_row("Dominant Exit", diagnostic.dominant_exit)
    table.add_row("Weaknesses", str(len(diagnostic.weaknesses)))
    table.add_row("Suggestions", str(len(diagnostic.suggestions)))

    for w in diagnostic.weaknesses[:10]:
        table.add_row(f"  [{w.tag}]", w.description)

    console.print(table)


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True),
              default="evo_workspace/config.yaml")
def status(config_path: str):
    """Show current workspace status."""
    from quantevoloop.config import QuantEvoLoopConfig

    config = QuantEvoLoopConfig.from_yaml(config_path)

    table = Table(title="QuantEvoLoop Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Workspace", str(config.workspace_dir))
    table.add_row("Strategy", str(config.strategy_path))
    table.add_row("Backend", config.backend.type)
    table.add_row("Lanes", str(config.n_lanes))
    table.add_row("Data: Train", config.data_splits.train_timerange)
    table.add_row("Data: Test", config.data_splits.test_timerange)
    table.add_row("Data: Holdout", config.data_splits.holdout_timerange)

    # Check workspace state
    if config.state_file.exists():
        import json
        state = json.loads(config.state_file.read_text())
        table.add_row("Current Gen", str(state.get("generation", 0)))
        table.add_row("Champion Sharpe", str(state.get("champion_sharpe_test", "N/A")))
        table.add_row("Status", state.get("status", "unknown"))
        table.add_row("Cost", f"${state.get('total_cost_usd', 0):.2f}")
        table.add_row("Promotions", str(state.get("total_promotions", 0)))
    else:
        table.add_row("Status", "[yellow]Not initialized (run 'quantevoloop init')[/yellow]")

    console.print(table)


@main.command("backend-check")
@click.option("--config", "config_path", type=click.Path(exists=True),
              default="evo_workspace/config.yaml")
def backend_check(config_path: str):
    """Check if the configured backend is available."""
    from quantevoloop.backends import create_backend
    from quantevoloop.config import QuantEvoLoopConfig

    config = QuantEvoLoopConfig.from_yaml(config_path)
    backend = create_backend(config.backend)

    async def _check():
        healthy, msg = await backend.check_health()
        if healthy:
            console.print(f"[green]OK:[/green] {msg}")
        else:
            console.print(f"[red]FAIL:[/red] {msg}")
            sys.exit(1)

    asyncio.run(_check())


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True),
              default="evo_workspace/config.yaml")
@click.option("--n", type=int, default=10, help="Number of iterations")
def batch(config_path: str, n: int):
    """Run N evolution iterations in batch mode."""
    from quantevoloop.config import QuantEvoLoopConfig
    from quantevoloop.backends import create_backend
    from quantevoloop.gateway.coordinator import EvolutionCoordinator

    config = QuantEvoLoopConfig.from_yaml(config_path)
    bk = create_backend(config.backend)
    eng = _create_engine(config, "mock")

    coordinator = EvolutionCoordinator(config, bk, eng)

    async def _run_batch():
        state = await coordinator.run(max_generations=n)
        return state

    state = asyncio.run(_run_batch())
    console.print(Panel(
        f"Generations: {state.generation}\n"
        f"Promotions: {state.total_promotions}\n"
        f"Status: {state.status}",
        title=f"Batch ({n} iterations) Complete",
        border_style="green",
    ))


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True),
              default="evo_workspace/config.yaml")
@click.option("--engine", type=click.Choice(["mock", "freqtrade", "backtrader", "zipline"]),
              default="mock", help="Backtest engine to use")
def gateway(config_path: str, engine: str):
    """Start the Gateway Server with IM bot integration.

    Launches the evolution loop + Telegram bot command handler.
    The bot listens for /pause, /diagnose, /status, /history, etc.
    """
    from quantevoloop.config import QuantEvoLoopConfig
    from quantevoloop.backends import create_backend
    from quantevoloop.channels import IMNotifier, CommandProcessor, TelegramBotPoller
    from quantevoloop.gateway.coordinator import EvolutionCoordinator

    config = QuantEvoLoopConfig.from_yaml(config_path)
    bk = create_backend(config.backend)
    eng = _create_engine(config, engine)
    notifier = IMNotifier(config.im) if (config.im.telegram_enabled or config.im.discord_enabled or config.im.webhook_enabled) else None

    coordinator = EvolutionCoordinator(config, bk, eng, notifier)
    cmd_processor = CommandProcessor(coordinator=coordinator, workspace=config.workspace_dir)

    console.print(Panel(
        f"[bold]QuantEvoLoop Gateway v{__version__}[/bold]\n\n"
        f"Backend: {config.backend.type}\n"
        f"Engine: {engine}\n"
        f"Telegram Bot: {'enabled' if config.im.telegram_enabled else 'disabled'}\n"
        f"Workspace: {config.workspace_dir}\n\n"
        f"Bot commands: /status /pause /diagnose /history /champion /bandit /help",
        title="Gateway Starting",
        border_style="blue",
    ))

    async def _run_gateway():
        tasks = []

        # Start evolution loop
        tasks.append(coordinator.run(max_generations=config.max_campaign_iter))

        # Start Telegram bot poller if enabled
        if config.im.telegram_enabled and config.im.telegram_bot_token:
            poller = TelegramBotPoller(config.im.telegram_bot_token, cmd_processor)
            tasks.append(poller.start())

        # Run both concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                console.print(f"[red]Task error:[/red] {r}")

    asyncio.run(_run_gateway())


if __name__ == "__main__":
    main()
