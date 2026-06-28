"""QuantEvoLoop CLI — main entry point.

Usage:
    quantevoloop init --strategy path/to/strategy.py [--config config.yaml]
    quantevoloop run [--config config.yaml] [--lanes N] [--backend TYPE]
    quantevoloop diagnose [--config config.yaml]
    quantevoloop status [--config config.yaml]
    quantevoloop backend-check [--config config.yaml]
    quantevoloop batch --n 10 [--config config.yaml]
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


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True),
              default="evo_workspace/config.yaml", help="Path to config YAML")
@click.option("--lanes", type=int, default=None, help="Override number of lanes")
@click.option("--backend", type=click.Choice(["claude-code", "codex", "qoder-cli"]),
              default=None, help="Override backend type")
@click.option("--max-gens", type=int, default=None, help="Max generations (override config)")
@click.option("--notify", type=click.Choice(["telegram", "discord", "webhook", "none"]),
              default="none", help="Enable IM notifications")
def run(config_path: str, lanes: int | None, backend: str | None,
        max_gens: int | None, notify: str):
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
        f"Lanes: {config.n_lanes}\n"
        f"Strategy: {config.strategy_path}\n"
        f"Workspace: {config.workspace_dir}",
        title="Starting Evolution Loop",
        border_style="blue",
    ))

    # Wire up the main evolution loop
    from quantevoloop.backends import create_backend
    from quantevoloop.channels import IMNotifier
    from quantevoloop.gateway.coordinator import EvolutionCoordinator
    from quantevoloop.engine.mock_engine import MockBacktestEngine

    backend = create_backend(config.backend)
    engine = MockBacktestEngine(config)
    notifier = IMNotifier(config.im) if (config.im.telegram_enabled or config.im.discord_enabled or config.im.webhook_enabled) else None

    coordinator = EvolutionCoordinator(config, backend, engine, notifier)

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

    config = QuantEvoLoopConfig.from_yaml(config_path)
    console.print(f"[blue]Diagnosing:[/blue] {config.champion_dir / 'strategy.py'}")
    console.print("[yellow]Diagnosis not yet implemented. Coming in Phase 4.[/yellow]")


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
        table.add_row("Champion Sharpe", str(state.get("champion_sharpe", "N/A")))
        table.add_row("Active Campaign", state.get("active_campaign", "None"))
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
    console.print(f"[yellow]Batch mode ({n} iterations) not yet implemented. Coming in Phase 5.[/yellow]")


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True),
              default="evo_workspace/config.yaml")
def gateway(config_path: str):
    """Start the Gateway Server with IM integration."""
    console.print("[yellow]Gateway mode not yet implemented. Coming in Phase 5.[/yellow]")


if __name__ == "__main__":
    main()
