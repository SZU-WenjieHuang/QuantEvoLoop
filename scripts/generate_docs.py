#!/usr/bin/env python3
"""Generate documentation visual assets for QuantEvoLoop README.

Generates:
  1. docs/architecture.png   - System architecture overview
  2. docs/pipeline.png       - 5-layer statistical gate pipeline
  3. docs/evolution_gif.gif  - Evolution loop animation
  4. docs/tournament_gif.gif - Multi-lane tournament animation

Usage:
    pip install matplotlib numpy Pillow
    python scripts/generate_docs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter

# Output directory
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# Color palette
COLORS = {
    "bg": "#1a1a2e",
    "panel": "#16213e",
    "accent": "#0f3460",
    "highlight": "#e94560",
    "text": "#eee",
    "subtext": "#aaa",
    "green": "#4ade80",
    "red": "#f87171",
    "blue": "#60a5fa",
    "purple": "#a78bfa",
    "yellow": "#fbbf24",
    "orange": "#fb923c",
}


def _rounded_box(ax, xy, w, h, label, color, fontsize=10, text_color="white"):
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.9,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold")


def _arrow(ax, xy1, xy2, color="white", style="->"):
    """Draw an arrow between two points."""
    ax.annotate("", xy=xy2, xytext=xy1,
                arrowprops=dict(arrowstyle=style, color=color, lw=1.5))


# ===========================================================================
# 1. Architecture Overview
# ===========================================================================

def generate_architecture():
    """Generate system architecture overview diagram."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS["bg"])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg"])

    # Title
    ax.text(7, 7.6, "QuantEvoLoop Architecture", ha="center", va="center",
            fontsize=18, color=COLORS["text"], fontweight="bold")

    # Layer 1: CLI / Gateway
    ax.add_patch(FancyBboxPatch((0.5, 6.5), 13, 0.8,
                 boxstyle="round,pad=0.05", facecolor=COLORS["accent"],
                 edgecolor=COLORS["blue"], linewidth=1.5, alpha=0.8))
    ax.text(7, 6.9, "CLI / Gateway    init | run | diagnose | status | gateway",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    # Layer 2: Core modules (5 sub-panels)
    y2 = 4.5
    h2 = 1.5
    modules = [
        ("Agents", ["LeadAgent", "SubAgent", "JudgeAgent"], COLORS["purple"]),
        ("Selection", ["UCB1 Bandit", "Tournament", "Population"], COLORS["blue"]),
        ("Evolution", ["Campaign", "Knowledge", "DeadEnds"], COLORS["green"]),
        ("Evaluation", ["PSR", "Bootstrap CI", "Walk-Forward", "Holdout"], COLORS["yellow"]),
        ("Channels", ["Telegram", "Discord", "Webhook"], COLORS["orange"]),
    ]
    w_panel = 2.35
    gap = 0.15
    x_start = 0.5
    for i, (title, items, color) in enumerate(modules):
        x = x_start + i * (w_panel + gap)
        ax.add_patch(FancyBboxPatch((x, y2), w_panel, h2,
                     boxstyle="round,pad=0.05", facecolor=COLORS["panel"],
                     edgecolor=color, linewidth=1.3, alpha=0.85))
        ax.text(x + w_panel / 2, y2 + h2 - 0.2, title, ha="center", va="top",
                fontsize=10, color=color, fontweight="bold")
        for j, item in enumerate(items):
            ax.text(x + w_panel / 2, y2 + h2 - 0.5 - j * 0.3, item,
                    ha="center", va="top", fontsize=8, color=COLORS["subtext"])

    # Layer 3: Backends
    y3 = 2.8
    ax.add_patch(FancyBboxPatch((0.5, y3), 13, 1.0,
                 boxstyle="round,pad=0.05", facecolor=COLORS["panel"],
                 edgecolor=COLORS["highlight"], linewidth=1.5, alpha=0.8))
    ax.text(7, y3 + 0.8, "Backends (Agent-as-CLI)", ha="center", va="center",
            fontsize=10, color=COLORS["highlight"], fontweight="bold")
    backends = ["Claude Code CLI", "Codex CLI", "Qoder CLI"]
    for i, b in enumerate(backends):
        bx = 2.5 + i * 4
        _rounded_box(ax, (bx, y3 + 0.1), 2.5, 0.5, b, COLORS["accent"], fontsize=9)

    # Layer 4: Engines
    y4 = 1.0
    ax.add_patch(FancyBboxPatch((0.5, y4), 13, 1.0,
                 boxstyle="round,pad=0.05", facecolor=COLORS["panel"],
                 edgecolor=COLORS["green"], linewidth=1.5, alpha=0.8))
    ax.text(7, y4 + 0.8, "Backtest Engines", ha="center", va="center",
            fontsize=10, color=COLORS["green"], fontweight="bold")
    engines = ["Freqtrade", "Mock", "Backtrader", "Zipline"]
    for i, e in enumerate(engines):
        ex = 1.5 + i * 3.2
        _rounded_box(ax, (ex, y4 + 0.1), 2.2, 0.5, e, COLORS["accent"], fontsize=9)

    # Arrows between layers
    _arrow(ax, (7, 6.5), (7, 6.0))
    _arrow(ax, (7, 4.5), (7, 3.8))
    _arrow(ax, (7, 2.8), (7, 2.0))

    plt.tight_layout(pad=0.5)
    out = DOCS_DIR / "architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Generated: {out}")


# ===========================================================================
# 2. Statistical Gate Pipeline
# ===========================================================================

def generate_pipeline():
    """Generate 5-layer statistical gate pipeline visualization."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS["bg"])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg"])

    ax.text(6, 6.6, "5-Layer Statistical Gate Pipeline", ha="center", va="center",
            fontsize=16, color=COLORS["text"], fontweight="bold")

    gates = [
        ("G1", "Hard Constraints", "Risk, trades >= 50, overfit gap", COLORS["red"]),
        ("G2", "Composite Score", "Sharpe + CAGR - MaxDD > 0", COLORS["orange"]),
        ("G3", "PSR", "P(true Sharpe > 0) >= 0.85", COLORS["yellow"]),
        ("G4", "Bootstrap CI", "CI lower > 0, drop-top-K", COLORS["blue"]),
        ("G5", "Walk-Forward + OOS", "3-fold + holdout regime", COLORS["purple"]),
    ]

    x_center = 6
    y_start = 5.5
    box_w, box_h = 5.0, 0.7
    gap = 0.15

    # Candidate entry
    _rounded_box(ax, (x_center - 1.5, y_start + 0.3), 3.0, 0.5,
                 "Candidate Strategy", COLORS["green"], fontsize=11)

    for i, (gate_id, name, desc, color) in enumerate(gates):
        y = y_start - i * (box_h + gap + 0.25)

        # Gate box
        ax.add_patch(FancyBboxPatch((x_center - box_w / 2, y - box_h / 2),
                     box_w, box_h,
                     boxstyle="round,pad=0.03",
                     facecolor=COLORS["panel"], edgecolor=color,
                     linewidth=1.5, alpha=0.85))
        ax.text(x_center - box_w / 2 + 0.2, y, gate_id, ha="left", va="center",
                fontsize=11, color=color, fontweight="bold")
        ax.text(x_center, y + 0.08, name, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
        ax.text(x_center, y - 0.18, desc, ha="center", va="center",
                fontsize=7.5, color=COLORS["subtext"])

        # Arrow from above
        if i == 0:
            _arrow(ax, (x_center, y_start + 0.3), (x_center, y + box_h / 2))
        else:
            prev_y = y_start - (i - 1) * (box_h + gap + 0.25)
            _arrow(ax, (x_center, prev_y - box_h / 2), (x_center, y + box_h / 2))

        # Reject arrow (right)
        rx = x_center + box_w / 2 + 0.3
        ax.annotate("", xy=(rx + 1.2, y), xytext=(x_center + box_w / 2, y),
                    arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.2))
        ax.text(rx + 1.5, y, "REJECT", ha="center", va="center",
                fontsize=8, color=COLORS["red"], fontweight="bold")

    # Promote at bottom
    last_y = y_start - 4 * (box_h + gap + 0.25) - box_h / 2 - 0.3
    _arrow(ax, (x_center, last_y + 0.2), (x_center, last_y - 0.2))
    _rounded_box(ax, (x_center - 1.5, last_y - 0.8), 3.0, 0.5,
                 "PROMOTE", COLORS["green"], fontsize=12)

    plt.tight_layout(pad=0.5)
    out = DOCS_DIR / "pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  Generated: {out}")


# ===========================================================================
# 3. Evolution Loop GIF
# ===========================================================================

def generate_evolution_gif():
    """Generate evolution loop animation GIF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS["bg"])
    fig.subplots_adjust(wspace=0.3)
    for ax in axes:
        ax.set_facecolor(COLORS["bg"])

    rng = np.random.default_rng(42)
    n_gens = 20
    sharpe_history = [0.4]
    promotions = []
    cost_history = [0.0]

    for gen in range(1, n_gens + 1):
        delta = rng.normal(0.02, 0.05)
        if gen in [3, 7, 12, 16]:  # promotion events
            delta = rng.uniform(0.05, 0.12)
            promotions.append(gen)
        sharpe_history.append(sharpe_history[-1] + delta)
        cost_history.append(cost_history[-1] + rng.uniform(1.5, 3.5))

    def animate(frame):
        for ax in axes:
            ax.clear()
            ax.set_facecolor(COLORS["bg"])

        # Left: Sharpe over generations
        ax1 = axes[0]
        gens = list(range(len(sharpe_history[:frame + 1])))
        vals = sharpe_history[:frame + 1]

        ax1.plot(gens, vals, color=COLORS["blue"], lw=2.5, zorder=3)
        ax1.fill_between(gens, vals, alpha=0.15, color=COLORS["blue"])

        # Mark promotions
        for p in promotions:
            if p <= frame:
                ax1.axvline(x=p, color=COLORS["green"], alpha=0.4, lw=1.5, ls="--")
                idx = sharpe_history.index(sharpe_history[p])
                ax1.scatter([p], [sharpe_history[p]], color=COLORS["green"],
                           s=80, zorder=5, edgecolors="white", linewidth=1)

        ax1.set_title("Champion Sharpe Ratio Over Generations",
                      color=COLORS["text"], fontsize=12, fontweight="bold")
        ax1.set_xlabel("Generation", color=COLORS["subtext"])
        ax1.set_ylabel("Sharpe Ratio (test)", color=COLORS["subtext"])
        ax1.tick_params(colors=COLORS["subtext"])
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["bottom"].set_color(COLORS["subtext"])
        ax1.spines["left"].set_color(COLORS["subtext"])
        ax1.grid(axis="y", alpha=0.15, color="white")

        # Right: Status panel
        ax2 = axes[1]
        ax2.axis("off")
        gen = min(frame, n_gens)
        current_sharpe = sharpe_history[gen] if gen < len(sharpe_history) else sharpe_history[-1]
        n_promo = sum(1 for p in promotions if p <= gen)
        n_reject = gen - n_promo
        total_cost = cost_history[gen] if gen < len(cost_history) else cost_history[-1]

        ax2.text(0.5, 0.92, "Evolution Status", ha="center", va="center",
                transform=ax2.transAxes, fontsize=14, color=COLORS["text"],
                fontweight="bold")
        ax2.text(0.5, 0.85, f"Generation: {gen} / {n_gens}", ha="center", va="center",
                transform=ax2.transAxes, fontsize=11, color=COLORS["blue"])

        # Metrics
        metrics = [
            ("Champion Sharpe", f"{current_sharpe:.3f}", COLORS["green"]),
            ("Promotions", f"{n_promo}", COLORS["green"]),
            ("Rejections", f"{n_reject}", COLORS["red"]),
            ("Total Cost", f"${total_cost:.2f}", COLORS["yellow"]),
            ("Lanes", "3", COLORS["blue"]),
        ]
        for i, (label, value, color) in enumerate(metrics):
            y = 0.72 - i * 0.14
            ax2.text(0.2, y, label, ha="left", va="center",
                    transform=ax2.transAxes, fontsize=10, color=COLORS["subtext"])
            ax2.text(0.8, y, value, ha="right", va="center",
                    transform=ax2.transAxes, fontsize=12, color=color,
                    fontweight="bold")

        # Progress bar
        progress = gen / n_gens
        ax2.add_patch(FancyBboxPatch((0.15, 0.05), 0.7, 0.06,
                      transform=ax2.transAxes,
                      boxstyle="round,pad=0.01",
                      facecolor=COLORS["panel"], edgecolor=COLORS["subtext"],
                      linewidth=0.8))
        if progress > 0:
            ax2.add_patch(FancyBboxPatch((0.15, 0.05), 0.7 * progress, 0.06,
                          transform=ax2.transAxes,
                          boxstyle="round,pad=0.01",
                          facecolor=COLORS["blue"], edgecolor="none", alpha=0.8))

        ax2.text(0.5, 0.08, f"{int(progress * 100)}%", ha="center", va="center",
                transform=ax2.transAxes, fontsize=9, color="white", fontweight="bold")

    anim = FuncAnimation(fig, animate, frames=n_gens + 1, interval=400, repeat=True)
    out = DOCS_DIR / "evolution_loop.gif"
    anim.save(out, writer=PillowWriter(fps=3))
    plt.close(fig)
    print(f"  Generated: {out}")


# ===========================================================================
# 4. Multi-Lane Tournament GIF
# ===========================================================================

def generate_tournament_gif():
    """Generate multi-lane tournament selection animation."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    rng = np.random.default_rng(123)
    n_lanes = 3
    n_gens = 8
    lane_names = ["Lane 0", "Lane 1", "Lane 2"]
    lane_colors = [COLORS["blue"], COLORS["purple"], COLORS["orange"]]

    # Pre-generate scores
    scores = {}
    for lane in range(n_lanes):
        scores[lane] = []
        base = 0.3 + lane * 0.05
        for gen in range(n_gens):
            base += rng.normal(0.02, 0.08)
            scores[lane].append(base)

    winners = []
    for gen in range(n_gens):
        gen_scores = [(lane, scores[lane][gen]) for lane in range(n_lanes)]
        winner = max(gen_scores, key=lambda x: x[1])
        winners.append(winner[0])

    def animate(frame):
        ax.clear()
        ax.set_facecolor(COLORS["bg"])

        gen = min(frame, n_gens - 1)
        ax.set_title(f"Multi-Lane Tournament — Generation {gen + 1}/{n_gens}",
                     color=COLORS["text"], fontsize=14, fontweight="bold", pad=15)

        x_pos = np.arange(n_lanes)
        bar_width = 0.6

        for lane in range(n_lanes):
            score = scores[lane][gen]
            is_winner = (winners[gen] == lane)
            color = COLORS["green"] if is_winner else lane_colors[lane]
            alpha = 1.0 if is_winner else 0.6

            bar = ax.bar(x_pos[lane], score, bar_width, color=color,
                        alpha=alpha, edgecolor="white" if is_winner else "none",
                        linewidth=2 if is_winner else 0)

            # Score label
            ax.text(x_pos[lane], score + 0.02, f"{score:.3f}",
                   ha="center", va="bottom", fontsize=10,
                   color=COLORS["text"] if is_winner else COLORS["subtext"],
                   fontweight="bold" if is_winner else "normal")

            # Winner badge
            if is_winner:
                ax.text(x_pos[lane], score - 0.05, "WINNER",
                       ha="center", va="center", fontsize=9,
                       color=COLORS["green"], fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(lane_names, fontsize=11, color=COLORS["text"])
        ax.set_ylabel("Composite Score", color=COLORS["subtext"], fontsize=11)
        ax.tick_params(colors=COLORS["subtext"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(COLORS["subtext"])
        ax.spines["left"].set_color(COLORS["subtext"])
        ax.grid(axis="y", alpha=0.15, color="white")
        ax.set_ylim(0, max(max(scores[l]) for l in range(n_lanes)) * 1.3)

    anim = FuncAnimation(fig, animate, frames=n_gens + 1, interval=600, repeat=True)
    out = DOCS_DIR / "tournament.gif"
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    print(f"  Generated: {out}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("Generating QuantEvoLoop documentation assets...\n")

    print("[1/4] Architecture overview...")
    generate_architecture()

    print("[2/4] Statistical gate pipeline...")
    generate_pipeline()

    print("[3/4] Evolution loop GIF (may take ~30s)...")
    generate_evolution_gif()

    print("[4/4] Tournament selection GIF (may take ~20s)...")
    generate_tournament_gif()

    print(f"\nDone! All assets saved to: {DOCS_DIR}")
    print("\nTo embed in README, add:")
    for f in sorted(DOCS_DIR.glob("*.png")) + sorted(DOCS_DIR.glob("*.gif")):
        rel = f.relative_to(DOCS_DIR.parent)
        print(f"  ![{f.stem}](docs/{f.name})")


if __name__ == "__main__":
    main()
