"""Champion promotion — swap candidate strategy as new champion.

Handles: strategy file copy, metrics update, ancestry tracking.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromotionResult:
    success: bool
    gen_id: int
    reason: str = ""
    old_champion_gen: int = 0
    new_champion_gen: int = 0


class Promoter:
    """Manages champion promotion when a candidate passes all gates."""

    def __init__(self, champion_dir: Path, generations_dir: Path, ancestry_path: Path | None = None):
        self.champion_dir = champion_dir
        self.generations_dir = generations_dir
        self.ancestry_path = ancestry_path or (champion_dir / "ancestry.md")

    def promote(
        self,
        gen_id: int,
        score: float,
        cand_train: dict[str, Any],
        cand_test: dict[str, Any],
        holdout: dict[str, Any] | None = None,
    ) -> PromotionResult:
        gen_dir = self.generations_dir / f"gen_{gen_id:04d}"
        strategy_src = gen_dir / "strategy.py"

        if not strategy_src.exists():
            return PromotionResult(
                success=False, gen_id=gen_id,
                reason=f"strategy.py not found in {gen_dir}",
            )

        # Read old champion info
        old_gen = 0
        metrics_path = self.champion_dir / "metrics.json"
        if metrics_path.exists():
            old_data = json.loads(metrics_path.read_text())
            old_gen = old_data.get("generation", 0)

        # Copy strategy to champion
        self.champion_dir.mkdir(parents=True, exist_ok=True)
        champion_strategy = self.champion_dir / "strategy.py"
        shutil.copy2(strategy_src, champion_strategy)

        # Write champion metrics
        metrics = {
            "generation": gen_id,
            "score": score,
            "train": cand_train,
            "test": cand_test,
            "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        if holdout:
            metrics["holdout"] = holdout

        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

        # Append to ancestry
        note = (
            f"\n- gen_{gen_id:04d} promoted at "
            f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} | "
            f"score={score:+.4f} | "
            f"test_sharpe={cand_test.get('sharpe', 0):.4f} | "
            f"train_sharpe={cand_train.get('sharpe', 0):.4f}"
        )
        if holdout:
            note += f" | holdout_sharpe={holdout.get('sharpe', 0):.4f}"
        self.ancestry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ancestry_path.open("a") as f:
            f.write(note + "\n")

        return PromotionResult(
            success=True,
            gen_id=gen_id,
            reason=f"gen_{old_gen:04d} → gen_{gen_id:04d}",
            old_champion_gen=old_gen,
            new_champion_gen=gen_id,
        )
