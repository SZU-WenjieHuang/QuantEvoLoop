"""UCB1 multi-armed bandit for mutation type selection.

Each mutation type is an "arm". UCB1 balances exploration of
untried mutations vs exploitation of known-good ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Arm:
    """One arm of the bandit (one mutation type)."""
    name: str
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / max(1, self.pulls)


class UCBBandit:
    """UCB1 multi-armed bandit for selecting mutation types.

    UCB1 = mean_reward + C × √(ln(N_total) / n_arm)
    Higher C = more exploration. Default C = √2 ≈ 1.414.
    """

    def __init__(self, arm_names: list[str], exploration_constant: float = 1.414):
        self.arms: dict[str, Arm] = {name: Arm(name=name) for name in arm_names}
        self.c = exploration_constant
        self.total_pulls = 0

    def select(self) -> str:
        """Select the arm with highest UCB1 value.
        Unpulled arms get infinite UCB (ensures all are tried at least once)."""
        best_arm = None
        best_ucb = float("-inf")

        for arm in self.arms.values():
            if arm.pulls == 0:
                return arm.name
            exploitation = arm.mean_reward
            exploration = self.c * math.sqrt(math.log(self.total_pulls) / arm.pulls)
            ucb = exploitation + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm

        return best_arm.name if best_arm else list(self.arms.keys())[0]

    def update(self, arm_name: str, reward: float) -> None:
        """Record the reward from pulling an arm."""
        if arm_name in self.arms:
            self.arms[arm_name].pulls += 1
            self.arms[arm_name].total_reward += reward
            self.total_pulls += 1

    def get_arm_stats(self) -> dict[str, dict]:
        return {
            name: {
                "pulls": arm.pulls,
                "mean_reward": round(arm.mean_reward, 4),
                "total_reward": round(arm.total_reward, 4),
            }
            for name, arm in self.arms.items()
        }
