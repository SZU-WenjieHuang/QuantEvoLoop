"""Reward signal computation for UCB bandit feedback.

Converts evaluation outcomes (verdict + score) into a scalar reward
signal that the UCB1 bandit uses to update mutation-type preferences.
"""

from __future__ import annotations


# Reward values by verdict
REWARD_MAP = {
    "promoted": 1.0,
    "reject": 0.0,
    "not-better": -0.2,
    "not-significant": -0.1,
    "not-robust": -0.3,
    "dead_end": -0.5,
    "regime-fragile": -0.5,
    "regress": -0.4,
    "risk-deteriorate": -0.6,
    "unprofitable": -0.5,
    "too-sparse": -0.3,
    "overfit-test": -0.5,
    "bt-error": -1.0,
}


def compute_reward(verdict: str, score: float | None = None) -> float:
    """Compute a scalar reward for the UCB bandit.

    - promoted: +1.0 (strong positive signal)
    - reject variants: small negative (exploration penalty)
    - dead_end/bt-error: larger negative (avoid this direction)
    - Score bonus: if promoted, add score * 0.5 as bonus
    """
    base = REWARD_MAP.get(verdict, 0.0)
    bonus = 0.0
    if verdict == "promoted" and score is not None:
        bonus = max(0.0, score) * 0.5
    return base + bonus
