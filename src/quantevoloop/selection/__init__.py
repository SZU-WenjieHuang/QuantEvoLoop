"""Selection layer — RL-inspired tournament and UCB selection.

Modules:
  - tournament: Multi-lane tournament selection
  - ucb: UCB1 exploration/exploitation arm selection
  - population: Candidate population management
  - reward: Reward signal computation for selection feedback
"""

from .tournament import Tournament
from .ucb import UCBBandit, Arm
from .population import Population, Candidate
from .reward import compute_reward

__all__ = [
    "Tournament",
    "UCBBandit", "Arm",
    "Population", "Candidate",
    "compute_reward",
]
