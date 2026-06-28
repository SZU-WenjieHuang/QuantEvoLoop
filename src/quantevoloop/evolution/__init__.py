"""Evolution core layer.

Modules:
  - state: Evolution state management (state.json)
  - campaign: Campaign lifecycle management
  - knowledge: Cross-campaign knowledge accumulation
  - dead_ends: Dead-end tracking and deduplication
  - promoter: Champion promotion and ancestry tracking
"""

from .state import EvolutionState, GenerationRecord
from .campaign import Campaign, CampaignStatus
from .knowledge import KnowledgeBase, MutationRecord
from .dead_ends import DeadEndTracker
from .promoter import Promoter

__all__ = [
    "EvolutionState", "GenerationRecord",
    "Campaign", "CampaignStatus",
    "KnowledgeBase", "MutationRecord",
    "DeadEndTracker",
    "Promoter",
]
