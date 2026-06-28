"""Agent orchestration layer.

Three agent types:
  - LeadAgent: diagnoses weakness, proposes hypotheses, launches campaigns
  - SubAgent: executes mutations in parallel lanes
  - JudgeAgent: evaluates candidates against champion
"""

from .lead_agent import LeadAgent
from .sub_agent import SubAgent
from .judge_agent import JudgeAgent

__all__ = ["LeadAgent", "SubAgent", "JudgeAgent"]
