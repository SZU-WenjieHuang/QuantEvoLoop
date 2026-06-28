"""Evaluation layer — 5-layer statistical gate pipeline.

Modules:
  - significance: PSR, Bootstrap CI, DSR
  - scorer: composite score + hard constraints + decide cascade
  - robustness: drop-top-K concentration risk
  - walkforward: 3-fold rolling walk-forward validation
  - holdout: OOS regime-robustness check
  - diagnostics: trade-level weakness extraction
"""

from .scorer import (
    DecideResult,
    HardConstraintResult,
    ScoreResult,
    check_hard_constraints,
    compute_score,
    decide,
    run_significance_checks,
)
from .significance import (
    annualized_sharpe,
    bootstrap_sharpe_ci,
    deflated_sharpe,
    probabilistic_sharpe,
    trade_sharpe,
)
from .robustness import check_robustness, drop_top_k_sharpe
from .walkforward import (
    DEFAULT_FOLDS,
    FoldResult,
    WalkForwardResult,
    WalkForwardSummary,
    summarize_folds,
    validate_walkforward,
)
from .holdout import HoldoutResult, check_holdout
from .diagnostics import DiagnosticReport, Weakness, diagnose

__all__ = [
    "DecideResult", "HardConstraintResult", "ScoreResult",
    "check_hard_constraints", "compute_score", "decide", "run_significance_checks",
    "annualized_sharpe", "bootstrap_sharpe_ci", "deflated_sharpe",
    "probabilistic_sharpe", "trade_sharpe",
    "check_robustness", "drop_top_k_sharpe",
    "DEFAULT_FOLDS", "FoldResult", "WalkForwardResult", "WalkForwardSummary",
    "summarize_folds", "validate_walkforward",
    "HoldoutResult", "check_holdout",
    "DiagnosticReport", "Weakness", "diagnose",
]
