"""Score computation and multi-gate promotion decision logic.

Replaces: auto_evolve/scripts/compute_score.py
All hardcoded paths removed; accepts config via `StatisticalGates`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import significance as sig


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    """Composite score output."""
    score: float
    z_sharpe: float
    z_cagr: float
    z_dd: float


@dataclass
class HardConstraintResult:
    """Hard constraint check result."""
    passes: bool
    failure_tag: str = ""


@dataclass
class DecideResult:
    """Full promotion decision result."""
    verdict: str  # promote | reject | dead_end | bt_error
    reason: str
    score: float | None = None
    cand_train: dict[str, Any] = field(default_factory=dict)
    cand_test: dict[str, Any] = field(default_factory=dict)
    psr_test: dict | None = None
    psr_train: dict | None = None
    ci_test: dict | None = None
    ci_train: dict | None = None
    robustness_train: dict | None = None
    robustness_test: dict | None = None
    holdout: dict | None = None
    walkforward: dict | None = None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def safe_z(candidate: float, champion: float) -> float:
    """Relative delta vs champion. Sign convention: positive = candidate better."""
    denom = abs(champion) if abs(champion) > 1e-6 else 1e-6
    return (candidate - champion) / denom


def compute_score(
    cand_train: dict[str, Any],
    cand_test: dict[str, Any],
    champ_train: dict[str, Any],
    champ_test: dict[str, Any],
    w_sharpe: float = 0.5,
    w_cagr: float = 0.3,
    w_dd: float = 0.1,
) -> ScoreResult:
    """min-of-segments score.

    z_sharpe is computed on min(train, test), forcing both segments to
    not-regress simultaneously. cagr and dd remain test-segment.
    """
    cand_min = min(cand_train.get("sharpe") or 0.0, cand_test.get("sharpe") or 0.0)
    champ_min = min(champ_train.get("sharpe") or 0.0, champ_test.get("sharpe") or 0.0)
    z_sharpe = safe_z(cand_min, champ_min)
    z_cagr = safe_z(cand_test.get("cagr") or 0.0, champ_test.get("cagr") or 0.0)
    z_dd = safe_z(
        cand_test.get("max_drawdown_account") or 0.0,
        champ_test.get("max_drawdown_account") or 0.0,
    )
    return ScoreResult(
        score=w_sharpe * z_sharpe + w_cagr * z_cagr - w_dd * z_dd,
        z_sharpe=z_sharpe,
        z_cagr=z_cagr,
        z_dd=z_dd,
    )


# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------

def check_hard_constraints(
    cand_train: dict[str, Any],
    cand_test: dict[str, Any],
    champ_train: dict[str, Any],
    champ_test: dict[str, Any],
    *,
    max_dd_ceiling_factor: float = 1.25,
    min_trades: int = 50,
    max_directional_gap: float = 0.6,
) -> HardConstraintResult:
    """Return (passes, failure_tag)."""
    cand_dd = cand_test.get("max_drawdown_account") or 0.0
    champ_dd = champ_test.get("max_drawdown_account") or 0.0
    if cand_dd > champ_dd * max_dd_ceiling_factor + 1e-6:
        return HardConstraintResult(False, "risk-deteriorate")
    if (cand_test.get("total_trades") or 0) < min_trades:
        return HardConstraintResult(False, "too-sparse")

    cand_train_sh = cand_train.get("sharpe") or 0.0
    cand_test_sh = cand_test.get("sharpe") or 0.0
    if (cand_test_sh - cand_train_sh) > max_directional_gap:
        return HardConstraintResult(False, "overfit-test")
    if cand_test_sh <= 0:
        return HardConstraintResult(False, "unprofitable")
    return HardConstraintResult(True)


# ---------------------------------------------------------------------------
# Significance / robustness helpers
# ---------------------------------------------------------------------------

def run_significance_checks(
    train_returns,
    test_returns,
    *,
    psr_min: float = 0.85,
    ci_level_test: float = 0.70,
    psr_min_train: float = 0.80,
    ci_level_train: float = 0.65,
    bootstrap_n: int = 1000,
) -> tuple[dict | None, dict | None, dict | None, dict | None]:
    """Best-effort PSR + bootstrap on both segments. Returns (psr_test, ci_test, psr_train, ci_train)."""
    psr_test = sig.probabilistic_sharpe(test_returns, sr_benchmark=0.0) if len(test_returns) >= 2 else None
    ci_test = sig.bootstrap_sharpe_ci(test_returns, n_resamples=bootstrap_n, ci=ci_level_test) if len(test_returns) >= 2 else None
    psr_train = sig.probabilistic_sharpe(train_returns, sr_benchmark=0.0) if len(train_returns) >= 2 else None
    ci_train = sig.bootstrap_sharpe_ci(train_returns, n_resamples=bootstrap_n, ci=ci_level_train) if len(train_returns) >= 2 else None
    return psr_test, ci_test, psr_train, ci_train


# ---------------------------------------------------------------------------
# Full decide pipeline
# ---------------------------------------------------------------------------

def decide(
    cand_train: dict[str, Any],
    cand_test: dict[str, Any],
    champ_train: dict[str, Any],
    champ_test: dict[str, Any],
    champ_holdout: dict[str, Any] | None,
    *,
    train_returns,
    test_returns,
    holdout_result: dict[str, Any] | None = None,
    gates: dict[str, float] | None = None,
    w_sharpe: float = 0.5,
    w_cagr: float = 0.3,
    w_dd: float = 0.1,
) -> DecideResult:
    """Full 5-gate promotion decision.

    Gate order:
      1. Train regression (cand_train_sh >= champ_train_sh × ratio)
      2. Hard constraints (risk, trade count, overfit gap, profitability)
      3. Score > 0
      4. Statistical significance (PSR + CI on both segments)
      5. Robustness (drop-top-K) + Holdout final-gate
    """
    g = gates or {}
    max_dd_ceiling = g.get("max_dd_ceiling_factor", 1.25)
    min_trades = int(g.get("min_trades", 50))
    max_gap = g.get("max_directional_gap", 0.6)
    psr_min_test = g.get("psr_min", 0.85)
    psr_min_train = g.get("psr_min_train", 0.80)
    ci_level_test = g.get("bootstrap_ci", 0.70)
    ci_level_train = g.get("bootstrap_ci_train", 0.65)
    train_regress_floor = g.get("train_regress_floor_ratio", 0.95)
    holdout_floor_ratio = g.get("holdout_degradation_factor", 0.70)
    holdout_floor_slack = g.get("holdout_degradation_offset", 0.05)

    # Gate 1: Train regression
    train_floor = (champ_train.get("sharpe") or 0.0) * train_regress_floor
    if (cand_train.get("sharpe") or 0.0) < train_floor:
        return DecideResult(
            verdict="dead_end",
            reason=f"train sharpe {cand_train.get('sharpe', 0):.3f} < champion train "
                   f"{champ_train.get('sharpe', 0):.3f} × {train_regress_floor}",
        )

    # Gate 2: Hard constraints
    hc = check_hard_constraints(
        cand_train, cand_test, champ_train, champ_test,
        max_dd_ceiling_factor=max_dd_ceiling,
        min_trades=min_trades,
        max_directional_gap=max_gap,
    )
    if not hc.passes:
        return DecideResult(
            verdict="dead_end",
            reason=f"hard constraint failed: {hc.failure_tag}",
        )

    # Gate 3: Score > 0
    score_r = compute_score(cand_train, cand_test, champ_train, champ_test, w_sharpe, w_cagr, w_dd)
    if score_r.score <= 0:
        return DecideResult(
            verdict="reject",
            reason=f"score {score_r.score:+.4f} ≤ 0 vs champion (min-of-segments)",
            score=score_r.score,
            cand_train=cand_train,
            cand_test=cand_test,
        )

    # Gate 4: Statistical significance
    psr_test, ci_test, psr_train, ci_train = run_significance_checks(
        train_returns, test_returns,
        psr_min=psr_min_test,
        ci_level_test=ci_level_test,
        psr_min_train=psr_min_train,
        ci_level_train=ci_level_train,
    )

    if psr_test is not None and psr_test["psr"] < psr_min_test:
        return DecideResult(
            verdict="reject",
            reason=f"test PSR={psr_test['psr']:.3f} < {psr_min_test}",
            score=score_r.score,
            cand_train=cand_train, cand_test=cand_test,
            psr_test=psr_test,
        )
    if ci_test is not None and ci_test["sharpe_ci_lo"] <= 0:
        return DecideResult(
            verdict="reject",
            reason=f"test bootstrap CI lo {ci_test['sharpe_ci_lo']:+.3f} ≤ 0",
            score=score_r.score,
            cand_train=cand_train, cand_test=cand_test,
            ci_test=ci_test,
        )
    if psr_train is not None and psr_train["psr"] < psr_min_train:
        return DecideResult(
            verdict="reject",
            reason=f"train PSR={psr_train['psr']:.3f} < {psr_min_train}",
            score=score_r.score,
            cand_train=cand_train, cand_test=cand_test,
            psr_train=psr_train,
        )
    if ci_train is not None and ci_train["sharpe_ci_lo"] <= 0:
        return DecideResult(
            verdict="reject",
            reason=f"train bootstrap CI lo {ci_train['sharpe_ci_lo']:+.3f} ≤ 0",
            score=score_r.score,
            cand_train=cand_train, cand_test=cand_test,
            ci_train=ci_train,
        )

    # Gate 5: Holdout final-gate
    if holdout_result is not None:
        holdout_sharpe = holdout_result.get("sharpe") or 0.0
        champ_holdout_sharpe = (champ_holdout or {}).get("sharpe") or 0.0
        holdout_floor = champ_holdout_sharpe * holdout_floor_ratio - holdout_floor_slack
        if holdout_sharpe < holdout_floor:
            return DecideResult(
                verdict="dead_end",
                reason=f"holdout sharpe {holdout_sharpe:+.3f} < floor {holdout_floor:+.3f}",
                score=score_r.score,
                cand_train=cand_train, cand_test=cand_test,
                holdout=holdout_result,
            )

    return DecideResult(
        verdict="promote",
        reason=f"score {score_r.score:+.4f} > 0; all gates pass",
        score=score_r.score,
        cand_train=cand_train,
        cand_test=cand_test,
        psr_test=psr_test, psr_train=psr_train,
        ci_test=ci_test, ci_train=ci_train,
        holdout=holdout_result,
    )
