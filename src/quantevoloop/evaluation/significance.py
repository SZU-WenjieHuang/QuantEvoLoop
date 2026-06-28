"""Statistical significance tests for backtest results.

Implements (López de Prado, 2014; Bailey & López de Prado, 2014):
  - Probabilistic Sharpe Ratio (PSR)
  - Bootstrap CI for per-trade Sharpe
  - Deflated Sharpe Ratio (DSR) — multi-testing correction
  - Drop-top-K concentration risk gate

All functions are pure (no file I/O). They take numpy arrays and return dicts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Trade extraction helpers
# ---------------------------------------------------------------------------

def extract_trade_returns_from_dict(strategy_data: dict) -> np.ndarray:
    """Extract per-trade profit_ratio from a strategy-section dict."""
    trades = strategy_data.get("trades", [])
    return np.array([t.get("profit_ratio", 0.0) for t in trades], dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-trade Sharpe
# ---------------------------------------------------------------------------

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def trade_sharpe(returns: np.ndarray) -> float:
    """Per-trade Sharpe (mean / std). NOT annualized."""
    if len(returns) < 2:
        return 0.0
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    if sigma <= 1e-12:
        return 0.0
    return mu / sigma


def annualized_sharpe(returns: np.ndarray, backtest_days: float) -> float:
    """Annualized Sharpe: SR_per_trade × √(trades/year)."""
    n = len(returns)
    if n < 2 or backtest_days <= 0:
        return 0.0
    trades_per_year = n / float(backtest_days) * 365.0
    return trade_sharpe(returns) * math.sqrt(trades_per_year)


# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio (PSR)
# ---------------------------------------------------------------------------

def probabilistic_sharpe(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
) -> dict[str, Any]:
    """PSR (Bailey & López de Prado 2012).

    PSR(SR*) = Φ( (SR_obs - SR*) · √(N-1) / √(1 - γ₃·SR_obs + (γ₄-1)/4·SR_obs²) )

    Returns: psr, sharpe_observed, n_trades, skew, kurtosis, sr_benchmark.
    """
    n = len(returns)
    if n < 2:
        return {
            "psr": 0.0, "sharpe_observed": 0.0, "n_trades": n,
            "skew": 0.0, "kurtosis": 3.0, "sr_benchmark": sr_benchmark,
            "note": "insufficient_trades",
        }

    sr = trade_sharpe(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma <= 1e-12:
        return {
            "psr": 0.0, "sharpe_observed": sr, "n_trades": n,
            "skew": 0.0, "kurtosis": 3.0, "sr_benchmark": sr_benchmark,
            "note": "zero_variance",
        }

    z = (returns - mu) / sigma
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())  # full kurtosis (Gaussian = 3)

    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)
    if denom_sq <= 1e-9:
        return {
            "psr": 0.5, "sharpe_observed": sr, "n_trades": n,
            "skew": skew, "kurtosis": kurt, "sr_benchmark": sr_benchmark,
            "note": "denominator_unstable",
        }

    z_score = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    psr = _normal_cdf(z_score)
    return {
        "psr": float(psr),
        "sharpe_observed": float(sr),
        "n_trades": int(n),
        "skew": skew,
        "kurtosis": kurt,
        "sr_benchmark": float(sr_benchmark),
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.90,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap CI for the per-trade Sharpe ratio."""
    n = len(returns)
    if n < 2:
        return {
            "sharpe_ci_lo": 0.0, "sharpe_ci_hi": 0.0,
            "sharpe_median": 0.0, "n_resamples": 0,
            "note": "insufficient_trades",
        }

    rng = np.random.default_rng(seed)
    sharps = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(returns, size=n, replace=True)
        sharps[i] = trade_sharpe(sample)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(sharps, alpha))
    hi = float(np.quantile(sharps, 1.0 - alpha))
    med = float(np.median(sharps))
    return {
        "sharpe_ci_lo": lo,
        "sharpe_ci_hi": hi,
        "sharpe_median": med,
        "n_resamples": n_resamples,
        "ci_level": ci,
    }


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio (DSR)
# ---------------------------------------------------------------------------

def _inv_normal_cdf(p: float) -> float:
    """Acklam's rational approximation for Φ⁻¹(p). Good to ~1e-9."""
    if p <= 0.0 or p >= 1.0:
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def deflated_sharpe(
    returns: np.ndarray,
    n_trials: int,
) -> dict[str, Any]:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Adjusts observed SR for selection bias from running N independent trials.
    """
    n = len(returns)
    if n < 2 or n_trials < 1:
        return {
            "dsr": 0.0, "expected_max_sr_under_null": 0.0,
            "sharpe_observed": 0.0, "n_trials": n_trials,
            "note": "insufficient_data",
        }

    sr = trade_sharpe(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma <= 1e-12:
        return {
            "dsr": 0.0, "expected_max_sr_under_null": 0.0,
            "sharpe_observed": sr, "n_trials": n_trials,
            "note": "zero_variance",
        }

    z = (returns - mu) / sigma
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())

    # E[max SR | N independent trials, true SR=0]
    if n_trials < 2:
        e_max_sr = 0.0
    else:
        z1 = _inv_normal_cdf(1.0 - 1.0 / n_trials)
        z2 = _inv_normal_cdf(1.0 - 1.0 / (n_trials * math.e))
        gamma = 0.5772156649
        e_max_sr = ((1 - gamma) * z1 + gamma * z2) / math.sqrt(n)

    denom_sq = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)
    if denom_sq <= 1e-9:
        return {
            "dsr": 0.5, "expected_max_sr_under_null": float(e_max_sr),
            "sharpe_observed": float(sr), "n_trials": n_trials,
            "note": "denominator_unstable",
        }

    z_score = (sr - e_max_sr) * math.sqrt(n - 1) / math.sqrt(denom_sq)
    dsr = _normal_cdf(z_score)
    return {
        "dsr": float(dsr),
        "expected_max_sr_under_null": float(e_max_sr),
        "sharpe_observed": float(sr),
        "n_trials": int(n_trials),
        "skew": skew,
        "kurtosis": kurt,
    }
