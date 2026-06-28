"""Trade-level weakness extractor for backtest results.

Replaces: auto_evolve/scripts/diagnose.py
Pure analysis: no file I/O. Accepts trade lists + strategy summary.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Per-trade derivations
# ---------------------------------------------------------------------------

def mfe_pct(t: dict) -> float:
    """Max favorable excursion as a fraction of open_rate."""
    op = t.get("open_rate") or 0.0
    if op <= 0:
        return 0.0
    if t.get("is_short"):
        mn = t.get("min_rate") or op
        return (op - mn) / op
    mx = t.get("max_rate") or op
    return (mx - op) / op


def mae_pct(t: dict) -> float:
    """Max adverse excursion as a fraction of open_rate."""
    op = t.get("open_rate") or 0.0
    if op <= 0:
        return 0.0
    if t.get("is_short"):
        mx = t.get("max_rate") or op
        return max(0.0, (mx - op) / op)
    mn = t.get("min_rate") or op
    return max(0.0, (op - mn) / op)


def holding_bucket(minutes: int | None) -> str:
    if not minutes:
        return "unknown"
    h = minutes / 60.0
    if h < 8:
        return "<8h"
    if h < 24:
        return "8-24h"
    if h < 72:
        return "1-3d"
    if h < 168:
        return "3-7d"
    return ">7d"


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def by_exit_reason(trades: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0, "_sum_h": 0.0})
    for t in trades:
        r = t.get("exit_reason") or "unknown"
        d = out[r]
        d["n"] += 1
        d["pnl"] += t.get("profit_abs") or 0.0
        if (t.get("profit_ratio") or 0) > 0:
            d["wins"] += 1
        d["_sum_h"] += (t.get("trade_duration") or 0) / 60.0
    for d in out.values():
        d["avg_hold_h"] = round(d["_sum_h"] / max(1, d["n"]), 1)
        d["winrate"] = round(d["wins"] / max(1, d["n"]), 3)
        del d["_sum_h"]
    return dict(out)


def by_direction(trades: list[dict]) -> dict[str, dict]:
    out = {"long": {"n": 0, "pnl": 0.0, "wins": 0},
           "short": {"n": 0, "pnl": 0.0, "wins": 0}}
    for t in trades:
        side = "short" if t.get("is_short") else "long"
        d = out[side]
        d["n"] += 1
        d["pnl"] += t.get("profit_abs") or 0.0
        if (t.get("profit_ratio") or 0) > 0:
            d["wins"] += 1
    for d in out.values():
        d["winrate"] = round(d["wins"] / max(1, d["n"]), 3)
    return out


def by_holding_bucket(trades: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        b = holding_bucket(t.get("trade_duration"))
        d = out[b]
        d["n"] += 1
        d["pnl"] += t.get("profit_abs") or 0.0
        if (t.get("profit_ratio") or 0) > 0:
            d["wins"] += 1
    for d in out.values():
        d["winrate"] = round(d["wins"] / max(1, d["n"]), 3)
    return dict(out)


def by_month(trades: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in trades:
        cd = t.get("close_date", "") or t.get("open_date", "")
        m = cd[:7] if cd else "unknown"
        out[m]["n"] += 1
        out[m]["pnl"] += t.get("profit_abs") or 0.0
    return dict(out)


def mfe_giveback(trades: list[dict], mfe_threshold: float = 0.03,
                 close_threshold: float = 0.01) -> dict:
    matched = [t for t in trades
               if mfe_pct(t) >= mfe_threshold and (t.get("profit_ratio") or 0) < close_threshold]
    loss = sum((t.get("profit_abs") or 0.0) for t in matched)
    return {"count": len(matched), "pnl_attributed": round(loss, 2)}


def mae_recovery(trades: list[dict], mae_threshold: float = 0.02) -> dict:
    matched = [t for t in trades
               if mae_pct(t) >= mae_threshold and (t.get("profit_ratio") or 0) > 0]
    pnl = sum((t.get("profit_abs") or 0.0) for t in matched)
    return {"count": len(matched), "pnl_attributed": round(pnl, 2)}


def funding_burden(trades: list[dict]) -> dict:
    funding = sum((t.get("funding_fees") or 0.0) for t in trades)
    profit = sum((t.get("profit_abs") or 0.0) for t in trades)
    ratio = funding / profit if abs(profit) > 1e-6 else 0.0
    return {"total_funding": round(funding, 2),
            "total_profit": round(profit, 2),
            "funding_to_profit_ratio": round(ratio, 4)}


# ---------------------------------------------------------------------------
# Weakness profile (the headline output for propose-mutation)
# ---------------------------------------------------------------------------

@dataclass
class Weakness:
    rank: int
    issue: str
    detail: str
    trades: int
    loss_share_pct: float
    hint_mutation: str
    hint_param: str


def _total_loss_abs(trades: list[dict]) -> float:
    return sum(abs(t.get("profit_abs") or 0.0) for t in trades if (t.get("profit_ratio") or 0) <= 0) or 1.0


def build_weakness_profile(
    trades: list[dict],
    by_exit: dict,
    by_dir: dict,
    by_hold: dict,
    by_mo: dict,
    mfe_gb: dict,
    mae_rc: dict,
    funding: dict,
) -> list[Weakness]:
    """Heuristic ranking of weaknesses by loss_share_pct. Returns top 5."""
    cands: list[dict] = []
    total_loss = _total_loss_abs(trades)
    n = len(trades)

    # 1. MFE giveback
    if n and mfe_gb["count"] >= 5:
        share_n = mfe_gb["count"] / n
        loss_share = abs(min(0.0, mfe_gb["pnl_attributed"])) / total_loss
        if share_n >= 0.10 or loss_share >= 0.15:
            cands.append({
                "issue": "MFE_giveback_>3%",
                "detail": f"{mfe_gb['count']}/{n} trades hit MFE>=3% but closed <+1%",
                "trades": mfe_gb["count"],
                "loss_share_pct": round(loss_share * 100, 1),
                "hint_mutation": "EXIT_TIGHTEN",
                "hint_param": "TP trailing dd_ratio / TRAILING_SUPPRESS_STAGE",
            })

    # 2. stop_loss heavy
    sl = by_exit.get("stop_loss", {})
    if sl.get("n", 0) >= 5 and sl.get("pnl", 0) < 0:
        loss_share = abs(sl["pnl"]) / total_loss
        if loss_share >= 0.30:
            cands.append({
                "issue": "stop_loss_dominates_losses",
                "detail": f"stop_loss exits: {sl['n']} trades, pnl {sl['pnl']:+.0f}",
                "trades": sl["n"],
                "loss_share_pct": round(loss_share * 100, 1),
                "hint_mutation": "NEW_FILTER",
                "hint_param": "entry guard (vol/ADX/regime)",
            })

    # 3. long/short asymmetry
    L, S = by_dir["long"], by_dir["short"]
    if min(L["n"], S["n"]) >= 10 and L["pnl"] * S["pnl"] < 0:
        losing_side = "long" if L["pnl"] < S["pnl"] else "short"
        losing = by_dir[losing_side]
        loss_share = abs(min(0.0, losing["pnl"])) / total_loss
        if loss_share >= 0.15:
            cands.append({
                "issue": f"directional_asymmetry_{losing_side}_negative",
                "detail": f"{losing_side}: {losing['n']} trades pnl {losing['pnl']:+.0f}",
                "trades": losing["n"],
                "loss_share_pct": round(loss_share * 100, 1),
                "hint_mutation": "BOX_FILTER",
                "hint_param": f"tighten {losing_side}-side confirmation",
            })

    # 4. Monthly concentration
    if by_mo:
        sorted_mo = sorted(by_mo.items(), key=lambda kv: kv[1]["pnl"])
        worst3 = sorted_mo[:3]
        worst3_loss = sum(m[1]["pnl"] for m in worst3 if m[1]["pnl"] < 0)
        if abs(worst3_loss) / total_loss >= 0.40 and len(worst3) >= 2:
            cands.append({
                "issue": "losses_concentrated_in_few_months",
                "detail": f"worst 3 months absorb {abs(worst3_loss)/total_loss*100:.0f}% of losses",
                "trades": sum(m[1]["n"] for m in worst3),
                "loss_share_pct": round(abs(worst3_loss) / total_loss * 100, 1),
                "hint_mutation": "NEW_FILTER",
                "hint_param": "regime detector (Choppiness Index / ADX)",
            })

    # 5. Holding-time concentration
    losers_by_hold = {b: d for b, d in by_hold.items() if d["pnl"] < 0}
    if losers_by_hold:
        worst_bucket = max(losers_by_hold.items(), key=lambda kv: abs(kv[1]["pnl"]))
        b, d = worst_bucket
        loss_share = abs(d["pnl"]) / total_loss
        if loss_share >= 0.30:
            if b in ("<8h", "8-24h"):
                hint_mut, hint_par = "COOLDOWN", f"bucket {b}: raise COOLDOWN_BARS"
            elif b == ">7d":
                hint_mut, hint_par = "EXIT_TIGHTEN", f"bucket {b}: shorten TIME_STOP_HOURS"
            else:
                hint_mut, hint_par = "BOX_FILTER", f"bucket {b}: tighten entry confirmation"
            cands.append({
                "issue": f"losses_in_holding_bucket_{b}",
                "detail": f"bucket {b}: {d['n']} trades, pnl {d['pnl']:+.0f}",
                "trades": d["n"],
                "loss_share_pct": round(loss_share * 100, 1),
                "hint_mutation": hint_mut,
                "hint_param": hint_par,
            })

    # 6. Funding burden
    fr = funding["funding_to_profit_ratio"]
    if abs(fr) >= 0.10 and abs(funding["total_funding"]) >= 50:
        cands.append({
            "issue": "funding_cost_significant",
            "detail": f"funding {funding['total_funding']:+.0f} vs profit {funding['total_profit']:+.0f}",
            "trades": n,
            "loss_share_pct": round(abs(fr) * 100, 1),
            "hint_mutation": "EXIT_TIGHTEN",
            "hint_param": "shorten avg holding or raise COOLDOWN",
        })

    # 7. MAE recovery
    if mae_rc["count"] >= 5 and n:
        share_n = mae_rc["count"] / n
        if share_n >= 0.20:
            cands.append({
                "issue": "MAE_recovery_>2%",
                "detail": f"{mae_rc['count']}/{n} trades dipped >=2% then recovered",
                "trades": mae_rc["count"],
                "loss_share_pct": round(share_n * 100, 1),
                "hint_mutation": "EXIT_LOOSEN",
                "hint_param": "widen ATR stop multiplier",
            })

    cands.sort(key=lambda c: -c["loss_share_pct"])
    result = []
    for i, c in enumerate(cands[:5], 1):
        result.append(Weakness(
            rank=i,
            issue=c["issue"],
            detail=c["detail"],
            trades=c["trades"],
            loss_share_pct=c["loss_share_pct"],
            hint_mutation=c["hint_mutation"],
            hint_param=c["hint_param"],
        ))
    return result


@dataclass
class DiagnosticReport:
    segment: str
    n_trades: int
    summary: dict[str, Any]
    by_exit_reason: dict
    by_direction: dict
    by_holding_bucket: dict
    by_month: dict
    mfe_giveback: dict
    mae_recovery: dict
    funding_burden: dict
    weaknesses: list[Weakness]


def diagnose(
    trades: list[dict],
    strategy_summary: dict[str, Any],
    segment: str = "test",
) -> DiagnosticReport:
    """Run full diagnostic analysis on a set of trades."""
    be = by_exit_reason(trades)
    bd = by_direction(trades)
    bh = by_holding_bucket(trades)
    bm = by_month(trades)
    mg = mfe_giveback(trades)
    mr = mae_recovery(trades)
    fb = funding_burden(trades)
    wp = build_weakness_profile(trades, be, bd, bh, bm, mg, mr, fb)

    return DiagnosticReport(
        segment=segment,
        n_trades=len(trades),
        summary=strategy_summary,
        by_exit_reason=be,
        by_direction=bd,
        by_holding_bucket=bh,
        by_month=bm,
        mfe_giveback=mg,
        mae_recovery=mr,
        funding_burden=fb,
        weaknesses=wp,
    )
