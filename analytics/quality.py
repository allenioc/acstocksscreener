"""Earnings-quality and balance-sheet health metrics from annual yfinance statements.

Inputs are yfinance-shaped DataFrames: line items as the index, fiscal period end dates as columns.
Every function tolerates missing rows/periods and reports what it could not evaluate instead of
treating missing data as a pass.
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

ROW_ALIASES = {
    "net_income": ["Net Income", "Net Income Common Stockholders",
                   "Net Income From Continuing Operation Net Minority Interest"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                            "Total Cash From Operating Activities"],
    "free_cash_flow": ["Free Cash Flow"],
    "capital_expenditure": ["Capital Expenditure", "Capital Expenditures"],
    "total_assets": ["Total Assets"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "shares": ["Ordinary Shares Number", "Share Issued"],
    "gross_profit": ["Gross Profit"],
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cost_of_revenue": ["Cost Of Revenue", "Reconciled Cost Of Revenue"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "diluted_eps": ["Diluted EPS", "Basic EPS"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "stockholders_equity": ["Stockholders Equity", "Common Stock Equity"],
    "total_debt": ["Total Debt"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "inventory": ["Inventory"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
}

SIGNAL_CATEGORIES = {
    "roa_positive": "Profitability",
    "cfo_positive": "Profitability",
    "accrual": "Profitability",
    "roa_up": "Profitability",
    "leverage_down": "Leverage & Liquidity",
    "current_ratio_up": "Leverage & Liquidity",
    "no_dilution": "Leverage & Liquidity",
    "gross_margin_up": "Operating Efficiency",
    "asset_turnover_up": "Operating Efficiency",
}

# Statement period ends for the same fiscal year can differ by a few days across statements.
PERIOD_MATCH_DAYS = 20
SLOAN_THRESHOLD = 0.10
DIVERGENCE_MULTIPLE = 1.5


def _clean(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def statement_series(df: Optional[pd.DataFrame], key: str) -> pd.Series:
    """Line item as a float Series indexed by period end date (newest first), NaNs dropped."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    lookup = {str(idx).strip().lower(): idx for idx in df.index}
    for alias in ROW_ALIASES[key]:
        idx = lookup.get(alias.lower())
        if idx is None:
            continue
        row = df.loc[idx]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        series = pd.to_numeric(row, errors="coerce").dropna()
        if series.empty:
            continue
        series.index = pd.to_datetime(series.index)
        return series.sort_index(ascending=False)
    return pd.Series(dtype=float)


def _at(series: pd.Series, period: pd.Timestamp) -> Optional[float]:
    """Value for the fiscal period ending near `period`."""
    if series.empty:
        return None
    gaps = abs(series.index - period)
    pos = gaps.argmin()
    if gaps[pos] > pd.Timedelta(days=PERIOD_MATCH_DAYS):
        return None
    return _clean(series.iloc[pos])


def _prior_period(series: pd.Series, period: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Period end roughly one fiscal year before `period`."""
    candidates = [d for d in series.index if pd.Timedelta(days=300) <= period - d <= pd.Timedelta(days=430)]
    return max(candidates) if candidates else None


def _gross_profit(financials_df: Optional[pd.DataFrame]) -> pd.Series:
    gross = statement_series(financials_df, "gross_profit")
    if not gross.empty:
        return gross
    revenue = statement_series(financials_df, "revenue")
    cost = statement_series(financials_df, "cost_of_revenue")
    if revenue.empty or cost.empty:
        return pd.Series(dtype=float)
    return (revenue - cost).dropna().sort_index(ascending=False)


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _period_label(period: Optional[pd.Timestamp]) -> str:
    return f"FY {period.year} ({period.date()})" if period is not None else "n/a"


# ---------------------------------------------------------------------------
# Sloan accrual ratio
# ---------------------------------------------------------------------------

def compute_sloan_ratio(
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    cash_flow_df: Optional[pd.DataFrame],
) -> dict:
    """
    Sloan Ratio = (Net Income - Cash Flow from Operations) / Total Assets, latest annual period.

    Returns a dict with `ratio` (fraction, or None), the inputs used, and a `badge`
    {level: red|green|unavailable, label, detail}.
    """
    net_income = statement_series(financials_df, "net_income")
    cfo = statement_series(cash_flow_df, "operating_cash_flow")
    assets = statement_series(balance_sheet_df, "total_assets")

    result = {"ratio": None, "period": None, "net_income": None, "operating_cash_flow": None, "total_assets": None}
    if net_income.empty or cfo.empty or assets.empty:
        missing = [name for name, s in [("net income", net_income), ("operating cash flow", cfo),
                                        ("total assets", assets)] if s.empty]
        result["badge"] = {"level": "unavailable", "label": "Data unavailable",
                           "detail": f"Missing annual {', '.join(missing)}."}
        return result

    period = net_income.index[0]
    ni, op_cf, ta = _clean(net_income.iloc[0]), _at(cfo, period), _at(assets, period)
    result.update(period=period, net_income=ni, operating_cash_flow=op_cf, total_assets=ta)
    ratio = _ratio(None if ni is None or op_cf is None else ni - op_cf, ta)
    if ratio is None:
        result["badge"] = {"level": "unavailable", "label": "Data unavailable",
                           "detail": f"Statements for {_period_label(period)} do not line up across all three inputs."}
        return result

    result["ratio"] = ratio
    if ratio > SLOAN_THRESHOLD:
        result["badge"] = {"level": "red", "label": "High Accruals / Poor Cash Conversion",
                           "detail": "Earnings may be inflated by non-cash items or one-time gains."}
    elif ratio < -SLOAN_THRESHOLD:
        result["badge"] = {"level": "green", "label": "High Earnings Quality",
                           "detail": "Operating cash flow significantly exceeds net income."}
    else:
        result["badge"] = {"level": "green", "label": "Healthy Cash Conversion",
                           "detail": "Net income is broadly backed by operating cash flow."}
    return result


# ---------------------------------------------------------------------------
# Piotroski F-score
# ---------------------------------------------------------------------------

def _signal(key: str, label: str, passed: Optional[bool], detail: str) -> dict:
    return {"key": key, "label": label, "passed": passed, "detail": detail, "category": SIGNAL_CATEGORIES[key]}


def _fmt_pct(value: Optional[float]) -> str:
    return f"{value * 100:.2f}%" if value is not None else "n/a"


def _fmt_num(value: Optional[float], digits: int = 2) -> str:
    return f"{value:,.{digits}f}" if value is not None else "n/a"


def _fmt_billions(value: Optional[float]) -> str:
    return f"${value / 1e9:,.2f}B" if value is not None else "n/a"


def _yoy_signal(key, label, current, prior, higher_is_better: bool, fmt, missing_detail: str) -> dict:
    if current is None or prior is None:
        return _signal(key, label, None, missing_detail)
    passed = current > prior if higher_is_better else current < prior
    return _signal(key, label, passed, f"{fmt(prior)} → {fmt(current)}")


def compute_piotroski_score(
    financials_df: Optional[pd.DataFrame],
    balance_sheet_df: Optional[pd.DataFrame],
    cash_flow_df: Optional[pd.DataFrame],
) -> dict:
    """
    Piotroski (2000) F-score from the two most recent fiscal years.

    ROA and asset turnover use beginning-of-year total assets as in the original paper; when the
    balance sheet lacks the extra prior year, both years fall back to year-end assets so the
    year-over-year comparison stays like-for-like.

    Returns {score, max_score, evaluated, fiscal_year, prior_year, signals, notes}. Signals whose
    inputs are missing have passed=None and score zero points.
    """
    ni = statement_series(financials_df, "net_income")
    revenue = statement_series(financials_df, "revenue")
    gross = _gross_profit(financials_df)
    cfo = statement_series(cash_flow_df, "operating_cash_flow")
    assets = statement_series(balance_sheet_df, "total_assets")
    ltd = statement_series(balance_sheet_df, "long_term_debt")
    cur_assets = statement_series(balance_sheet_df, "current_assets")
    cur_liab = statement_series(balance_sheet_df, "current_liabilities")
    shares = statement_series(balance_sheet_df, "shares")

    result = {"score": None, "max_score": 9, "evaluated": 0, "fiscal_year": None,
              "prior_year": None, "signals": [], "notes": [], "asset_basis": None}
    if len(ni) < 2:
        result["notes"].append("At least two annual income statements are required.")
        return result

    t, t1 = ni.index[0], _prior_period(ni, ni.index[0])
    if t1 is None:
        result["notes"].append("No prior fiscal year found to compare against.")
        return result
    result["fiscal_year"], result["prior_year"] = _period_label(t), _period_label(t1)

    t2 = _prior_period(assets, t1) if not assets.empty else None
    if t2 is not None and _at(assets, t1) is not None and _at(assets, t2) is not None:
        denom_t, denom_t1 = _at(assets, t1), _at(assets, t2)
        denom_basis = "beginning-of-year assets"
    else:
        denom_t, denom_t1 = _at(assets, t), _at(assets, t1)
        denom_basis = "year-end assets"
        result["notes"].append("Only two balance sheets available; ROA and asset turnover use year-end assets.")
    result["asset_basis"] = denom_basis

    ni_t, ni_t1 = _at(ni, t), _at(ni, t1)
    cfo_t = _at(cfo, t)
    roa_t, roa_t1 = _ratio(ni_t, denom_t), _ratio(ni_t1, denom_t1)

    signals = []
    # Profitability
    signals.append(_signal("roa_positive", "Positive ROA", None if roa_t is None else roa_t > 0,
                           f"ROA {_fmt_pct(roa_t)}" if roa_t is not None
                           else "Net income or total assets unavailable."))
    signals.append(_signal("cfo_positive", "Positive operating cash flow",
                           None if cfo_t is None else cfo_t > 0,
                           f"CFO {_fmt_billions(cfo_t)}" if cfo_t is not None else "Operating cash flow unavailable."))
    signals.append(_signal("accrual", "CFO > net income (accrual test)",
                           None if cfo_t is None or ni_t is None else cfo_t > ni_t,
                           f"CFO {_fmt_billions(cfo_t)} vs NI {_fmt_billions(ni_t)}"
                           if cfo_t is not None and ni_t is not None else "Operating cash flow or net income unavailable."))
    signals.append(_yoy_signal("roa_up", "ROA improved YoY", roa_t, roa_t1, True, _fmt_pct,
                               "Prior-year ROA unavailable."))

    # Leverage, liquidity, dilution
    ltd_t, ltd_t1 = _at(ltd, t), _at(ltd, t1)
    lev_t, lev_t1 = _ratio(ltd_t, _at(assets, t)), _ratio(ltd_t1, _at(assets, t1))
    if lev_t is not None and lev_t1 is not None and lev_t == 0 and lev_t1 == 0:
        signals.append(_signal("leverage_down", "Long-term debt / assets decreased YoY", True,
                               "No long-term debt in either year."))
    else:
        signals.append(_yoy_signal("leverage_down", "Long-term debt / assets decreased YoY", lev_t, lev_t1,
                                   False, _fmt_pct, "Long-term debt not reported for both years."))
    cr_t = _ratio(_at(cur_assets, t), _at(cur_liab, t))
    cr_t1 = _ratio(_at(cur_assets, t1), _at(cur_liab, t1))
    signals.append(_yoy_signal("current_ratio_up", "Current ratio improved YoY", cr_t, cr_t1, True,
                               _fmt_num, "Current assets/liabilities not reported (common for banks and insurers)."))
    sh_t, sh_t1 = _at(shares, t), _at(shares, t1)
    if sh_t is None or sh_t1 is None:
        signals.append(_signal("no_dilution", "No share dilution", None, "Share count unavailable for both years."))
    else:
        signals.append(_signal("no_dilution", "No share dilution", sh_t <= sh_t1,
                               f"{sh_t1 / 1e6:,.1f}M → {sh_t / 1e6:,.1f}M shares"))

    # Operating efficiency
    gm_t = _ratio(_at(gross, t), _at(revenue, t))
    gm_t1 = _ratio(_at(gross, t1), _at(revenue, t1))
    signals.append(_yoy_signal("gross_margin_up", "Gross margin improved YoY", gm_t, gm_t1, True, _fmt_pct,
                               "Gross profit not reported (common for banks and insurers)."))
    at_t, at_t1 = _ratio(_at(revenue, t), denom_t), _ratio(_at(revenue, t1), denom_t1)
    signals.append(_yoy_signal("asset_turnover_up", "Asset turnover improved YoY", at_t, at_t1, True,
                               lambda v: _fmt_num(v, 3), "Revenue or total assets unavailable."))

    result["signals"] = signals
    result["evaluated"] = sum(1 for s in signals if s["passed"] is not None)
    result["score"] = sum(1 for s in signals if s["passed"] is True)
    return result


def piotroski_interpretation(score: Optional[int], evaluated: int) -> str:
    if score is None:
        return "Data unavailable"
    if evaluated < 9:
        return f"{score}/9 with {9 - evaluated} signal(s) not evaluable — treat as a lower bound"
    if score >= 8:
        return "Strong (8–9)"
    if score <= 2:
        return "Weak (0–2)"
    return "Moderate (3–7)"


# ---------------------------------------------------------------------------
# Earnings vs cash divergence
# ---------------------------------------------------------------------------

def build_earnings_divergence(
    financials_df: Optional[pd.DataFrame],
    cash_flow_df: Optional[pd.DataFrame],
    max_periods: int = 5,
) -> dict:
    """
    Net income, operating cash flow and free cash flow per annual period, plus a divergence flag.

    The flag fires when the latest period's net income is positive and exceeds
    1.5x operating cash flow. Free cash flow falls back to CFO + capital expenditure (capex is negative).
    """
    ni = statement_series(financials_df, "net_income")
    cfo = statement_series(cash_flow_df, "operating_cash_flow")
    fcf = statement_series(cash_flow_df, "free_cash_flow")
    capex = statement_series(cash_flow_df, "capital_expenditure")

    periods = sorted(set(ni.index) | set(cfo.index), reverse=True)
    rows = []
    for period in periods:
        if any(abs(period - r["period"]) <= pd.Timedelta(days=PERIOD_MATCH_DAYS) for r in rows):
            continue
        n, c = _at(ni, period), _at(cfo, period)
        f = _at(fcf, period)
        if f is None and c is not None and _at(capex, period) is not None:
            f = c + _at(capex, period)
        if n is None and c is None:
            continue
        rows.append({"period": period, "Net Income": n, "Operating Cash Flow": c, "Free Cash Flow": f})
        if len(rows) == max_periods:
            break

    table = pd.DataFrame(rows).sort_values("period") if rows else pd.DataFrame(
        columns=["period", "Net Income", "Operating Cash Flow", "Free Cash Flow"])

    flag = False
    latest = rows[0] if rows else None
    if latest and latest["Net Income"] is not None and latest["Operating Cash Flow"] is not None:
        flag = latest["Net Income"] > 0 and latest["Net Income"] > DIVERGENCE_MULTIPLE * latest["Operating Cash Flow"]
    return {"table": table, "divergence": flag, "latest": latest}
