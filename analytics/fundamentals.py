"""Statement-derived fallbacks for quote fundamentals that Yahoo's `.info` did not return.

Statement inputs are stored as `fs_*` fields so price-dependent ratios (market cap, P/E, P/S...)
can be recomputed against the latest price without refetching statements. Fallbacks only fill
fields that are missing; values Yahoo reports are never overwritten.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from analytics.quality import _clean, statement_series

# field -> (statement, line-item alias key, fiscal years back from latest)
STATEMENT_INPUTS: dict[str, tuple[str, str, int]] = {
    "fs_shares": ("balance", "shares", 0),
    "fs_diluted_eps": ("income", "diluted_eps", 0),
    "fs_diluted_eps_prev": ("income", "diluted_eps", 1),
    "fs_revenue": ("income", "revenue", 0),
    "fs_revenue_prev": ("income", "revenue", 1),
    "fs_operating_income": ("income", "operating_income", 0),
    "fs_net_income": ("income", "net_income", 0),
    "fs_ebitda": ("income", "ebitda", 0),
    "fs_equity": ("balance", "stockholders_equity", 0),
    "fs_total_assets": ("balance", "total_assets", 0),
    "fs_current_assets": ("balance", "current_assets", 0),
    "fs_current_liabilities": ("balance", "current_liabilities", 0),
    "fs_inventory": ("balance", "inventory", 0),
    "fs_total_debt": ("balance", "total_debt", 0),
    "fs_cash": ("balance", "cash", 0),
    "fs_dividends_paid": ("cashflow", "dividends_paid", 0),
}
# Trailing-twelve-month sums from the last four quarterly income statements.
TTM_INPUTS: dict[str, str] = {
    "fs_ttm_eps": "diluted_eps",
    "fs_ttm_revenue": "revenue",
    "fs_ttm_net_income": "net_income",
    "fs_ttm_operating_income": "operating_income",
}
STATEMENT_INPUT_FIELDS = list(STATEMENT_INPUTS) + list(TTM_INPUTS)

DERIVED_FIELD_LABELS = {
    "market_cap": "Market cap",
    "pe": "P/E",
    "ps": "P/S",
    "pb": "P/B",
    "ev_ebitda": "EV/EBITDA",
    "operating_margin": "Operating margin",
    "profit_margin": "Profit margin",
    "roe": "ROE",
    "roa": "ROA",
    "current_ratio": "Current ratio",
    "quick_ratio": "Quick ratio",
    "debt_to_equity": "Debt/equity",
    "revenue_growth": "Revenue growth",
    "earnings_growth": "EPS growth",
    "dividend_yield": "Dividend yield",
    "beta": "Beta (1Y daily vs SPY)",
}


def _value(series: pd.Series, years_back: int) -> Optional[float]:
    """Latest value, or the value one fiscal year earlier (rejecting gaps that are not ~1 year)."""
    if len(series) <= years_back:
        return None
    if years_back == 1:
        gap = (series.index[0] - series.index[1]).days
        if not 300 <= gap <= 430:
            return None
    return _clean(series.iloc[years_back])


def _ttm(series: pd.Series) -> Optional[float]:
    """Sum of the four most recent quarters, only if they span roughly one year."""
    if len(series) < 4:
        return None
    if (series.index[0] - series.index[3]).days > 300:
        return None
    values = [_clean(v) for v in series.iloc[:4]]
    return None if any(v is None for v in values) else sum(values)


def extract_statement_inputs(
    income: Optional[pd.DataFrame],
    balance: Optional[pd.DataFrame],
    cashflow: Optional[pd.DataFrame],
    quarterly_income: Optional[pd.DataFrame] = None,
) -> dict:
    """`fs_*` line items from annual statements plus TTM income items (None where not reported)."""
    frames = {"income": income, "balance": balance, "cashflow": cashflow}
    inputs = {
        field: _value(statement_series(frames[stmt], key), years_back)
        for field, (stmt, key, years_back) in STATEMENT_INPUTS.items()
    }
    for field, key in TTM_INPUTS.items():
        inputs[field] = _ttm(statement_series(quarterly_income, key))
    return inputs


def _ratio(numerator: Optional[float], denominator: Optional[float], positive_denominator: bool = True):
    if numerator is None or denominator is None or denominator == 0:
        return None
    if positive_denominator and denominator < 0:
        return None
    return numerator / denominator


def _pct(numerator, denominator) -> Optional[float]:
    r = _ratio(numerator, denominator)
    return r * 100 if r is not None else None


def _growth(current, previous) -> Optional[float]:
    if current is None or previous is None or previous <= 0:
        return None
    return (current / previous - 1) * 100


def derive_fallback_metrics(row: dict) -> tuple[dict, list[str]]:
    """
    Fill missing quote metrics from `fs_*` statement inputs, `price` and `beta_1y`.

    Units follow the screener's conventions: margins/returns/growth/yield in percent,
    debt/equity as Yahoo reports it (percent, e.g. 150 = 1.5x).
    Returns (updates, names of fields that were derived).
    """
    def get(key):
        return _clean(row.get(key))

    updates: dict[str, float] = {}

    def fill(field: str, value: Optional[float]) -> None:
        if value is not None and get(field) is None:
            updates[field] = value

    def current(field: str) -> Optional[float]:
        return updates.get(field, get(field))

    def ttm_or_annual(ttm_key: str, annual_key: str) -> Optional[float]:
        value = get(ttm_key)
        return value if value is not None else get(annual_key)

    price = get("price")
    shares = get("fs_shares")
    eps = ttm_or_annual("fs_ttm_eps", "fs_diluted_eps")
    revenue = ttm_or_annual("fs_ttm_revenue", "fs_revenue")
    net_income = ttm_or_annual("fs_ttm_net_income", "fs_net_income")
    operating_income = ttm_or_annual("fs_ttm_operating_income", "fs_operating_income")
    equity, assets = get("fs_equity"), get("fs_total_assets")
    debt, cash = get("fs_total_debt"), get("fs_cash")
    current_assets, current_liabilities = get("fs_current_assets"), get("fs_current_liabilities")
    inventory, ebitda = get("fs_inventory"), get("fs_ebitda")

    if price is not None and shares is not None and shares > 0:
        fill("market_cap", price * shares)
    market_cap = current("market_cap")

    if price is not None and eps is not None and eps > 0:
        fill("pe", price / eps)
    fill("ps", _ratio(market_cap, revenue))
    fill("pb", _ratio(market_cap, equity))
    if market_cap is not None and debt is not None and cash is not None:
        fill("ev_ebitda", _ratio(market_cap + debt - cash, ebitda))

    fill("operating_margin", _pct(operating_income, revenue))
    fill("profit_margin", _pct(net_income, revenue))
    fill("roe", _pct(net_income, equity))
    fill("roa", _pct(net_income, assets))
    fill("current_ratio", _ratio(current_assets, current_liabilities))
    if current_assets is not None and inventory is not None:
        fill("quick_ratio", _ratio(current_assets - inventory, current_liabilities))
    fill("debt_to_equity", _pct(debt, equity))

    fill("revenue_growth", _growth(get("fs_revenue"), get("fs_revenue_prev")))
    fill("earnings_growth", _growth(get("fs_diluted_eps"), get("fs_diluted_eps_prev")))

    dividends = get("fs_dividends_paid")
    if dividends is not None and market_cap:
        fill("dividend_yield", abs(dividends) / market_cap * 100)

    fill("beta", get("beta_1y"))
    return updates, list(updates)


def describe_derived(fields) -> str:
    """Human-readable list of derived field names."""
    if isinstance(fields, str):
        fields = [f for f in fields.split(",") if f]
    return ", ".join(DERIVED_FIELD_LABELS.get(f, f) for f in fields or [])
