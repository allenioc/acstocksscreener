"""Screener filter configuration and pass/fail logic with debug categories."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

# Full default slider bounds — filter is inactive when UI matches these exactly.
DEFAULT_BOUNDS = {
    "price": (0.0, 5000.0),
    "pe": (-100.0, 500.0),
    "pb": (-50.0, 50.0),
    "ps": (0.0, 200.0),
    "ev_ebitda": (-50.0, 200.0),
    "peg": (-10.0, 50.0),
    "roe": (-100.0, 200.0),
    "roa": (-100.0, 100.0),
    "profit_margin": (-100.0, 100.0),
    "operating_margin": (-100.0, 100.0),
    "debt_to_equity": (0.0, 500.0),
    "debt_to_assets": (0.0, 100.0),
    "equity_ratio": (0.0, 100.0),
    "interest_coverage": (0.0, 100.0),
    "current_ratio": (0.0, 20.0),
    "quick_ratio": (0.0, 20.0),
    "beta": (0.0, 5.0),
    "rsi": (0.0, 100.0),
}

DEFAULT_EPS_GROWTH_MIN = -100.0
DEFAULT_REV_GROWTH_MIN = -100.0
DEFAULT_PERF_MIN = -100.0
DEFAULT_PERF_MAX = 300.0


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _range_active(lo: float, hi: float, default_lo: float, default_hi: float, tol: float = 0.05) -> bool:
    """True when the user narrowed the range away from the neutral defaults."""
    return (lo > default_lo + tol) or (hi < default_hi - tol)


def build_filter_config(
    *,
    price_min: float,
    price_max: float,
    min_volume: int,
    market_cap: str,
    sector_filter: list,
    pe_min: float,
    pe_max: float,
    pb_min: float,
    pb_max: float,
    ps_min: float,
    ps_max: float,
    ev_ebitda_min: float,
    ev_ebitda_max: float,
    peg_min: float,
    peg_max: float,
    min_dividend: float,
    roe_min: float,
    roe_max: float,
    roa_min: float,
    roa_max: float,
    profit_margin_min: float,
    profit_margin_max: float,
    operating_margin_min: float,
    operating_margin_max: float,
    debt_to_equity_min: float,
    debt_to_equity_max: float,
    debt_to_assets_min: float,
    debt_to_assets_max: float,
    equity_ratio_min: float,
    equity_ratio_max: float,
    interest_coverage_min: float,
    interest_coverage_max: float,
    current_ratio_min: float,
    current_ratio_max: float,
    quick_ratio_min: float,
    quick_ratio_max: float,
    beta_min: float,
    beta_max: float,
    eps_growth_min: float,
    revenue_growth_min: float,
    perf_period: str,
    perf_min: float,
    perf_max: float,
    rsi_min: float,
    rsi_max: float,
    above_sma20: bool,
    above_sma50: bool,
    above_sma200: bool,
    macd_bullish: bool,
    bb_position: str,
) -> dict:
    def rb(name: str, lo: float, hi: float) -> dict:
        dlo, dhi = DEFAULT_BOUNDS[name]
        return {"min": lo, "max": hi, "active": _range_active(lo, hi, dlo, dhi)}

    return {
        "price": rb("price", price_min, price_max),
        "volume": {"min": min_volume, "active": min_volume > 0},
        "market_cap": {"value": market_cap, "active": market_cap != "Any"},
        "sector": {"values": sector_filter, "active": sector_filter and "Any" not in sector_filter},
        "pe": rb("pe", pe_min, pe_max),
        "pb": rb("pb", pb_min, pb_max),
        "ps": rb("ps", ps_min, ps_max),
        "ev_ebitda": rb("ev_ebitda", ev_ebitda_min, ev_ebitda_max),
        "peg": rb("peg", peg_min, peg_max),
        "dividend": {"min": min_dividend, "active": min_dividend > 0},
        "roe": rb("roe", roe_min, roe_max),
        "roa": rb("roa", roa_min, roa_max),
        "profit_margin": rb("profit_margin", profit_margin_min, profit_margin_max),
        "operating_margin": rb("operating_margin", operating_margin_min, operating_margin_max),
        "debt_to_equity": rb("debt_to_equity", debt_to_equity_min, debt_to_equity_max),
        "debt_to_assets": rb("debt_to_assets", debt_to_assets_min, debt_to_assets_max),
        "equity_ratio": rb("equity_ratio", equity_ratio_min, equity_ratio_max),
        "interest_coverage": rb("interest_coverage", interest_coverage_min, interest_coverage_max),
        "current_ratio": rb("current_ratio", current_ratio_min, current_ratio_max),
        "quick_ratio": rb("quick_ratio", quick_ratio_min, quick_ratio_max),
        "beta": rb("beta", beta_min, beta_max),
        "eps_growth": {"min": eps_growth_min, "active": eps_growth_min > DEFAULT_EPS_GROWTH_MIN},
        "revenue_growth": {"min": revenue_growth_min, "active": revenue_growth_min > DEFAULT_REV_GROWTH_MIN},
        "performance": {
            "period": perf_period,
            "min": perf_min,
            "max": perf_max,
            "active": perf_period != "Any" and _range_active(
                perf_min, perf_max, DEFAULT_PERF_MIN, DEFAULT_PERF_MAX
            ),
        },
        "rsi": rb("rsi", rsi_min, rsi_max),
        "above_sma20": above_sma20,
        "above_sma50": above_sma50,
        "above_sma200": above_sma200,
        "macd_bullish": macd_bullish,
        "bb_position": {"value": bb_position, "active": bb_position != "Any"},
    }


_STOCK_KEYS = {
    "price": "Price",
    "pe": "PE",
    "pb": "PB",
    "ps": "PS",
    "ev_ebitda": "EV_EBITDA",
    "peg": "PEG",
    "roe": "ROE",
    "roa": "ROA",
    "profit_margin": "ProfitMargin",
    "operating_margin": "OperatingMargin",
    "debt_to_equity": "DebtToEquity",
    "debt_to_assets": "DebtToAssets",
    "equity_ratio": "EquityRatio",
    "interest_coverage": "InterestCoverage",
    "current_ratio": "CurrentRatio",
    "quick_ratio": "QuickRatio",
    "beta": "Beta",
    "rsi": "RSI",
}


def passes_filters(stock: Optional[dict], cfg: dict) -> Tuple[bool, Optional[str]]:
    """Return (passed, rejection_category)."""
    if stock is None:
        return False, "fetch_failed"

    def range_filter(field: str, category: str) -> Tuple[bool, Optional[str]]:
        f = cfg[field]
        if not f["active"]:
            return True, None
        val = _num(stock.get(_STOCK_KEYS[field]))
        if val is None:
            return False, category
        if val < f["min"] or val > f["max"]:
            return False, category
        return True, None

    # Price
    ok, reason = range_filter("price", "price")
    if not ok:
        return False, reason

    # Volume
    if cfg["volume"]["active"]:
        vol = _num(stock.get("Volume"))
        if vol is None or vol < cfg["volume"]["min"]:
            return False, "volume"

    # Market cap
    if cfg["market_cap"]["active"]:
        mc = _num(stock.get("MarketCap"))
        if mc is None:
            return False, "market_cap"
        cap = cfg["market_cap"]["value"]
        if cap == "Mega (>$200B)" and mc < 200e9:
            return False, "market_cap"
        if cap == "Large ($10B-$200B)" and not (10e9 <= mc < 200e9):
            return False, "market_cap"
        if cap == "Mid ($2B-$10B)" and not (2e9 <= mc < 10e9):
            return False, "market_cap"
        if cap == "Small ($300M-$2B)" and not (300e6 <= mc < 2e9):
            return False, "market_cap"
        if cap == "Micro (<$300M)" and mc >= 300e6:
            return False, "market_cap"

    for field in ("pe", "pb", "ps", "ev_ebitda", "peg", "roe", "roa", "profit_margin",
                  "operating_margin", "debt_to_equity", "debt_to_assets", "equity_ratio",
                  "interest_coverage", "current_ratio", "quick_ratio", "beta", "rsi"):
        ok, reason = range_filter(field, field)
        if not ok:
            return False, reason

    # Dividend
    if cfg["dividend"]["active"]:
        div = _num(stock.get("DividendYield"))
        if div is None or div < cfg["dividend"]["min"]:
            return False, "dividend"

    # EPS / revenue growth
    if cfg["eps_growth"]["active"]:
        g = _num(stock.get("EPSGrowth"))
        if g is None or g < cfg["eps_growth"]["min"]:
            return False, "eps_growth"

    if cfg["revenue_growth"]["active"]:
        g = _num(stock.get("RevenueGrowth"))
        if g is None or g < cfg["revenue_growth"]["min"]:
            return False, "revenue_growth"

    # Performance
    if cfg["performance"]["active"]:
        period_map = {
            "1 Week": "Week",
            "1 Month": "Month",
            "3 Months": "3Months",
            "6 Months": "6Months",
            "1 Year": "Year",
        }
        key = period_map.get(cfg["performance"]["period"])
        val = _num(stock.get(key)) if key else None
        if val is None:
            return False, "performance"
        if val < cfg["performance"]["min"] or val > cfg["performance"]["max"]:
            return False, "performance"

    # Sector
    if cfg["sector"]["active"]:
        sector = stock.get("Sector")
        if sector is None or sector not in cfg["sector"]["values"]:
            return False, "sector"

    # Technical
    if cfg["above_sma20"]:
        sma = _num(stock.get("SMA20"))
        price = _num(stock.get("Price"))
        if sma is None or price is None or price < sma:
            return False, "technical"

    if cfg["above_sma50"]:
        sma = _num(stock.get("SMA50"))
        price = _num(stock.get("Price"))
        if sma is None or price is None or price < sma:
            return False, "technical"

    if cfg["above_sma200"]:
        sma = _num(stock.get("SMA200"))
        price = _num(stock.get("Price"))
        if sma is None or price is None or price < sma:
            return False, "technical"

    if cfg["macd_bullish"]:
        if not stock.get("MACD_Bullish", False):
            return False, "technical"

    if cfg["bb_position"]["active"]:
        bb_pos = stock.get("BB_Position")
        if bb_pos is None or bb_pos != cfg["bb_position"]["value"]:
            return False, "technical"

    return True, None


def empty_debug_counts() -> dict:
    return {
        "scanned": 0,
        "fetched": 0,
        "matched": 0,
        "fetch_failed": 0,
        "price": 0,
        "volume": 0,
        "market_cap": 0,
        "sector": 0,
        "pe": 0,
        "pb": 0,
        "ps": 0,
        "ev_ebitda": 0,
        "peg": 0,
        "dividend": 0,
        "roe": 0,
        "roa": 0,
        "profit_margin": 0,
        "operating_margin": 0,
        "debt_to_equity": 0,
        "debt_to_assets": 0,
        "equity_ratio": 0,
        "interest_coverage": 0,
        "current_ratio": 0,
        "quick_ratio": 0,
        "beta": 0,
        "eps_growth": 0,
        "revenue_growth": 0,
        "performance": 0,
        "rsi": 0,
        "technical": 0,
    }
