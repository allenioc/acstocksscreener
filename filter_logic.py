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

# Ranges treated as "no filter" — includes current defaults and legacy Streamlit session values.
NEUTRAL_RANGES = {
    "price": [(0.0, 5000.0)],
    "pe": [(-100.0, 500.0), (0.0, 100.0)],
    "pb": [(-50.0, 50.0), (0.0, 20.0)],
    "ps": [(0.0, 200.0), (0.0, 50.0)],
    "ev_ebitda": [(-50.0, 200.0), (0.0, 50.0)],
    "peg": [(-10.0, 50.0), (0.0, 10.0)],
    "roe": [(-100.0, 200.0), (-50.0, 100.0)],
    "roa": [(-100.0, 100.0), (-50.0, 50.0)],
    "profit_margin": [(-100.0, 100.0), (-50.0, 50.0)],
    "operating_margin": [(-100.0, 100.0), (-50.0, 50.0)],
    "debt_to_equity": [(0.0, 500.0), (0.0, 10.0)],
    "debt_to_assets": [(0.0, 100.0)],
    "equity_ratio": [(0.0, 100.0)],
    "interest_coverage": [(0.0, 100.0), (0.0, 50.0)],
    "current_ratio": [(0.0, 20.0), (0.0, 10.0)],
    "quick_ratio": [(0.0, 20.0), (0.0, 10.0)],
    "beta": [(0.0, 5.0)],
    "rsi": [(0.0, 100.0)],
    "performance": [(-100.0, 300.0), (-50.0, 300.0)],
}


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


def _range_active(name: str, lo: float, hi: float, tol: float = 0.5) -> bool:
    """False when slider matches any known neutral/default range (filter not applied)."""
    ranges = NEUTRAL_RANGES.get(name, [DEFAULT_BOUNDS[name]])
    for dlo, dhi in ranges:
        if abs(lo - dlo) <= tol and abs(hi - dhi) <= tol:
            return False
    return True


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
        return {"min": lo, "max": hi, "active": _range_active(name, lo, hi)}

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
                "performance", perf_min, perf_max
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


def _df_col(df, *names):
    import pandas as pd

    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(float("nan"), index=df.index)


def apply_screener_filters(df, cfg: dict):
    """Vectorized DataFrame filtering. Returns (filtered_df, rejection_counts, excluded_samples)."""
    import pandas as pd

    if df is None or df.empty:
        return df, {}, []

    mask = pd.Series(True, index=df.index)
    reasons = pd.Series(None, dtype="object", index=df.index)
    rejections: dict[str, int] = {}

    def _reject(category: str, fail_mask):
        nonlocal mask
        newly_failed = mask & fail_mask
        count = int(newly_failed.sum())
        if count:
            rejections[category] = rejections.get(category, 0) + count
            reasons.loc[newly_failed & reasons.isna()] = category
        mask &= ~fail_mask

    def _apply_range(category: str, col, f: dict):
        if not f.get("active"):
            return
        col = pd.to_numeric(col, errors="coerce")
        fail = col.isna() | (col < f["min"]) | (col > f["max"])
        _reject(category, fail)

    _apply_range("price", _df_col(df, "price", "Price"), cfg["price"])

    if cfg["volume"]["active"]:
        vol = _df_col(df, "volume", "Volume", "average_volume", "AvgVolume")
        _reject("volume", vol.isna() | (vol < cfg["volume"]["min"]))

    if cfg["market_cap"]["active"]:
        mc = _df_col(df, "market_cap", "MarketCap")
        cap = cfg["market_cap"]["value"]
        fail = mc.isna()
        if cap == "Mega (>$200B)":
            fail |= mc < 200e9
        elif cap == "Large ($10B-$200B)":
            fail |= (mc < 10e9) | (mc >= 200e9)
        elif cap == "Mid ($2B-$10B)":
            fail |= (mc < 2e9) | (mc >= 10e9)
        elif cap == "Small ($300M-$2B)":
            fail |= (mc < 300e6) | (mc >= 2e9)
        elif cap == "Micro (<$300M)":
            fail |= mc >= 300e6
        _reject("market_cap", fail)

    for field in (
        "pe", "pb", "ps", "ev_ebitda", "peg", "roe", "roa", "profit_margin",
        "operating_margin", "debt_to_equity", "debt_to_assets", "equity_ratio",
        "interest_coverage", "current_ratio", "quick_ratio", "beta", "rsi",
    ):
        key_map = {
            "pe": ("pe", "PE"),
            "pb": ("pb", "PB"),
            "ps": ("ps", "PS"),
            "ev_ebitda": ("ev_ebitda", "EV_EBITDA"),
            "peg": ("peg", "PEG"),
            "roe": ("roe", "ROE"),
            "roa": ("roa", "ROA"),
            "profit_margin": ("profit_margin", "ProfitMargin"),
            "operating_margin": ("operating_margin", "OperatingMargin"),
            "debt_to_equity": ("debt_to_equity", "DebtToEquity"),
            "debt_to_assets": ("debt_to_assets", "DebtToAssets"),
            "equity_ratio": ("equity_ratio", "EquityRatio"),
            "interest_coverage": ("interest_coverage", "InterestCoverage"),
            "current_ratio": ("current_ratio", "CurrentRatio"),
            "quick_ratio": ("quick_ratio", "QuickRatio"),
            "beta": ("beta", "Beta"),
            "rsi": ("rsi", "RSI"),
        }
        _apply_range(field, _df_col(df, *key_map[field]), cfg[field])

    if cfg["dividend"]["active"]:
        div = _df_col(df, "dividend_yield", "DividendYield")
        _reject("dividend", div.isna() | (div < cfg["dividend"]["min"]))

    if cfg["eps_growth"]["active"]:
        g = _df_col(df, "earnings_growth", "EPSGrowth")
        _reject("eps_growth", g.isna() | (g < cfg["eps_growth"]["min"]))

    if cfg["revenue_growth"]["active"]:
        g = _df_col(df, "revenue_growth", "RevenueGrowth")
        _reject("revenue_growth", g.isna() | (g < cfg["revenue_growth"]["min"]))

    if cfg["performance"]["active"]:
        period_map = {
            "1 Week": ("week", "Week"),
            "1 Month": ("month", "Month"),
            "3 Months": ("quarter", "3Months"),
            "6 Months": ("6Months",),
            "1 Year": ("year", "Year"),
        }
        keys = period_map.get(cfg["performance"]["period"], ())
        val = _df_col(df, *keys) if keys else pd.Series(float("nan"), index=df.index)
        fail = val.isna() | (val < cfg["performance"]["min"]) | (val > cfg["performance"]["max"])
        _reject("performance", fail)

    if cfg["sector"]["active"]:
        sector_col = df["sector"] if "sector" in df.columns else df.get("Sector")
        if sector_col is None:
            _reject("sector", pd.Series(True, index=df.index))
        else:
            _reject("sector", ~sector_col.isin(cfg["sector"]["values"]))

    # Technical filters only when columns exist
    if cfg.get("above_sma20"):
        price = _df_col(df, "price", "Price")
        sma = _df_col(df, "SMA20")
        if sma.notna().any():
            _reject("technical", sma.isna() | price.isna() | (price < sma))

    if cfg.get("above_sma50"):
        price = _df_col(df, "price", "Price")
        sma = _df_col(df, "SMA50")
        if sma.notna().any():
            _reject("technical", sma.isna() | price.isna() | (price < sma))

    if cfg.get("above_sma200"):
        price = _df_col(df, "price", "Price")
        sma = _df_col(df, "SMA200")
        if sma.notna().any():
            _reject("technical", sma.isna() | price.isna() | (price < sma))

    if cfg.get("macd_bullish") and "MACD_Bullish" in df.columns:
        _reject("technical", ~df["MACD_Bullish"].fillna(False))

    if cfg["bb_position"]["active"] and "BB_Position" in df.columns:
        _reject("technical", df["BB_Position"] != cfg["bb_position"]["value"])

    filtered = df[mask].copy()
    excluded = df[~mask].copy()
    samples = []
    for idx in excluded.head(10).index:
        ticker = df.at[idx, "ticker"] if "ticker" in df.columns else df.at[idx, "Ticker"]
        samples.append({"ticker": ticker, "reason": reasons.at[idx] or "unknown"})
    return filtered, rejections, samples


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
