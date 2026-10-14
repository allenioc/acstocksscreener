"""Column layouts and formatting for the screener results tables."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

NAME_SUFFIXES = (" - Common Stock", " Common Stock", " - Class A", " - Class B", " - Class C")


def format_compact_money(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    a = abs(float(value))
    sign = "-" if value < 0 else ""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= threshold:
            return f"{sign}${a / threshold:.2f}{suffix}"
    return f"{sign}${a:,.0f}"


def clean_company_name(name) -> str:
    """Drop Nasdaq symbol-file suffixes ("Apple Inc. - Common Stock" -> "Apple Inc.")."""
    text = str(name or "")
    for suffix in NAME_SUFFIXES:
        if suffix.lower() in text.lower():
            text = text[: text.lower().index(suffix.lower())]
    return text.strip() or "—"


def _money(label: str, help_text: str = None):
    return st.column_config.NumberColumn(label, format="$%.2f", help=help_text)


def _pct(label: str, help_text: str = None, signed: bool = False):
    return st.column_config.NumberColumn(label, format="%+.2f%%" if signed else "%.2f%%", help=help_text)


def _ratio(label: str, help_text: str = None, suffix: str = ""):
    return st.column_config.NumberColumn(label, format=f"%.2f{suffix}", help=help_text)


FSCORE_COLUMN = st.column_config.ProgressColumn(
    "F-Score", min_value=0, max_value=9, format="%d/9",
    help="Piotroski F-score (0–9) from the two most recent annual reports.",
)
SLOAN_COLUMN = _pct(
    "Sloan %", "(Net income − operating cash flow) / total assets. Above +10% flags poor cash conversion.",
    signed=True,
)

# view -> list of (source column, display label, column config)
TABLE_LAYOUTS = {
    "Summary": [
        ("ticker", "Ticker", st.column_config.TextColumn("Ticker", width="small")),
        ("_name", "Name", st.column_config.TextColumn("Name", width="medium")),
        ("price", "Price", _money("Price")),
        ("_market_cap", "Market Cap", st.column_config.TextColumn("Market Cap")),
        ("sector", "Sector", st.column_config.TextColumn("Sector")),
        ("pe", "P/E", _ratio("P/E", "Trailing price / earnings", "x")),
        ("piotroski_score", "F-Score", FSCORE_COLUMN),
        ("sloan_ratio", "Sloan %", SLOAN_COLUMN),
        ("year", "1Y Return", _pct("1Y Return", signed=True)),
        ("sharpe", "Sharpe", _ratio("Sharpe", "1Y Sharpe ratio, 4% risk-free rate")),
    ],
    "Valuation": [
        ("ticker", "Ticker", st.column_config.TextColumn("Ticker", width="small")),
        ("price", "Price", _money("Price")),
        ("_market_cap", "Market Cap", st.column_config.TextColumn("Market Cap")),
        ("pe", "P/E", _ratio("P/E", suffix="x")),
        ("forward_pe", "Fwd P/E", _ratio("Fwd P/E", suffix="x")),
        ("pb", "P/B", _ratio("P/B", suffix="x")),
        ("ps", "P/S", _ratio("P/S", suffix="x")),
        ("peg", "PEG", _ratio("PEG")),
        ("ev_ebitda", "EV/EBITDA", _ratio("EV/EBITDA", suffix="x")),
        ("dividend_yield", "Div Yield", _pct("Div Yield")),
    ],
    "Financial Health": [
        ("ticker", "Ticker", st.column_config.TextColumn("Ticker", width="small")),
        ("roe", "ROE", _pct("ROE")),
        ("roa", "ROA", _pct("ROA")),
        ("profit_margin", "Profit Margin", _pct("Profit Margin")),
        ("operating_margin", "Oper. Margin", _pct("Oper. Margin")),
        ("debt_to_equity", "Debt/Equity", _ratio("Debt/Equity", "Total debt / equity × 100 (Yahoo convention)")),
        ("current_ratio", "Current Ratio", _ratio("Current Ratio")),
        ("quick_ratio", "Quick Ratio", _ratio("Quick Ratio")),
        ("revenue_growth", "Rev Growth", _pct("Rev Growth", signed=True)),
        ("earnings_growth", "EPS Growth", _pct("EPS Growth", signed=True)),
        ("piotroski_score", "F-Score", FSCORE_COLUMN),
        ("sloan_ratio", "Sloan %", SLOAN_COLUMN),
    ],
    "Momentum": [
        ("ticker", "Ticker", st.column_config.TextColumn("Ticker", width="small")),
        ("price", "Price", _money("Price")),
        ("week", "1W", _pct("1W", signed=True)),
        ("month", "1M", _pct("1M", signed=True)),
        ("quarter", "3M", _pct("3M", signed=True)),
        ("six_month", "6M", _pct("6M", signed=True)),
        ("year", "1Y", _pct("1Y", signed=True)),
        ("rsi", "RSI (14)", _ratio("RSI (14)")),
        ("volatility", "Volatility", _pct("Volatility", "Annualized 1Y daily volatility")),
        ("sharpe", "Sharpe", _ratio("Sharpe")),
        ("beta", "Beta", _ratio("Beta")),
        ("_above_sma200", "> SMA200", st.column_config.CheckboxColumn("> SMA200", help="Price above 200-day SMA")),
    ],
}


def _fmt(value, template: str) -> str:
    return template.format(value) if value is not None and pd.notna(value) else "—"


def selection_banner_html(row: pd.Series) -> str:
    """One-line snapshot of the selected ticker shown above the results tables."""
    items = [
        ("Price", _fmt(row.get("price"), "${:,.2f}")),
        ("Mkt Cap", format_compact_money(row.get("market_cap"))),
        ("P/E", _fmt(row.get("pe"), "{:.1f}x")),
        ("F-Score", _fmt(row.get("piotroski_score"), "{:.0f}/9")),
        ("Sloan", _fmt(row.get("sloan_ratio"), "{:+.2f}%")),
        ("1Y", _fmt(row.get("year"), "{:+.2f}%")),
    ]
    stats = "".join(
        f'<span class="dd-stat"><span class="dd-stat-label">{label}</span>'
        f'<span class="dd-stat-value">{html.escape(value).replace("$", "&#36;")}</span></span>'
        for label, value in items
    )
    name = html.escape(clean_company_name(row.get("company_name")))
    return (
        f'<div class="dd-banner"><span class="dd-title">Selected: <b>{html.escape(str(row.get("ticker")))}</b>'
        f' · {name}</span>{stats}</div>'
    )


def build_table(df: pd.DataFrame, view: str) -> tuple[pd.DataFrame, dict]:
    """Display frame (same row order as `df`) and its column_config for a table view."""
    source = df.copy()
    source["_name"] = source.get("company_name", source["ticker"]).map(clean_company_name)
    source["_market_cap"] = source["market_cap"].map(format_compact_money) if "market_cap" in source else "—"
    if "sma200" in source and "price" in source:
        source["_above_sma200"] = (source["price"] > source["sma200"]).where(source["sma200"].notna())
    if "sector" in source:
        source["sector"] = source["sector"].fillna("—")

    out, config = pd.DataFrame(index=source.index), {}
    for col, label, column_config in TABLE_LAYOUTS[view]:
        out[label] = source[col] if col in source else None
        config[label] = column_config
    return out.reset_index(drop=True), config
