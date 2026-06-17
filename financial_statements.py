"""Fetch and display yfinance financial statements for stock detail view."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from theme import THEME

INCOME_LINES = [
    ("Total Revenue", ["Total Revenue", "TotalRevenue", "Revenue"]),
    ("Gross Profit", ["Gross Profit", "GrossProfit"]),
    ("Operating Income", ["Operating Income", "OperatingIncome"]),
    ("Net Income", ["Net Income", "NetIncome", "Net Income Common Stockholders"]),
    ("EBITDA", ["EBITDA", "Normalized EBITDA"]),
    ("Diluted EPS", ["Diluted EPS", "Diluted EPS", "Basic EPS"]),
]

BALANCE_LINES = [
    ("Cash and Cash Equivalents", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"]),
    ("Total Assets", ["Total Assets", "TotalAssets"]),
    ("Total Debt", ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"]),
    ("Total Liabilities", ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"]),
    ("Stockholders Equity", ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]),
    ("Working Capital", ["__working_capital__"]),
]

CASHFLOW_LINES = [
    ("Operating Cash Flow", ["Operating Cash Flow", "Total Cash From Operating Activities"]),
    ("Capital Expenditures", ["Capital Expenditure", "Capital Expenditures"]),
    ("Free Cash Flow", ["Free Cash Flow", "__free_cash_flow__"]),
    ("Investing Cash Flow", ["Investing Cash Flow", "Total Cashflows From Investing Activities"]),
    ("Financing Cash Flow", ["Financing Cash Flow", "Total Cash From Financing Activities"]),
]

EPS_LABELS = {"Diluted EPS"}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financial_statements(ticker: str, frequency: str) -> dict[str, Optional[pd.DataFrame]]:
    """Load income, balance, and cash flow statements from yfinance."""
    result: dict[str, Optional[pd.DataFrame]] = {
        "income": None,
        "balance": None,
        "cashflow": None,
    }
    try:
        t = yf.Ticker(ticker)
        if frequency == "quarterly":
            result["income"] = _safe_df(t.quarterly_income_stmt)
            if result["income"] is None:
                result["income"] = _safe_df(t.quarterly_financials)
            result["balance"] = _safe_df(t.quarterly_balance_sheet)
            result["cashflow"] = _safe_df(t.quarterly_cashflow)
        else:
            result["income"] = _safe_df(t.income_stmt)
            if result["income"] is None:
                result["income"] = _safe_df(t.financials)
            result["balance"] = _safe_df(t.balance_sheet)
            result["cashflow"] = _safe_df(t.cashflow)
    except Exception:
        pass
    return result


def _safe_df(df) -> Optional[pd.DataFrame]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return df.copy()


def _find_row(df: pd.DataFrame, keys: list[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    index_map = {str(i).lower(): i for i in df.index}
    for key in keys:
        if key == "__working_capital__":
            ca = _find_row(df, ["Current Assets", "Total Current Assets"])
            cl = _find_row(df, ["Current Liabilities", "Total Current Liabilities"])
            if ca is not None and cl is not None:
                return ca - cl
            return None
        kl = key.lower()
        if kl in index_map:
            return df.loc[index_map[kl]]
        for idx_lower, idx_orig in index_map.items():
            if kl in idx_lower or idx_lower in kl:
                return df.loc[idx_orig]
    return None


def _period_columns(df: pd.DataFrame, max_periods: int = 5) -> list:
    cols = list(df.columns)
    try:
        cols = sorted(cols, reverse=True)
    except TypeError:
        pass
    return cols[:max_periods]


def _period_label(col, frequency: str = "annual") -> str:
    try:
        ts = pd.Timestamp(col)
        if frequency == "annual":
            return f"FY {ts.year}"
        if ts.month in (1, 2, 3):
            q = "Q1"
        elif ts.month in (4, 5, 6):
            q = "Q2"
        elif ts.month in (7, 8, 9):
            q = "Q3"
        else:
            q = "Q4"
        if ts.month == 12 and ts.day >= 28:
            return f"FY {ts.year}"
        return f"{q} {ts.year}"
    except Exception:
        return str(col)[:10]


def _fmt_amount(val: Any, is_eps: bool = False) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        num = float(val)
    except (TypeError, ValueError):
        return "—"
    if is_eps:
        return f"${num:.2f}"
    sign = "-" if num < 0 else ""
    abs_num = abs(num)
    if abs_num >= 1e9:
        return f"{sign}${abs_num / 1e9:.2f}B"
    if abs_num >= 1e6:
        return f"{sign}${abs_num / 1e6:.2f}M"
    if abs_num >= 1e3:
        return f"{sign}${abs_num / 1e3:.2f}K"
    return f"{sign}${abs_num:,.0f}"


def _compute_fcf(cashflow_df: pd.DataFrame, col) -> Optional[float]:
    ocf = _find_row(cashflow_df, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex = _find_row(cashflow_df, ["Capital Expenditure", "Capital Expenditures"])
    if ocf is None:
        return None
    ocf_val = _num_at(ocf, col)
    if ocf_val is None:
        return None
    if capex is not None:
        capex_val = _num_at(capex, col)
        if capex_val is not None:
            return ocf_val + capex_val
    fcf_row = _find_row(cashflow_df, ["Free Cash Flow"])
    if fcf_row is not None:
        return _num_at(fcf_row, col)
    return None


def _num_at(series: pd.Series, col) -> Optional[float]:
    if col not in series.index:
        return None
    val = series[col]
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def build_statement_table(
    df: Optional[pd.DataFrame],
    line_defs: list[tuple[str, list[str]]],
    cashflow_df: Optional[pd.DataFrame] = None,
    max_periods: int = 5,
    frequency: str = "annual",
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    periods = _period_columns(df, max_periods)
    if not periods:
        return None

    rows = []
    for label, keys in line_defs:
        is_eps = label in EPS_LABELS
        row_vals = [label]
        for col in periods:
            if keys == ["__free_cash_flow__"] and cashflow_df is not None:
                val = _compute_fcf(cashflow_df, col)
            else:
                series = _find_row(df if keys != ["__free_cash_flow__"] else cashflow_df, keys)
                val = _num_at(series, col) if series is not None else None
            row_vals.append(_fmt_amount(val, is_eps=is_eps))
        rows.append(row_vals)

    headers = ["Line Item"] + [_period_label(c, frequency) for c in periods]
    return pd.DataFrame(rows, columns=headers)


def _series_for_summary(
    income: Optional[pd.DataFrame],
    balance: Optional[pd.DataFrame],
    cashflow: Optional[pd.DataFrame],
    keys: list[str],
    computed: str = "",
) -> list[Optional[float]]:
    if computed == "fcf" and cashflow is not None:
        periods = _period_columns(cashflow, 5)
        return [_compute_fcf(cashflow, c) for c in periods]
    df = income if keys[0] in ("Total Revenue", "Net Income") else balance
    if keys[0] in ("Operating Cash Flow",):
        df = cashflow
    if df is None:
        return []
    series = _find_row(df, keys)
    if series is None:
        return []
    periods = _period_columns(df, 5)
    return [_num_at(series, c) for c in periods]


def _trend_text(label: str, values: list[Optional[float]]) -> str:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return f"{label}: Insufficient data for trend."
    newest, oldest = clean[0], clean[-1]
    if oldest == 0:
        return f"{label}: {_fmt_amount(newest)} (latest); prior period near zero — trend unclear."
    pct = (newest - oldest) / abs(oldest) * 100
    direction = "up" if pct > 5 else "down" if pct < -5 else "relatively flat"
    return f"{label}: Trending {direction} ({pct:+.1f}% from oldest to newest period shown)."


def statement_summary(
    income: Optional[pd.DataFrame],
    balance: Optional[pd.DataFrame],
    cashflow: Optional[pd.DataFrame],
) -> tuple[list[str], list[str]]:
    """Return trend bullets and missing-data warnings."""
    trends = []
    warnings = []

    rev = _series_for_summary(income, balance, cashflow, ["Total Revenue", "Revenue"])
    ni = _series_for_summary(income, balance, cashflow, ["Net Income", "NetIncome"])
    debt = _series_for_summary(income, balance, cashflow, ["Total Debt", "Long Term Debt"])
    fcf = _series_for_summary(income, balance, cashflow, [], computed="fcf")

    if rev:
        trends.append(_trend_text("Revenue", rev))
    else:
        warnings.append("Revenue: Financial statement data unavailable.")

    if ni:
        trends.append(_trend_text("Net income", ni))
    else:
        warnings.append("Net income: Financial statement data unavailable.")

    if debt:
        trends.append(_trend_text("Total debt", debt))
    else:
        warnings.append("Total debt: Financial statement data unavailable.")

    if fcf and any(v is not None for v in fcf):
        trends.append(_trend_text("Free cash flow", fcf))
    else:
        warnings.append("Free cash flow: Financial statement data unavailable.")

    if income is None:
        warnings.append("Income statement: Financial statement data unavailable.")
    if balance is None:
        warnings.append("Balance sheet: Financial statement data unavailable.")
    if cashflow is None:
        warnings.append("Cash flow statement: Financial statement data unavailable.")

    return trends, warnings


def _render_table(title: str, table: Optional[pd.DataFrame]) -> None:
    st.markdown(f"**{title}**")
    if table is None or table.empty:
        st.info("Financial statement data unavailable.")
        return
    st.dataframe(table, use_container_width=True, hide_index=True, height=min(56 + len(table) * 35, 320))


def render_financial_statements(ticker: str) -> None:
    """Render financial statements section inside stock detail."""
    st.markdown("---")
    st.markdown('<div class="section-header">Financial Statements</div>', unsafe_allow_html=True)
    st.caption("Source: Yahoo Finance via yfinance. Educational analysis only — not financial advice.")

    freq_label = st.radio(
        "Statement period",
        ["Annual", "Quarterly"],
        horizontal=True,
        index=0,
        key=f"fs_freq_{ticker}",
    )
    frequency = "annual" if freq_label == "Annual" else "quarterly"

    with st.spinner("Loading financial statements..."):
        data = fetch_financial_statements(ticker, frequency)

    income = data.get("income")
    balance = data.get("balance")
    cashflow = data.get("cashflow")

    trends, warnings = statement_summary(income, balance, cashflow)

    st.markdown("**Financial statement summary**")
    for t in trends:
        st.write(f"• {t}")
    if warnings:
        st.markdown(f'<span style="color:{THEME["text_muted"]};">**Missing data warnings**</span>', unsafe_allow_html=True)
        for w in warnings:
            st.write(f"• {w}")

    income_table = build_statement_table(income, INCOME_LINES, frequency=frequency)
    balance_table = build_statement_table(balance, BALANCE_LINES, frequency=frequency)
    cashflow_table = build_statement_table(
        cashflow,
        CASHFLOW_LINES,
        cashflow_df=cashflow,
        frequency=frequency,
    )

    tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow Statement"])
    with tab1:
        _render_table("Income Statement", income_table)
    with tab2:
        _render_table("Balance Sheet", balance_table)
    with tab3:
        _render_table("Cash Flow Statement", cashflow_table)
