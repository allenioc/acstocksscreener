"""Rule-based full picture stock analysis for screener detail panel."""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from filter_logic import build_filter_config, any_filter_active
from theme import THEME, apply_dark_plotly_layout
from financial_statements import render_financial_statements

KEY_METRICS = [
    ("price", "Price"),
    ("market_cap", "MarketCap"),
    ("pe", "PE"),
    ("pb", "PB"),
    ("ps", "PS"),
    ("ev_ebitda", "EV_EBITDA"),
    ("roe", "ROE"),
    ("roa", "ROA"),
    ("profit_margin", "ProfitMargin"),
    ("operating_margin", "OperatingMargin"),
    ("revenue_growth", "RevenueGrowth"),
    ("earnings_growth", "EPSGrowth"),
    ("debt_to_equity", "DebtToEquity"),
    ("current_ratio", "CurrentRatio"),
    ("quick_ratio", "QuickRatio"),
    ("beta", "Beta"),
    ("volume", "volume"),
    ("average_volume", "average_volume"),
    ("fifty_two_week_high", "fifty_two_week_high"),
    ("fifty_two_week_low", "fifty_two_week_low"),
    ("sector", "Sector"),
    ("industry", "industry"),
]


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().replace("$", "").replace(",", "").replace("%", "")
        if v in {"", "-", "N/A", "nan", "None"}:
            return None
        try:
            value = float(v)
        except ValueError:
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _get(row: pd.Series, *keys: str) -> Any:
    for key in keys:
        if key in row.index and pd.notna(row[key]):
            return row[key]
    return None


def extract_metrics(row: pd.Series) -> dict:
    """Normalize a screener row into a metrics dict."""
    ticker = _get(row, "ticker", "Ticker")
    company = _get(row, "company_name", "Company")
    sector = _get(row, "sector", "Sector")
    industry = _get(row, "industry", "Industry")

    metrics = {
        "ticker": str(ticker) if ticker is not None else "Unknown",
        "company_name": str(company) if company is not None else str(ticker or "Unknown"),
        "sector": str(sector) if sector is not None and pd.notna(sector) else None,
        "industry": str(industry) if industry is not None and pd.notna(industry) else None,
        "month": _num(_get(row, "month", "Month")),
        "year": _num(_get(row, "year", "Year")),
    }
    for snake, pascal in KEY_METRICS:
        if snake in ("sector", "industry"):
            continue
        metrics[snake] = _num(_get(row, snake, pascal))
    return metrics


def data_confidence(metrics: dict) -> tuple[int, list[str]]:
    """Return confidence score (0-100) and missing important fields."""
    important = [
        ("price", "Price"),
        ("market_cap", "Market cap"),
        ("pe", "P/E"),
        ("roe", "ROE"),
        ("profit_margin", "Profit margin"),
        ("revenue_growth", "Revenue growth"),
        ("earnings_growth", "EPS growth"),
        ("debt_to_equity", "Debt/equity"),
        ("beta", "Beta"),
        ("sector", "Sector"),
    ]
    present = 0
    missing = []
    for key, label in important:
        val = metrics.get(key)
        if key in ("sector",) and val:
            present += 1
        elif key not in ("sector",) and val is not None:
            present += 1
        else:
            missing.append(label)
    score = int(round((present / len(important)) * 100))
    return score, missing


def detect_red_flags(metrics: dict) -> list[str]:
    flags = []
    pe = metrics.get("pe")
    pb = metrics.get("pb")
    ps = metrics.get("ps")
    de = metrics.get("debt_to_equity")
    roe = metrics.get("roe")
    roa = metrics.get("roa")
    pm = metrics.get("profit_margin")
    om = metrics.get("operating_margin")
    rev_g = metrics.get("revenue_growth")
    eps_g = metrics.get("earnings_growth")
    beta = metrics.get("beta")
    vol = metrics.get("volume")
    avg_vol = metrics.get("average_volume")

    if pe is None and metrics.get("earnings_growth") is None:
        flags.append("Negative or missing earnings (P/E unavailable)")
    elif pe is not None and pe < 0:
        flags.append("Negative P/E (unprofitable or negative earnings)")

    if de is not None and de > 200:
        flags.append("High debt/equity (above 200)")
    elif de is not None and de > 100:
        flags.append("Elevated debt/equity (above 100)")

    if pe is not None and pe > 45:
        flags.append("Expensive valuation (P/E above 45)")
    if pb is not None and pb > 8:
        flags.append("Expensive valuation (P/B above 8)")
    if ps is not None and ps > 15:
        flags.append("Expensive valuation (P/S above 15)")

    if pm is not None and pm < 5:
        flags.append("Weak profit margin (below 5%)")
    if om is not None and om < 5:
        flags.append("Weak operating margin (below 5%)")
    if roe is not None and roe < 5:
        flags.append("Weak ROE (below 5%)")
    if roa is not None and roa < 2:
        flags.append("Weak ROA (below 2%)")

    if rev_g is not None and rev_g < 0:
        flags.append("Declining revenue growth")
    elif rev_g is not None and rev_g < 3:
        flags.append("Weak revenue growth (below 3%)")

    if eps_g is not None and eps_g < 0:
        flags.append("Negative EPS growth")
    elif eps_g is not None and eps_g < 3:
        flags.append("Weak EPS growth (below 3%)")

    if beta is not None and beta > 1.8:
        flags.append("High beta (above 1.8)")
    if beta is not None and beta > 2.5:
        flags.append("Very high beta (above 2.5)")

    if vol is not None and avg_vol is not None and avg_vol > 0 and vol < avg_vol * 0.3:
        flags.append("Low recent liquidity vs average volume")

    if metrics.get("price") is None:
        flags.append("Missing price data")
    if metrics.get("market_cap") is None:
        flags.append("Missing market cap data")

    # Missing important data warnings as flags
    _, missing = data_confidence(metrics)
    if len(missing) >= 5:
        flags.append(f"Many key metrics missing ({len(missing)} fields)")

    return flags


def sector_context(metrics: dict, full_df: pd.DataFrame) -> dict:
    """Compare stock metrics to sector peers in the current result set."""
    sector = metrics.get("sector")
    if not sector or "Sector" not in full_df.columns:
        return {}

    peers = full_df[full_df["Sector"] == sector]
    if len(peers) < 2:
        return {"peer_count": len(peers), "sector": sector}

    compare_fields = [
        ("pe", "PE", "P/E"),
        ("pb", "PB", "P/B"),
        ("roe", "ROE", "ROE"),
        ("profit_margin", "ProfitMargin", "Profit margin"),
        ("revenue_growth", "RevenueGrowth", "Revenue growth"),
        ("beta", "Beta", "Beta"),
    ]
    ctx = {"peer_count": len(peers), "sector": sector}
    stock_val = metrics

    for snake, pascal, label in compare_fields:
        peer_col = pascal if pascal in peers.columns else snake
        if peer_col not in peers.columns:
            continue
        peer_avg = pd.to_numeric(peers[peer_col], errors="coerce").mean()
        val = stock_val.get(snake)
        if val is None or pd.isna(peer_avg):
            ctx[snake] = {"label": label, "status": "unavailable"}
            continue
        diff_pct = ((val - peer_avg) / abs(peer_avg) * 100) if peer_avg != 0 else 0
        if abs(diff_pct) < 5:
            status = "inline"
        elif val > peer_avg:
            status = "above"
        else:
            status = "below"
        ctx[snake] = {
            "label": label,
            "status": status,
            "stock": val,
            "sector_avg": peer_avg,
            "diff_pct": diff_pct,
        }
    return ctx


def full_picture_score(metrics: dict, flags: list[str], confidence: int) -> int:
    """Rule-based composite score 0-100 (not investment advice)."""
    score = 50.0

    roe = metrics.get("roe")
    roa = metrics.get("roa")
    pm = metrics.get("profit_margin")
    om = metrics.get("operating_margin")
    rev_g = metrics.get("revenue_growth")
    eps_g = metrics.get("earnings_growth")
    pe = metrics.get("pe")
    de = metrics.get("debt_to_equity")
    beta = metrics.get("beta")
    month = _num(metrics.get("month"))

    if roe is not None:
        score += min(roe / 5, 10)
    if roa is not None:
        score += min(roa / 3, 8)
    if pm is not None:
        score += min(pm / 4, 8)
    if om is not None:
        score += min(om / 4, 6)
    if rev_g is not None:
        score += min(rev_g / 5, 10)
    if eps_g is not None:
        score += min(eps_g / 5, 10)

    if pe is not None and 5 < pe < 30:
        score += 5
    if de is not None and de < 80:
        score += 5
    if beta is not None and 0.8 <= beta <= 1.3:
        score += 3
    if month is not None and month > 0:
        score += 3
    elif month is not None and month < -10:
        score -= 5

    score -= min(len(flags) * 4, 30)
    score *= confidence / 100.0
    score += (confidence / 100.0) * 10

    return int(max(0, min(100, round(score))))


def why_passed_screen(metrics: dict, filter_kwargs: Optional[dict]) -> list[str]:
    """Explain which active screen criteria the stock met."""
    if not filter_kwargs:
        return ["Included in current screener results with default or applied filters."]

    cfg = build_filter_config(**filter_kwargs)
    if not any_filter_active(cfg):
        return [
            "No restrictive filters were active; stock appeared in the broad screener universe with available quote data.",
        ]

    reasons = []

    def in_range(field: str, val_key: str) -> bool:
        f = cfg.get(field, {})
        if not f.get("active"):
            return False
        val = metrics.get(val_key)
        if val is None:
            return True  # missing passes inactive-style filters
        return f["min"] <= val <= f["max"]

    if cfg["market_cap"]["active"] and metrics.get("market_cap") is not None:
        reasons.append(f"Market cap fits selected bucket ({cfg['market_cap']['value']}).")
    if cfg["sector"]["active"] and metrics.get("sector"):
        reasons.append(f"Sector matches filter: {metrics['sector']}.")
    if in_range("pe", "pe"):
        reasons.append("P/E is within the active P/E range.")
    if in_range("pb", "pb"):
        reasons.append("P/B is within the active P/B range.")
    if in_range("roe", "roe"):
        reasons.append("ROE is within the active ROE range.")
    if in_range("debt_to_equity", "debt_to_equity"):
        reasons.append("Debt/equity is within the active range.")
    if cfg["eps_growth"]["active"] and metrics.get("earnings_growth") is not None:
        if metrics["earnings_growth"] >= cfg["eps_growth"]["min"]:
            reasons.append("EPS growth meets the minimum threshold.")
    if cfg["revenue_growth"]["active"] and metrics.get("revenue_growth") is not None:
        if metrics["revenue_growth"] >= cfg["revenue_growth"]["min"]:
            reasons.append("Revenue growth meets the minimum threshold.")

    if not reasons:
        reasons.append("Passed active filters where data was available; some metrics may be missing.")

    return reasons


def build_narrative(metrics: dict, flags: list[str], sector_ctx: dict, passed: list[str]) -> str:
    """Rule-based explanation paragraph."""
    strengths = []
    concerns = []

    if metrics.get("roe") is not None and metrics["roe"] >= 15:
        strengths.append("profitability (ROE) is strong")
    if metrics.get("revenue_growth") is not None and metrics["revenue_growth"] >= 10:
        strengths.append("revenue growth is solid")
    if metrics.get("earnings_growth") is not None and metrics["earnings_growth"] >= 10:
        strengths.append("earnings growth is solid")
    if metrics.get("profit_margin") is not None and metrics["profit_margin"] >= 15:
        strengths.append("margins are healthy")

    if flags:
        concerns = flags[:3]

    month = metrics.get("month")
    if month is not None and month < -5:
        concerns.append("recent one-month performance is weak")

    parts = []
    if passed:
        parts.append("This stock passed the screen because " + "; ".join(passed[:2]).lower() + ".")
    elif strengths:
        parts.append("This stock shows " + ", ".join(strengths) + ".")
    else:
        parts.append("This stock met the current screener criteria with limited standout metrics.")

    if concerns:
        parts.append("However, " + "; ".join(concerns[:3]) + ".")
    elif strengths:
        parts.append("No major rule-based red flags were triggered from available data.")

    pe_ctx = sector_ctx.get("pe")
    if isinstance(pe_ctx, dict) and pe_ctx.get("status") == "above":
        parts.append(f"Valuation (P/E) is above the sector average in this result set ({pe_ctx['sector_avg']:.1f}).")
    elif isinstance(pe_ctx, dict) and pe_ctx.get("status") == "below":
        parts.append(f"Valuation (P/E) is below the sector average in this result set ({pe_ctx['sector_avg']:.1f}).")

    return " ".join(parts)


def _fmt_money(val: Optional[float]) -> str:
    if val is None:
        return "Data unavailable"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    if val >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.2f}"


def _fmt_ratio(val: Optional[float]) -> str:
    return f"{val:.2f}" if val is not None else "Data unavailable"


def _fmt_pct(val: Optional[float]) -> str:
    return f"{val:+.2f}%" if val is not None else "Data unavailable"


def _fmt_vol(val: Optional[float]) -> str:
    return f"{val:,.0f}" if val is not None else "Data unavailable"


def render_stock_detail_panel(
    selected_ticker: str,
    df: pd.DataFrame,
    filter_kwargs: Optional[dict],
    fetch_history,
) -> None:
    """Render full picture analysis below the results table."""
    match = df[df["Ticker"] == selected_ticker]
    if match.empty and "ticker" in df.columns:
        match = df[df["ticker"] == selected_ticker]
    if match.empty:
        st.warning("Selected ticker not found in current results.")
        return

    row = match.iloc[0]
    metrics = extract_metrics(row)
    confidence, missing_fields = data_confidence(metrics)
    flags = detect_red_flags(metrics)
    sector_ctx = sector_context(metrics, df)
    passed = why_passed_screen(metrics, filter_kwargs)
    score = full_picture_score(metrics, flags, confidence)
    narrative = build_narrative(metrics, flags, sector_ctx, passed)

    st.markdown("---")
    st.markdown('<div class="section-header">Full Picture Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f"**{metrics['ticker']}** · {metrics['company_name']}  \n"
        f"{metrics.get('sector') or 'Data unavailable'}"
        f"{(' · ' + metrics['industry']) if metrics.get('industry') else ''}",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Full Picture Score", f"{score}/100")
    with c2:
        conf_label = "High" if confidence >= 75 else "Medium" if confidence >= 50 else "Low"
        st.metric("Data Confidence", f"{confidence}% ({conf_label})")
    with c3:
        st.metric("Red Flags", str(len(flags)))
    with c4:
        price = metrics.get("price")
        st.metric("Price", _fmt_money(price) if price is not None else "Data unavailable")

    st.markdown(f'<p style="color:{THEME["text_secondary"]};font-size:13px;">{narrative}</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.caption("Valuation")
        st.write(f"**Market cap:** {_fmt_money(metrics.get('market_cap'))}")
        st.write(f"**P/E:** {_fmt_ratio(metrics.get('pe'))}")
        st.write(f"**P/B:** {_fmt_ratio(metrics.get('pb'))}")
        st.write(f"**P/S:** {_fmt_ratio(metrics.get('ps'))}")
        st.write(f"**EV/EBITDA:** {_fmt_ratio(metrics.get('ev_ebitda'))}")
    with m2:
        st.caption("Profitability & Growth")
        st.write(f"**ROE:** {_fmt_pct(metrics.get('roe'))}")
        st.write(f"**ROA:** {_fmt_pct(metrics.get('roa'))}")
        st.write(f"**Profit margin:** {_fmt_pct(metrics.get('profit_margin'))}")
        st.write(f"**Oper. margin:** {_fmt_pct(metrics.get('operating_margin'))}")
        st.write(f"**Revenue growth:** {_fmt_pct(metrics.get('revenue_growth'))}")
        st.write(f"**EPS growth:** {_fmt_pct(metrics.get('earnings_growth'))}")
    with m3:
        st.caption("Balance Sheet & Risk")
        st.write(f"**Debt/equity:** {_fmt_ratio(metrics.get('debt_to_equity'))}")
        st.write(f"**Current ratio:** {_fmt_ratio(metrics.get('current_ratio'))}")
        st.write(f"**Quick ratio:** {_fmt_ratio(metrics.get('quick_ratio'))}")
        st.write(f"**Beta:** {_fmt_ratio(metrics.get('beta'))}")
        st.write(f"**Volume:** {_fmt_vol(metrics.get('volume'))}")
    with m4:
        st.caption("Range & Sector")
        st.write(f"**52-week high:** {_fmt_money(metrics.get('fifty_two_week_high'))}")
        st.write(f"**52-week low:** {_fmt_money(metrics.get('fifty_two_week_low'))}")
        if sector_ctx.get("peer_count", 0) >= 2:
            st.write(f"**Sector peers in results:** {sector_ctx['peer_count']}")
            for key in ("pe", "roe", "revenue_growth", "beta"):
                item = sector_ctx.get(key)
                if isinstance(item, dict) and item.get("status") not in (None, "unavailable"):
                    status = item["status"].replace("inline", "near")
                    st.write(f"**{item['label']} vs sector:** {status} avg ({item['sector_avg']:.2f})")
        else:
            st.write("**Sector context:** Not enough peers in current results.")

    col_chart, col_info = st.columns([2, 1])
    with col_chart:
        hist = fetch_history(metrics["ticker"], period="1y")
        if hist is not None and not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="Close",
                line=dict(color=THEME["accent_blue"], width=2),
            ))
            apply_dark_plotly_layout(fig, title=f"{metrics['ticker']} — 1 Year Price", height=320)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price chart data unavailable.")

    with col_info:
        st.markdown("**Why it passed the screen**")
        for r in passed:
            st.write(f"• {r}")
        st.markdown("**Red flags**")
        if flags:
            for f in flags:
                st.markdown(f'<span style="color:{THEME["negative"]};">• {f}</span>', unsafe_allow_html=True)
        else:
            st.write("No rule-based red flags from available data.")
        if missing_fields:
            st.markdown("**Missing data warnings**")
            for m in missing_fields:
                st.write(f"• {m}: Data unavailable")

    render_financial_statements(metrics["ticker"])

    st.caption("Rule-based analysis for educational purposes only. Not financial advice. No buy/sell recommendation.")
