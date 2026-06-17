"""Functional top-navigation page sections."""

import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from theme import THEME, apply_dark_plotly_layout, delta_class, format_pct, format_market_cap_display, results_bar
from universe import get_stock_universe


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_universe_snapshot(markets_key: tuple):
    """Fetch real quote/fundamental snapshot for heatmap and groups."""
    markets = list(markets_key)
    symbols = get_stock_universe(markets)
    rows = []

    for sym in symbols:
        row = {
            "Ticker": sym,
            "Sector": None,
            "Industry": None,
            "MarketCap": None,
            "ChangePct": None,
            "Volume": None,
            "PE": None,
            "Price": None,
        }
        try:
            ticker = yf.Ticker(sym)
            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                pass

            row["Sector"] = info.get("sector") or "Unknown"
            row["Industry"] = info.get("industry") or "Unknown"
            row["MarketCap"] = info.get("marketCap")
            row["PE"] = info.get("trailingPE") or info.get("forwardPE")
            row["Volume"] = info.get("averageVolume") or info.get("regularMarketVolume")

            hist = ticker.history(period="5d")
            if hist is not None and not hist.empty:
                row["Price"] = float(hist["Close"].iloc[-1])
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    if prev != 0:
                        row["ChangePct"] = ((row["Price"] - prev) / prev) * 100
        except Exception:
            pass
        rows.append(row)
        time.sleep(0.015)

    return pd.DataFrame(rows)


def _ticker_picker(label: str = "Ticker", default: str = "AAPL", key_prefix: str = "page"):
    universe = sorted(get_stock_universe(["US", "Canada"]))
    c1, c2 = st.columns([1, 2])
    with c1:
        manual = st.text_input(f"{label} symbol", value=default, key=f"{key_prefix}_ticker_input").upper().strip()
    with c2:
        picked = st.selectbox("Or select from universe", options=[""] + universe, key=f"{key_prefix}_ticker_select")
    return manual or picked


def _fetch_ticker_info(symbol: str):
    if not symbol:
        return None
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        hist = ticker.history(period="5d")
        price = None
        change_pct = None
        if hist is not None and not hist.empty:
            price = float(hist["Close"].iloc[-1])
            if len(hist) >= 2:
                prev = float(hist["Close"].iloc[-2])
                if prev != 0:
                    change_pct = ((price - prev) / prev) * 100
        return {
            "info": info,
            "hist": hist,
            "price": price,
            "change_pct": change_pct,
        }
    except Exception:
        return None


def render_charts_page(fetch_historical_data, create_advanced_candlestick_chart, create_rsi_chart):
    st.markdown('<div class="section-header">Price Charts</div>', unsafe_allow_html=True)
    symbol = _ticker_picker(key_prefix="charts")
    period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key="charts_period")

    if st.button("Load Chart", key="charts_load", type="primary"):
        st.session_state.charts_symbol = symbol
        st.session_state.charts_period = period

    symbol = st.session_state.get("charts_symbol", symbol)
    period = st.session_state.get("charts_period", period)

    if not symbol:
        st.info("Enter or select a ticker, then click **Load Chart**.")
        return

    snapshot = _fetch_ticker_info(symbol)
    hist = fetch_historical_data(symbol, period=period)

    if snapshot is None and (hist is None or hist.empty):
        st.warning("Data unavailable for this ticker.")
        return

    info = snapshot["info"] if snapshot else {}
    price = snapshot["price"] if snapshot else (float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None)
    change_pct = snapshot["change_pct"] if snapshot else None

    mc = info.get("marketCap")
    pe = info.get("trailingPE") or info.get("forwardPE")
    hi52 = info.get("fiftyTwoWeekHigh")
    lo52 = info.get("fiftyTwoWeekLow")
    vol = info.get("averageVolume") or info.get("regularMarketVolume")
    sector = info.get("sector") or "Data unavailable"

    cols = st.columns(6)
    metrics = [
        ("Price", f"${price:.2f}" if price else "Data unavailable", delta_class(change_pct)),
        ("Change", format_pct(change_pct) if change_pct is not None else "Data unavailable", delta_class(change_pct)),
        ("Market Cap", format_market_cap_display(mc) if mc else "Data unavailable", ""),
        ("P/E", f"{pe:.2f}" if pe else "Data unavailable", ""),
        ("52W Range", f"${lo52:.2f} - ${hi52:.2f}" if hi52 and lo52 else "Data unavailable", ""),
        ("Sector", sector, ""),
    ]
    for col, (label, val, css) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="fv-kpi"><div class="fv-kpi-label">{label}</div>'
                f'<div class="fv-kpi-val {css}">{val}</div></div>',
                unsafe_allow_html=True,
            )

    if hist is not None and not hist.empty:
        line_fig = go.Figure()
        line_fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close", line=dict(color=THEME["accent_blue"], width=2)))
        apply_dark_plotly_layout(line_fig, title=f"{symbol} — Price", height=420)
        st.plotly_chart(line_fig, use_container_width=True)

        indicators = st.multiselect("Indicators", ["SMA", "BB", "MACD"], default=["SMA"], key="charts_indicators")
        if indicators:
            candle_fig = create_advanced_candlestick_chart(hist, symbol, indicators)
            st.plotly_chart(candle_fig, use_container_width=True)
        rsi_fig = create_rsi_chart(hist, symbol)
        if rsi_fig:
            st.plotly_chart(rsi_fig, use_container_width=True)
    else:
        st.warning("Historical price data unavailable.")


def _load_snapshot_with_progress(markets):
    markets_key = tuple(markets)
    cache_key = f"snapshot_{markets_key}"

    if st.button("Load Market Data", type="primary", key=f"load_{cache_key}"):
        with st.spinner(f"Fetching market data for {len(get_stock_universe(markets))} symbols..."):
            progress = st.progress(0)
            df = fetch_universe_snapshot(markets_key)
            progress.progress(1.0)
            progress.empty()
            st.session_state[f"data_{cache_key}"] = df

    if f"data_{cache_key}" not in st.session_state:
        st.info("Click **Load Market Data** to fetch live quotes from Yahoo Finance.")
        return pd.DataFrame()

    return st.session_state[f"data_{cache_key}"]


def render_maps_page():
    st.markdown('<div class="section-header">Market Heatmap</div>', unsafe_allow_html=True)
    markets = st.multiselect("Markets", ["US", "Canada"], default=["US", "Canada"], key="maps_markets")
    group_by = st.radio("Group by", ["Sector", "Industry"], horizontal=True, key="maps_group")

    if not markets:
        st.info("Select at least one market.")
        return

    df = _load_snapshot_with_progress(markets)
    if df.empty:
        st.warning("Data unavailable.")
        return

    plot_df = df.copy()
    plot_df["MarketCap"] = pd.to_numeric(plot_df["MarketCap"], errors="coerce")
    plot_df = plot_df[plot_df["MarketCap"].notna() & (plot_df["MarketCap"] > 0)]
    if plot_df.empty:
        st.warning("Market cap data unavailable for heatmap.")
        return

    plot_df["ChangePct"] = pd.to_numeric(plot_df["ChangePct"], errors="coerce")
    plot_df["ChangePctDisplay"] = plot_df["ChangePct"].fillna(0)
    path_col = "Sector" if group_by == "Sector" else "Industry"
    plot_df[path_col] = plot_df[path_col].fillna("Unknown")

    plot_df["ChangeLabel"] = plot_df["ChangePct"].apply(
        lambda x: format_pct(x) if pd.notna(x) else "Data unavailable"
    )
    plot_df["PriceLabel"] = plot_df["Price"].apply(
        lambda x: f"${x:.2f}" if pd.notna(x) else "Data unavailable"
    )
    plot_df["CapLabel"] = plot_df["MarketCap"].apply(
        lambda x: format_market_cap_display(x) if pd.notna(x) else "Data unavailable"
    )
    fig = px.treemap(
        plot_df,
        path=[path_col, "Ticker"],
        values="MarketCap",
        color="ChangePctDisplay",
        color_continuous_scale=[
            [0.0, THEME["negative"]],
            [0.5, THEME["neutral"]],
            [1.0, THEME["positive"]],
        ],
        range_color=[-5, 5],
        custom_data=["ChangeLabel", "PriceLabel", "CapLabel"],
        title=f"Market Heatmap by {group_by} (size = market cap, color = daily % change)",
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b>",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Change: %{customdata[0]}<br>"
            "Price: %{customdata[1]}<br>"
            "Mkt Cap: %{customdata[2]}<extra></extra>"
        ),
    )
    apply_dark_plotly_layout(fig, height=640)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Green = positive daily change · Red = negative · Gray center = neutral/unavailable")


def render_groups_page():
    st.markdown('<div class="section-header">Sector & Industry Groups</div>', unsafe_allow_html=True)
    markets = st.multiselect("Markets", ["US", "Canada"], default=["US", "Canada"], key="groups_markets")
    group_level = st.radio("Aggregate by", ["Sector", "Industry"], horizontal=True, key="groups_level")

    if not markets:
        st.info("Select at least one market.")
        return

    df = _load_snapshot_with_progress(markets)
    if df.empty:
        st.warning("Data unavailable.")
        return

    group_col = "Sector" if group_level == "Sector" else "Industry"
    work = df.copy()
    work[group_col] = work[group_col].fillna("Unknown")
    work["ChangePct"] = pd.to_numeric(work["ChangePct"], errors="coerce")
    work["PE"] = pd.to_numeric(work["PE"], errors="coerce")
    work["MarketCap"] = pd.to_numeric(work["MarketCap"], errors="coerce")
    work["Volume"] = pd.to_numeric(work["Volume"], errors="coerce")

    grouped = work.groupby(group_col, dropna=False).agg(
        Stocks=("Ticker", "count"),
        AvgChangePct=("ChangePct", "mean"),
        AvgPE=("PE", "mean"),
        TotalMarketCap=("MarketCap", "sum"),
        AvgVolume=("Volume", "mean"),
    ).reset_index()

    grouped = grouped.rename(columns={group_col: group_level})
    grouped["AvgChangePct"] = grouped["AvgChangePct"].apply(lambda x: format_pct(x) if pd.notna(x) else "Data unavailable")
    grouped["AvgPE"] = grouped["AvgPE"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "Data unavailable")
    grouped["TotalMarketCap"] = grouped["TotalMarketCap"].apply(
        lambda x: format_market_cap_display(x) if pd.notna(x) and x > 0 else "Data unavailable"
    )
    grouped["AvgVolume"] = grouped["AvgVolume"].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "Data unavailable")
    grouped = grouped.sort_values("Stocks", ascending=False)

    st.markdown(results_bar(count=len(grouped)), unsafe_allow_html=True)
    st.dataframe(grouped, use_container_width=True, hide_index=True, height=480)


def render_insider_page():
    st.markdown('<div class="section-header">Insider Transactions</div>', unsafe_allow_html=True)
    symbol = _ticker_picker(default="AAPL", key_prefix="insider")

    if st.button("Load Insider Data", key="insider_load", type="primary"):
        st.session_state.insider_symbol = symbol

    symbol = st.session_state.get("insider_symbol", symbol)
    if not symbol:
        st.info("Enter or select a ticker, then click **Load Insider Data**.")
        return

    try:
        ticker = yf.Ticker(symbol)
        frames = []
        for attr in ("insider_transactions", "insider_purchases", "insider_roster_holders"):
            data = getattr(ticker, attr, None)
            if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                display = data.copy()
                display.insert(0, "Source", attr.replace("_", " ").title())
                frames.append(display)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            st.markdown(results_bar(message=f"Insider activity for {symbol}"), unsafe_allow_html=True)
            st.dataframe(combined, use_container_width=True, hide_index=True, height=520)
        else:
            st.warning("Insider data unavailable for this ticker through the current free data source.")
    except Exception:
        st.warning("Insider data unavailable for this ticker through the current free data source.")


def render_news_page():
    st.markdown('<div class="section-header">Ticker News</div>', unsafe_allow_html=True)
    symbol = _ticker_picker(default="AAPL", key_prefix="news")

    if st.button("Load News", key="news_load", type="primary"):
        st.session_state.news_symbol = symbol

    symbol = st.session_state.get("news_symbol", symbol)
    if not symbol:
        st.info("Enter or select a ticker, then click **Load News**.")
        return

    try:
        ticker = yf.Ticker(symbol)
        articles = ticker.news or []
        if not articles:
            st.warning("News unavailable for this ticker.")
            return

        st.markdown(results_bar(count=len(articles)), unsafe_allow_html=True)
        for article in articles:
            title = article.get("title") or "Untitled"
            publisher = article.get("publisher") or article.get("source") or "Unknown source"
            link = article.get("link") or article.get("url") or ""
            ts = article.get("providerPublishTime") or article.get("publishedAt")
            date_str = ""
            if ts:
                try:
                    date_str = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_str = str(ts)

            st.markdown(
                f'<div class="fv-news-item">'
                f'<div class="fv-news-title">{title}</div>'
                f'<div class="fv-news-meta">{publisher} · {date_str}</div>'
                f'{"<a href=\"" + link + "\" target=\"_blank\" class=\"fv-news-link\">Read article</a>" if link else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.warning("News unavailable for this ticker.")
