import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AC Stocks Screener",
    page_icon="📈",
    layout="wide",
)

# =========================
# HEADER
# =========================
st.markdown("""
<div style="background: linear-gradient(135deg,#667eea,#764ba2);
padding:2rem;border-radius:12px;margin-bottom:1.5rem;color:white;text-align:center;">
<h1>📈 AC Stocks Screener</h1>
<p>Finviz-style screening for US & Canadian markets</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🎯 Filters")

    preset = st.selectbox(
        "Preset",
        ["Custom", "Value", "Growth", "Dividend"]
    )

    markets = st.multiselect(
        "Markets",
        ["US", "Canada"],
        default=["US", "Canada"]
    )

    market_cap = st.selectbox(
        "Market Cap",
        ["Any", "Mega", "Large", "Mid", "Small"]
    )

    price_min, price_max = st.slider(
        "Price ($)",
        0.0, 5000.0, (5.0, 5000.0)
    )

    min_volume = st.selectbox(
        "Min Avg Volume",
        [0, 100_000, 500_000, 1_000_000],
        index=1
    )

    pe_max = st.number_input("Max P/E", value=100.0)

    perf_period = st.selectbox(
        "Performance",
        ["Any", "Week", "Month", "Year"]
    )

    perf_min = -100.0
    if perf_period != "Any":
        perf_min = st.slider("Min Performance (%)", -100.0, 300.0, -10.0)

    sector_filter = st.multiselect(
        "Sectors",
        ["Any", "Technology", "Healthcare", "Financial",
         "Consumer", "Industrial", "Energy",
         "Materials", "Utilities", "Real Estate"],
        default=["Any"]
    )

    sort_by = st.selectbox(
        "Sort by",
        ["MarketCap", "Price", "PE", "Week", "Month", "Year"]
    )

    run = st.button("🚀 RUN SCREENER", use_container_width=True)

# =========================
# PRESETS
# =========================
if preset == "Value":
    pe_max = 20
    perf_min = -50
    perf_period = "Any"
elif preset == "Growth":
    pe_max = 80
    perf_period = "Year"
    perf_min = 20
elif preset == "Dividend":
    pe_max = 25
    perf_period = "Any"

# =========================
# STOCK UNIVERSE (EXPANDED)
# =========================
@st.cache_data(ttl=3600)
def get_universe(markets):
    us = [
        # Mega / Large
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX",
        "JPM","BAC","WFC","V","MA","BRK-B","UNH","JNJ","LLY","PFE",
        "WMT","HD","COST","PG","KO","PEP","DIS","MCD",
        "XOM","CVX","COP","SLB","CAT","BA","GE","MMM",
        # Mid / Growth
        "AMD","INTC","QCOM","ORCL","CRM","ADBE","SHOP","SQ",
        "PLTR","SNOW","UBER","ABNB","RBLX"
    ]

    ca = [
        "RY.TO","TD.TO","BNS.TO","BMO.TO",
        "ENB.TO","CNQ.TO","SU.TO",
        "SHOP.TO","OTEX.TO",
        "BCE.TO","T.TO",
        "CNR.TO","CP.TO",
        "ATD.TO","DOL.TO"
    ]

    stocks = []
    if "US" in markets:
        stocks += us
    if "Canada" in markets:
        stocks += ca

    return sorted(set(stocks))

# =========================
# FETCH DATA (SAFE)
# =========================
@st.cache_data(ttl=1800)
def fetch_stock(symbol):
    t = yf.Ticker(symbol)
    hist = t.history(period="1y")

    if hist is None or hist.empty:
        return None

    price = float(hist["Close"].iloc[-1])

    def perf(days):
        if len(hist) > days:
            return (price - hist["Close"].iloc[-days]) / hist["Close"].iloc[-days] * 100
        return None

    try:
        info = t.fast_info
    except:
        info = {}

    return {
        "Ticker": symbol,
        "Price": price,
        "MarketCap": info.get("marketCap"),
        "Volume": info.get("averageVolume"),
        "PE": info.get("trailingPE"),
        "Sector": info.get("sector"),
        "Week": perf(5),
        "Month": perf(21),
        "Year": perf(252),
    }

# =========================
# FILTER LOGIC
# =========================
def passes(s):
    if s is None:
        return False

    if not (price_min <= s["Price"] <= price_max):
        return False

    if s["Volume"] and s["Volume"] < min_volume:
        return False

    mc = s["MarketCap"]
    if mc:
        if market_cap == "Mega" and mc < 200e9: return False
        if market_cap == "Large" and not (10e9 <= mc < 200e9): return False
        if market_cap == "Mid" and not (2e9 <= mc < 10e9): return False
        if market_cap == "Small" and mc >= 2e9: return False

    if s["PE"] and s["PE"] > pe_max:
        return False

    if "Any" not in sector_filter:
        if s["Sector"] not in sector_filter:
            return False

    if perf_period != "Any":
        v = s.get(perf_period)
        if v is not None and v < perf_min:
            return False

    return True

# =========================
# RUN
# =========================
if run:
    universe = get_universe(markets)

    st.info(f"🔍 Screening {len(universe)} stocks…")
    bar = st.progress(0)

    rows = []
    for i, sym in enumerate(universe):
        bar.progress((i + 1) / len(universe))
        d = fetch_stock(sym)
        if passes(d):
            rows.append(d)
        time.sleep(0.03)

    bar.empty()

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(sort_by, ascending=False, na_position="last")

        st.success(f"✅ Found {len(df)} stocks")

        st.dataframe(
            df.style.format({
                "Price": "${:.2f}",
                "MarketCap": "${:,.0f}",
                "Week": "{:.1f}%",
                "Month": "{:.1f}%",
                "Year": "{:.1f}%",
            }),
            use_container_width=True,
            height=600
        )

        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False),
            f"ac_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
    else:
        st.warning("⚠️ No stocks match your criteria.")

else:
    st.info("👈 Choose filters or a preset and click **RUN SCREENER**")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center><small>AC Stocks Screener | Yahoo Finance data | Educational use only</small></center>",
    unsafe_allow_html=True
)
