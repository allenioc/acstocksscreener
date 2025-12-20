import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AC Stocks Screener",
    page_icon="📈",
    layout="wide",
)

# =============================
# HEADER
# =============================
st.markdown("""
<div style="background: linear-gradient(135deg,#667eea,#764ba2);
padding:2rem;border-radius:12px;margin-bottom:1.5rem;color:white;text-align:center;">
<h1>📈 AC Stocks Screener</h1>
<p>Finviz-Style Stock Screener (US & Canada)</p>
</div>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR FILTERS
# =============================
with st.sidebar:
    st.header("🎯 Filters")

    markets = st.multiselect(
        "Markets",
        ["US", "Canada"],
        default=["US", "Canada"]
    )

    market_cap = st.selectbox(
        "Market Cap",
        ["Any", "Mega", "Large", "Mid", "Small", "Micro"]
    )

    price_min, price_max = st.slider(
        "Price Range ($)",
        0.0, 5000.0, (0.0, 5000.0)
    )

    min_volume = st.selectbox(
        "Min Avg Volume",
        [0, 100_000, 500_000, 1_000_000],
        index=0
    )

    pe_max = st.number_input("Max P/E", value=100.0)

    min_dividend = st.slider("Min Dividend Yield (%)", 0.0, 10.0, 0.0)

    perf_period = st.selectbox(
        "Performance Period",
        ["Any", "Week", "Month", "Year"]
    )

    perf_min = -100.0
    if perf_period != "Any":
        perf_min = st.slider("Min Performance (%)", -100.0, 300.0, -20.0)

    sector_filter = st.multiselect(
        "Sectors",
        ["Any", "Technology", "Healthcare", "Financial",
         "Consumer", "Industrial", "Energy", "Materials",
         "Utilities", "Real Estate"],
        default=["Any"]
    )

    run_button = st.button("🚀 RUN SCREENER", use_container_width=True)

# =============================
# STOCK UNIVERSE
# =============================
@st.cache_data(ttl=3600)
def get_stock_universe(markets):
    stocks = []

    if "US" in markets:
        stocks += [
            'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX',
            'JPM','BAC','WFC','V','MA','JNJ','UNH','LLY','PFE',
            'WMT','HD','MCD','PG','KO','PEP','COST',
            'XOM','CVX','COP','CAT','GE','BA',
            'INTC','AMD','QCOM','AVGO','TXN',
            'ADBE','CRM','ORCL','IBM'
        ]

    if "Canada" in markets:
        stocks += [
            'RY.TO','TD.TO','BNS.TO','BMO.TO',
            'ENB.TO','CNQ.TO','SU.TO',
            'SHOP.TO','OTEX.TO',
            'BCE.TO','T.TO',
            'CNR.TO','CP.TO'
        ]

    return stocks

# =============================
# FETCH STOCK DATA (SAFE)
# =============================
@st.cache_data(ttl=1800)
def fetch_stock(symbol):
    t = yf.Ticker(symbol)
    hist = t.history(period="1y")

    if hist is None or hist.empty:
        return None

    price = float(hist["Close"].iloc[-1])

    def pct_change(days):
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
        "Volume": info.get("averageVolume"),
        "MarketCap": info.get("marketCap"),
        "PE": info.get("trailingPE"),
        "DividendYield": (info.get("dividendYield") or 0) * 100,
        "Week": pct_change(5),
        "Month": pct_change(21),
        "Year": pct_change(252),
        "Sector": info.get("sector"),
    }

# =============================
# FILTER LOGIC
# =============================
def passes_filters(s):
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
        if market_cap == "Small" and not (300e6 <= mc < 2e9): return False
        if market_cap == "Micro" and mc >= 300e6: return False

    if s["PE"] and s["PE"] > pe_max:
        return False

    if s["DividendYield"] < min_dividend:
        return False

    if perf_period != "Any":
        val = s.get(perf_period)
        if val is not None and val < perf_min:
            return False

    return True

# =============================
# RUN SCREENER
# =============================
if run_button:
    universe = get_stock_universe(markets)
    st.info(f"🔍 Screening {len(universe)} stocks...")
    bar = st.progress(0)

    results = []
    for i, sym in enumerate(universe):
        bar.progress((i + 1) / len(universe))
        data = fetch_stock(sym)
        if passes_filters(data):
            results.append(data)
        time.sleep(0.05)

    bar.empty()

    if results:
        df = pd.DataFrame(results)
        st.success(f"✅ Found {len(df)} stocks")

        st.dataframe(
            df,
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
    st.info("👈 Set filters and click **RUN SCREENER**")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "<center>AC Stocks Screener | Powered by Yahoo Finance<br>"
    "<small>For educational purposes only</small></center>",
    unsafe_allow_html=True
)
