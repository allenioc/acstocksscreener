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
<div style="
background: linear-gradient(135deg,#667eea,#764ba2);
padding:2rem;
border-radius:12px;
margin-bottom:1.5rem;
color:white;
text-align:center;">
<h1>📈 AC Stocks Screener</h1>
<p>Professional Stock Screening for US & Canadian Markets</p>
</div>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.header("🎯 Filters")

    markets = st.multiselect(
        "Markets",
        ["US", "Canada"],
        default=["US", "Canada"]
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

    run_button = st.button("🚀 RUN SCREENER", use_container_width=True)

# =============================
# STOCK UNIVERSE (SMALL + RELIABLE)
# =============================
@st.cache_data(ttl=3600)
def get_stock_universe(markets):
    stocks = []

    if "US" in markets:
        stocks += [
            "AAPL","MSFT","NVDA","AMZN","GOOGL",
            "META","TSLA","JPM","V","MA",
            "JNJ","UNH","XOM","HD","COST"
        ]

    if "Canada" in markets:
        stocks += [
            "RY.TO","TD.TO","BNS.TO","BMO.TO",
            "ENB.TO","CNQ.TO","SU.TO",
            "SHOP.TO","BCE.TO","CNR.TO"
        ]

    return stocks

# =============================
# FETCH STOCK DATA (BULLETPROOF)
# =============================
@st.cache_data(ttl=1800)
def fetch_stock(symbol):
    try:
        t = yf.Ticker(symbol)

        hist = t.history(period="1y")
        if hist is None or hist.empty:
            return None

        price = float(hist["Close"].iloc[-1])

        avg_volume = hist["Volume"].mean()

        return {
            "Ticker": symbol,
            "Price": price,
            "Avg Volume": avg_volume,
            "1Y Change %": ((price - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
        }

    except Exception:
        return None

# =============================
# RUN SCREENER
# =============================
if run_button:
    universe = get_stock_universe(markets)

    st.info(f"🔍 Screening {len(universe)} stocks...")
    progress = st.progress(0)

    results = []

    for i, sym in enumerate(universe):
        progress.progress((i + 1) / len(universe))
        data = fetch_stock(sym)

        if data is None:
            continue

        # PRICE FILTER
        if not (price_min <= data["Price"] <= price_max):
            continue

        # VOLUME FILTER
        if min_volume > 0 and data["Avg Volume"] < min_volume:
            continue

        results.append(data)
        time.sleep(0.05)

    progress.empty()

    if results:
        df = pd.DataFrame(results)

        st.success(f"✅ Found {len(df)} stocks")

        st.dataframe(
            df.style.format({
                "Price": "${:.2f}",
                "Avg Volume": "{:,.0f}",
                "1Y Change %": "{:.1f}%"
            }),
            use_container_width=True
        )

        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"ac_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
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
    "<center><strong>AC Stocks Screener</strong> | Powered by Yahoo Finance<br>"
    "<small>For educational purposes only</small></center>",
    unsafe_allow_html=True
)
