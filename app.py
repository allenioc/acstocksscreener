import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

st.set_page_config(
    page_title="AC Stocks Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AC Stocks Screener")
st.caption("Simple, reliable stock screener for US & Canadian markets")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("Filters")

    markets = st.multiselect(
        "Markets",
        ["US", "Canada"],
        default=["US", "Canada"]
    )

    price_min, price_max = st.slider(
        "Price range ($)",
        0.0, 5000.0, (0.0, 5000.0)
    )

    run = st.button("🚀 Run Screener")

# -----------------------------
# STOCK LIST
# -----------------------------
def get_universe(markets):
    stocks = []
    if "US" in markets:
        stocks += [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "JPM", "V", "MA", "WMT"
        ]
    if "Canada" in markets:
        stocks += [
            "RY.TO", "TD.TO", "BNS.TO", "ENB.TO", "SHOP.TO"
        ]
    return stocks

# -----------------------------
# FETCH DATA (BULLETPROOF)
# -----------------------------
@st.cache_data(ttl=1800)
def fetch_stock(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y")

        if hist.empty:
            return None

        price = float(hist["Close"].iloc[-1])

        return {
            "Ticker": symbol,
            "Price": price,
        }
    except:
        return None

# -----------------------------
# RUN SCREENER
# -----------------------------
if run:
    universe = get_universe(markets)

    st.info(f"Screening {len(universe)} stocks...")
    results = []

    for sym in universe:
        data = fetch_stock(sym)
        if data and price_min <= data["Price"] <= price_max:
            results.append(data)
        time.sleep(0.05)

    if results:
        df = pd.DataFrame(results)
        st.success(f"Found {len(df)} stocks")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    else:
        st.warning("No stocks matched your criteria.")
