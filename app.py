import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AC Stocks Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .positive { color: #00ff88; font-weight: bold; }
    .negative { color: #ff4444; font-weight: bold; }
    .stDataFrame { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("""
<div style="background: linear-gradient(135deg,#667eea,#764ba2);
padding:2rem;border-radius:12px;margin-bottom:1.5rem;color:white;text-align:center;">
<h1>📈 Finviz-Style Stock Screener</h1>
<p>Professional Stock Screening with Advanced Filters & Analytics</p>
</div>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR FILTERS (ENHANCED)
# =============================
with st.sidebar:
    st.header("🎯 Screening Filters")
    
    # Market Selection
    markets = st.multiselect(
        "Markets",
        ["US", "Canada"],
        default=["US"]
    )
    
    # Market Cap Filter
    market_cap = st.selectbox(
        "Market Cap",
        ["Any", "Mega (>$200B)", "Large ($10B-$200B)", "Mid ($2B-$10B)", "Small ($300M-$2B)", "Micro (<$300M)"]
    )
    
    st.markdown("---")
    
    # Price Filters
    st.subheader("💰 Price")
    price_min, price_max = st.slider(
        "Price Range ($)",
        0.0, 5000.0, (0.0, 5000.0),
        step=0.1
    )
    
    # Volume Filter
    st.subheader("📊 Volume")
    min_volume = st.selectbox(
        "Min Avg Volume",
        [0, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
        index=0
    )
    
    st.markdown("---")
    
    # Fundamental Filters
    st.subheader("📈 Fundamentals")
    pe_min, pe_max = st.slider(
        "P/E Ratio",
        0.0, 100.0, (0.0, 100.0),
        step=0.1
    )
    
    pb_min, pb_max = st.slider(
        "P/B Ratio",
        0.0, 20.0, (0.0, 20.0),
        step=0.1
    )
    
    ps_min, ps_max = st.slider(
        "P/S Ratio",
        0.0, 50.0, (0.0, 50.0),
        step=0.1
    )
    
    min_dividend = st.slider("Min Dividend Yield (%)", 0.0, 10.0, 0.0, step=0.1)
    
    beta_min, beta_max = st.slider(
        "Beta",
        0.0, 5.0, (0.0, 5.0),
        step=0.1
    )
    
    eps_growth_min = st.number_input("Min EPS Growth (YoY %)", value=-100.0, step=1.0)
    
    st.markdown("---")
    
    # Performance Filters
    st.subheader("📉 Performance")
    perf_period = st.selectbox(
        "Performance Period",
        ["Any", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
    )
    
    perf_min, perf_max = -100.0, 300.0
    if perf_period != "Any":
        perf_min = st.slider("Min Performance (%)", -100.0, 300.0, -50.0, step=1.0)
        perf_max = st.slider("Max Performance (%)", -100.0, 300.0, 300.0, step=1.0)
    
    st.markdown("---")
    
    # Technical Filters
    st.subheader("🔧 Technical")
    rsi_min, rsi_max = st.slider(
        "RSI (14)",
        0.0, 100.0, (0.0, 100.0),
        step=1.0
    )
    
    above_sma20 = st.checkbox("Above SMA 20")
    above_sma50 = st.checkbox("Above SMA 50")
    above_sma200 = st.checkbox("Above SMA 200")
    
    st.markdown("---")
    
    # Sector Filter
    st.subheader("🏢 Sectors")
    sector_filter = st.multiselect(
        "Sectors",
        ["Any", "Technology", "Healthcare", "Financial Services",
         "Consumer Cyclical", "Consumer Defensive", "Industrials", 
         "Energy", "Basic Materials", "Communication Services",
         "Utilities", "Real Estate"],
        default=["Any"]
    )
    
    st.markdown("---")
    
    # Run Button
    run_button = st.button("🚀 RUN SCREENER", use_container_width=True, type="primary")

# =============================
# STOCK UNIVERSE (EXPANDED)
# =============================
@st.cache_data(ttl=86400)
def get_stock_universe(markets):
    stocks = []
    
    if "US" in markets:
        # Tech
        stocks += ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'AMAT', 'LRCX']
        # Finance
        stocks += ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PYPL']
        # Healthcare
        stocks += ['JNJ', 'UNH', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'CVS']
        # Consumer
        stocks += ['WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'PG', 'KO', 'PEP', 'CL', 'UL']
        # Industrial
        stocks += ['CAT', 'GE', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'DE']
        # Energy
        stocks += ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC']
        # Other
        stocks += ['ADBE', 'CRM', 'ORCL', 'IBM', 'CSCO', 'INTU', 'NOW']
    
    if "Canada" in markets:
        stocks += [
            'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO',
            'ENB.TO', 'CNQ.TO', 'SU.TO', 'IMO.TO',
            'SHOP.TO', 'OTEX.TO', 'WCN.TO',
            'BCE.TO', 'T.TO', 'RCI-B.TO',
            'CNR.TO', 'CP.TO',
            'ATD.TO', 'L.TO'
        ]
    
    return list(set(stocks))  # Remove duplicates

# =============================
# TECHNICAL INDICATORS
# =============================
def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return None
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return None
    return prices.rolling(window=period).mean().iloc[-1]

# =============================
# FETCH STOCK DATA (ENHANCED)
# =============================
@st.cache_data(ttl=1800)
def fetch_stock(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="2y")
        
        if hist is None or hist.empty:
            return None
        
        current_price = float(hist["Close"].iloc[-1])
        prices = hist["Close"]
        
        # Performance calculations
        def pct_change(days):
            if len(hist) > days:
                old_price = hist["Close"].iloc[-days-1]
                return ((current_price - old_price) / old_price) * 100
            return None
        
        # Get info
        try:
            info = t.fast_info
        except:
            info = t.info if hasattr(t, 'info') else {}
        
        # Calculate technical indicators
        rsi = calculate_rsi(prices, 14)
        sma20 = calculate_sma(prices, 20)
        sma50 = calculate_sma(prices, 50)
        sma200 = calculate_sma(prices, 200)
        
        # Calculate earnings growth (if available)
        try:
            earnings_growth = info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth')
            if earnings_growth:
                eps_growth = float(earnings_growth) * 100
            else:
                eps_growth = None
        except:
            eps_growth = None
        
        return {
            "Ticker": symbol,
            "Price": current_price,
            "Volume": info.get("averageVolume") or info.get("volume24Hr"),
            "MarketCap": info.get("marketCap"),
            "PE": info.get("trailingPE") or info.get("forwardPE"),
            "PB": info.get("priceToBook"),
            "PS": info.get("priceToSalesTrailing12Months"),
            "DividendYield": (info.get("dividendYield") or 0) * 100,
            "Beta": info.get("beta"),
            "EPSGrowth": eps_growth,
            "Week": pct_change(5),
            "Month": pct_change(21),
            "3Months": pct_change(63),
            "6Months": pct_change(126),
            "Year": pct_change(252),
            "RSI": rsi,
            "SMA20": sma20,
            "SMA50": sma50,
            "SMA200": sma200,
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "52WHigh": info.get("fiftyTwoWeekHigh"),
            "52WLow": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        return None

# =============================
# FILTER LOGIC (ENHANCED)
# =============================
def passes_filters(s):
    if s is None:
        return False
    
    # Price filter
    if not (price_min <= s["Price"] <= price_max):
        return False
    
    # Volume filter
    if s["Volume"] and s["Volume"] < min_volume:
        return False
    
    # Market cap filter
    mc = s["MarketCap"]
    if mc:
        if market_cap == "Mega (>$200B)" and mc < 200e9: return False
        if market_cap == "Large ($10B-$200B)" and not (10e9 <= mc < 200e9): return False
        if market_cap == "Mid ($2B-$10B)" and not (2e9 <= mc < 10e9): return False
        if market_cap == "Small ($300M-$2B)" and not (300e6 <= mc < 2e9): return False
        if market_cap == "Micro (<$300M)" and mc >= 300e6: return False
    
    # P/E filter
    if s["PE"]:
        if s["PE"] < pe_min or s["PE"] > pe_max:
            return False
    
    # P/B filter
    if s["PB"]:
        if s["PB"] < pb_min or s["PB"] > pb_max:
            return False
    
    # P/S filter
    if s["PS"]:
        if s["PS"] < ps_min or s["PS"] > ps_max:
            return False
    
    # Dividend yield
    if s["DividendYield"] < min_dividend:
        return False
    
    # Beta filter
    if s["Beta"]:
        if s["Beta"] < beta_min or s["Beta"] > beta_max:
            return False
    
    # EPS growth filter
    if s["EPSGrowth"] is not None and s["EPSGrowth"] < eps_growth_min:
        return False
    
    # Performance filter
    if perf_period != "Any":
        period_map = {
            "1 Week": "Week",
            "1 Month": "Month",
            "3 Months": "3Months",
            "6 Months": "6Months",
            "1 Year": "Year"
        }
        val = s.get(period_map[perf_period])
        if val is not None:
            if val < perf_min or val > perf_max:
                return False
    
    # RSI filter
    if s["RSI"]:
        if s["RSI"] < rsi_min or s["RSI"] > rsi_max:
            return False
    
    # SMA filters
    if above_sma20 and (s["SMA20"] is None or s["Price"] < s["SMA20"]):
        return False
    if above_sma50 and (s["SMA50"] is None or s["Price"] < s["SMA50"]):
        return False
    if above_sma200 and (s["SMA200"] is None or s["Price"] < s["SMA200"]):
        return False
    
    # Sector filter
    if "Any" not in sector_filter and s["Sector"] not in sector_filter:
        return False
    
    return True

# =============================
# FORMAT DATA FOR DISPLAY
# =============================
def format_dataframe(df):
    """Format DataFrame with color coding and better display"""
    df = df.copy()
    
    # Format numeric columns
    if "Price" in df.columns:
        df["Price"] = df["Price"].apply(lambda x: f"${x:,.2f}")
    
    if "MarketCap" in df.columns:
        df["MarketCap"] = df["MarketCap"].apply(lambda x: format_market_cap(x) if pd.notna(x) else "N/A")
    
    # Format percentages
    percent_cols = ["DividendYield", "Week", "Month", "3Months", "6Months", "Year", "EPSGrowth"]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Format ratios
    ratio_cols = ["PE", "PB", "PS", "Beta", "RSI"]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    return df

def format_market_cap(mc):
    """Format market cap in readable format"""
    if mc >= 1e12:
        return f"${mc/1e12:.2f}T"
    elif mc >= 1e9:
        return f"${mc/1e9:.2f}B"
    elif mc >= 1e6:
        return f"${mc/1e6:.2f}M"
    else:
        return f"${mc:,.0f}"

# =============================
# MAIN APP LOGIC
# =============================
if run_button:
    universe = get_stock_universe(markets)
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.info(f"🔍 Screening {len(universe)} stocks with advanced filters...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    results = []
    errors = 0
    
    for i, sym in enumerate(universe):
        progress = (i + 1) / len(universe)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {sym} ({i+1}/{len(universe)})")
        
        try:
            data = fetch_stock(sym)
            if passes_filters(data):
                results.append(data)
        except Exception as e:
            errors += 1
        
        time.sleep(0.03)  # Rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        df = pd.DataFrame(results)
        
        # Sort by market cap (largest first) or by performance
        if "MarketCap" in df.columns:
            df = df.sort_values("MarketCap", ascending=False, na_position='last')
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stocks Found", len(df))
        with col2:
            avg_pe = df["PE"].mean() if "PE" in df.columns and df["PE"].notna().any() else None
            st.metric("Avg P/E", f"{avg_pe:.2f}" if avg_pe else "N/A")
        with col3:
            avg_perf = df["Year"].mean() if "Year" in df.columns and df["Year"].notna().any() else None
            st.metric("Avg YTD Return", f"{avg_perf:.2f}%" if avg_perf else "N/A")
        with col4:
            total_mc = df["MarketCap"].sum() if "MarketCap" in df.columns and df["MarketCap"].notna().any() else None
            st.metric("Total Market Cap", format_market_cap(total_mc) if total_mc else "N/A")
        
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Table View", "📈 Performance Chart", "📉 Fundamentals Chart"])
        
        with tab1:
            # Display formatted table
            df_display = format_dataframe(df)
            
            # Reorder columns for better display
            display_cols = ["Ticker", "Price", "MarketCap", "PE", "PB", "PS", 
                          "DividendYield", "Beta", "EPSGrowth",
                          "RSI", "Week", "Month", "Year", "Sector"]
            available_cols = [col for col in display_cols if col in df_display.columns]
            df_display = df_display[available_cols]
            
            st.dataframe(
                df_display,
                use_container_width=True,
                height=600,
                hide_index=True
            )
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"finviz_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with tab2:
            # Performance comparison chart
            if len(df) > 0:
                perf_df = df[["Ticker", "Week", "Month", "3Months", "6Months", "Year"]].copy()
                perf_df = perf_df.set_index("Ticker")
                perf_df = perf_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='all')
                
                if not perf_df.empty:
                    fig = px.bar(
                        perf_df.T,
                        title="Performance Comparison by Period",
                        labels={"value": "Return (%)", "index": "Period"},
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_layout(height=600, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Fundamentals comparison
            if len(df) > 0:
                fund_df = df[["Ticker", "PE", "PB", "PS"]].copy()
                fund_df = fund_df.set_index("Ticker")
                fund_df = fund_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='all')
                
                if not fund_df.empty:
                    fig = px.scatter(
                        fund_df.reset_index(),
                        x="PE",
                        y="PB",
                        size="PS",
                        hover_data=["Ticker"],
                        title="Fundamentals Scatter (PE vs PB, sized by PS)",
                        labels={"PE": "P/E Ratio", "PB": "P/B Ratio"}
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No stocks match your criteria. Try adjusting your filters.")
        if errors > 0:
            st.info(f"ℹ️ {errors} stocks could not be processed.")

else:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to AC Stock Screener</h2>
        <p style="font-size: 1.1em;">Set your filters in the sidebar and click <strong>RUN SCREENER</strong> to find stocks matching your criteria.</p>
        <br>
        <h3>✨ Features:</h3>
        <ul style="text-align: left; display: inline-block;">
        <li>Advanced fundamental filters (P/E, P/B, P/S, Beta, EPS Growth)</li>
        <li>Technical indicators (RSI, Moving Averages)</li>
        <li>Performance analysis across multiple timeframes</li>
        <li>Interactive charts and visualizations</li>
        <li>Export results to CSV</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "<center>AC Finviz-Style Stock Screener | Powered by Yahoo Finance<br>"
    "<small>For educational purposes only | Not financial advice</small></center>",
    unsafe_allow_html=True
)

