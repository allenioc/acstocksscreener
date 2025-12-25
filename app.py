import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Professional Stock Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CUSTOM CSS - ENHANCED
# =============================
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
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
    .stDataFrame { font-size: 0.85rem; }
    h1, h2, h3 {
        color: #1f1f1f;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE INITIALIZATION
# =============================
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'screened_stocks' not in st.session_state:
    st.session_state.screened_stocks = None

# =============================
# HEADER
# =============================
st.markdown("""
<div style="background: linear-gradient(135deg,#667eea,#764ba2);
padding:2rem;border-radius:12px;margin-bottom:1.5rem;color:white;text-align:center;box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
<h1 style="color:white;margin:0;">Professional Stock Screener</h1>
<p style="color:white;font-size:1.1em;margin:0.5rem 0 0 0;">Advanced Stock Analysis & Screening Platform</p>
</div>
""", unsafe_allow_html=True)

# =============================
# SIDEBAR FILTERS
# =============================
with st.sidebar:
    st.header("Screening Filters")
    
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
    st.subheader("Price")
    price_min, price_max = st.slider(
        "Price Range ($)",
        0.0, 5000.0, (0.0, 5000.0),
        step=0.1
    )
    
    # Volume Filter
    st.subheader("Volume & Liquidity")
    min_volume = st.selectbox(
        "Min Avg Volume",
        [0, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000],
        index=0
    )
    
    st.markdown("---")
    
    # Fundamental Filters
    st.subheader("Fundamentals")
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
    
    ev_ebitda_min, ev_ebitda_max = st.slider(
        "EV/EBITDA",
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
    revenue_growth_min = st.number_input("Min Revenue Growth (YoY %)", value=-100.0, step=1.0)
    
    st.markdown("---")
    
    # Performance Filters
    st.subheader("Performance")
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
    st.subheader("Technical Indicators")
    rsi_min, rsi_max = st.slider(
        "RSI (14)",
        0.0, 100.0, (0.0, 100.0),
        step=1.0
    )
    
    above_sma20 = st.checkbox("Above SMA 20")
    above_sma50 = st.checkbox("Above SMA 50")
    above_sma200 = st.checkbox("Above SMA 200")
    
    macd_bullish = st.checkbox("MACD Bullish Signal")
    bb_position = st.selectbox("Bollinger Band Position", ["Any", "Above Upper", "Below Lower", "Between Bands"])
    
    st.markdown("---")
    
    # Sector Filter
    st.subheader("Sectors & Industries")
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
    run_button = st.button("RUN SCREENER", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.markdown("**Watchlist:** " + str(len(st.session_state.watchlist)) + " stocks")

# =============================
# EXPANDED STOCK UNIVERSE
# =============================
@st.cache_data(ttl=86400)
def get_stock_universe(markets):
    stocks = []
    
    if "US" in markets:
        # S&P 500 Major Components
        stocks += ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MU']
        # Finance
        stocks += ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PYPL', 'SCHW', 'BLK', 'COF']
        # Healthcare
        stocks += ['JNJ', 'UNH', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'CVS', 'CI', 'HUM', 'ELV']
        # Consumer
        stocks += ['WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'PG', 'KO', 'PEP', 'CL', 'UL', 'DIS', 'CMCSA']
        # Industrial
        stocks += ['CAT', 'GE', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'ETN', 'EMR']
        # Energy
        stocks += ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO']
        # Tech/Software
        stocks += ['ADBE', 'CRM', 'ORCL', 'IBM', 'CSCO', 'INTU', 'NOW', 'SNPS', 'CDNS', 'ANSS']
        # Communication
        stocks += ['VZ', 'T', 'TMUS', 'LUMN']
        # Utilities
        stocks += ['NEE', 'DUK', 'SO', 'AEP']
        # Real Estate
        stocks += ['AMT', 'PLD', 'EQIX', 'SPG']
        # Materials
        stocks += ['LIN', 'APD', 'ECL', 'SHW']
    
    if "Canada" in markets:
        stocks += [
            'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'NA.TO',
            'ENB.TO', 'CNQ.TO', 'SU.TO', 'IMO.TO', 'TRP.TO',
            'SHOP.TO', 'OTEX.TO', 'WCN.TO', 'CGI.TO',
            'BCE.TO', 'T.TO', 'RCI-B.TO', 'QBR-B.TO',
            'CNR.TO', 'CP.TO',
            'ATD.TO', 'L.TO', 'MRU.TO'
        ]
    
    return list(set(stocks))

# =============================
# ADVANCED TECHNICAL INDICATORS
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
    return prices.rolling(window=period).mean()

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None
    return prices.ewm(span=period, adjust=False).mean()

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    if len(prices) < slow:
        return None, None, None
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper.iloc[-1], sma.iloc[-1], lower.iloc[-1]

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    if len(close) < k_period:
        return None, None
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent.iloc[-1], d_percent.iloc[-1]

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(close) < period + 1:
        return None
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.iloc[-1]

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe Ratio"""
    if len(returns) < 2:
        return None
    excess_returns = returns - (risk_free_rate / 252)
    if excess_returns.std() == 0:
        return None
    return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))

def calculate_volatility(returns, period=252):
    """Calculate Annualized Volatility"""
    if len(returns) < 2:
        return None
    return returns.std() * np.sqrt(period)

# =============================
# FETCH HISTORICAL DATA
# =============================
@st.cache_data(ttl=900)
def fetch_historical_data(symbol, period="2y"):
    """Fetch historical data for charting"""
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period)
        if hist is None or hist.empty:
            return None
        return hist
    except:
        return None

# =============================
# FETCH STOCK DATA (COMPREHENSIVE)
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
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]
        
        # Calculate returns for risk metrics
        returns = prices.pct_change().dropna()
        
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
        
        sma20_val = sma20.iloc[-1] if sma20 is not None else None
        sma50_val = sma50.iloc[-1] if sma50 is not None else None
        sma200_val = sma200.iloc[-1] if sma200 is not None else None
        
        macd, macd_signal, macd_hist = calculate_macd(prices)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
        stoch_k, stoch_d = calculate_stochastic(high, low, prices)
        atr = calculate_atr(high, low, prices)
        
        # Risk metrics
        sharpe = calculate_sharpe_ratio(returns)
        volatility = calculate_volatility(returns)
        
        # Calculate earnings and revenue growth
        try:
            earnings_growth = info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth')
            if earnings_growth:
                eps_growth = float(earnings_growth) * 100
            else:
                eps_growth = None
        except:
            eps_growth = None
        
        try:
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth:
                revenue_growth_pct = float(revenue_growth) * 100
            else:
                revenue_growth_pct = None
        except:
            revenue_growth_pct = None
        
        # MACD bullish signal
        macd_bullish_signal = (macd is not None and macd_signal is not None and macd > macd_signal and macd_hist > 0) if macd and macd_signal else False
        
        # Bollinger Band position
        bb_pos = None
        if bb_upper and bb_lower:
            if current_price > bb_upper:
                bb_pos = "Above Upper"
            elif current_price < bb_lower:
                bb_pos = "Below Lower"
            else:
                bb_pos = "Between Bands"
        
        return {
            "Ticker": symbol,
            "Price": current_price,
            "Volume": info.get("averageVolume") or info.get("volume24Hr"),
            "MarketCap": info.get("marketCap"),
            "PE": info.get("trailingPE") or info.get("forwardPE"),
            "PB": info.get("priceToBook"),
            "PS": info.get("priceToSalesTrailing12Months"),
            "EV_EBITDA": info.get("enterpriseToEbitda"),
            "DividendYield": (info.get("dividendYield") or 0) * 100,
            "Beta": info.get("beta"),
            "EPSGrowth": eps_growth,
            "RevenueGrowth": revenue_growth_pct,
            "Week": pct_change(5),
            "Month": pct_change(21),
            "3Months": pct_change(63),
            "6Months": pct_change(126),
            "Year": pct_change(252),
            "RSI": rsi,
            "SMA20": sma20_val,
            "SMA50": sma50_val,
            "SMA200": sma200_val,
            "MACD": macd,
            "MACD_Signal": macd_signal,
            "MACD_Hist": macd_hist,
            "MACD_Bullish": macd_bullish_signal,
            "BB_Upper": bb_upper,
            "BB_Middle": bb_middle,
            "BB_Lower": bb_lower,
            "BB_Position": bb_pos,
            "Stoch_K": stoch_k,
            "Stoch_D": stoch_d,
            "ATR": atr,
            "Sharpe": sharpe,
            "Volatility": volatility,
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "52WHigh": info.get("fiftyTwoWeekHigh"),
            "52WLow": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        return None

# =============================
# FILTER LOGIC
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
    if s.get("PE"):
        if s["PE"] < pe_min or s["PE"] > pe_max:
            return False
    
    # P/B filter
    if s.get("PB"):
        if s["PB"] < pb_min or s["PB"] > pb_max:
            return False
    
    # P/S filter
    if s.get("PS"):
        if s["PS"] < ps_min or s["PS"] > ps_max:
            return False
    
    # EV/EBITDA filter
    if s.get("EV_EBITDA"):
        if s["EV_EBITDA"] < ev_ebitda_min or s["EV_EBITDA"] > ev_ebitda_max:
            return False
    
    # Dividend yield
    if s.get("DividendYield", 0) < min_dividend:
        return False
    
    # Beta filter
    if s.get("Beta"):
        if s["Beta"] < beta_min or s["Beta"] > beta_max:
            return False
    
    # EPS growth filter
    if s.get("EPSGrowth") is not None and s["EPSGrowth"] < eps_growth_min:
        return False
    
    # Revenue growth filter
    if s.get("RevenueGrowth") is not None and s["RevenueGrowth"] < revenue_growth_min:
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
    if s.get("RSI"):
        if s["RSI"] < rsi_min or s["RSI"] > rsi_max:
            return False
    
    # SMA filters
    if above_sma20 and (s.get("SMA20") is None or s["Price"] < s["SMA20"]):
        return False
    if above_sma50 and (s.get("SMA50") is None or s["Price"] < s["SMA50"]):
        return False
    if above_sma200 and (s.get("SMA200") is None or s["Price"] < s["SMA200"]):
        return False
    
    # MACD filter
    if macd_bullish and not s.get("MACD_Bullish", False):
        return False
    
    # Bollinger Band filter
    if bb_position != "Any" and s.get("BB_Position") != bb_position:
        return False
    
    # Sector filter
    if "Any" not in sector_filter and s.get("Sector") not in sector_filter:
        return False
    
    return True

# =============================
# CHART FUNCTIONS
# =============================
def create_advanced_candlestick_chart(hist, symbol, indicators=['SMA', 'BB', 'MACD']):
    """Create advanced candlestick chart with multiple indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price Chart', 'Volume', 'Technical Indicators'),
        row_width=[0.5, 0.2, 0.3]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if 'SMA' in indicators:
        if len(hist) >= 20:
            sma20 = hist['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(x=hist.index, y=sma20, name='SMA 20', line=dict(color='orange', width=1)),
                row=1, col=1
            )
        if len(hist) >= 50:
            sma50 = hist['Close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(x=hist.index, y=sma50, name='SMA 50', line=dict(color='blue', width=1)),
                row=1, col=1
            )
        if len(hist) >= 200:
            sma200 = hist['Close'].rolling(window=200).mean()
            fig.add_trace(
                go.Scatter(x=hist.index, y=sma200, name='SMA 200', line=dict(color='red', width=1)),
                row=1, col=1
            )
    
    # Bollinger Bands
    if 'BB' in indicators and len(hist) >= 20:
        bb_period = 20
        sma = hist['Close'].rolling(window=bb_period).mean()
        std = hist['Close'].rolling(window=bb_period).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        
        fig.add_trace(
            go.Scatter(x=hist.index, y=bb_upper, name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=bb_lower, name='BB Lower', line=dict(color='gray', width=1, dash='dash'), fill='tonexty'),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if hist['Close'].iloc[i] < hist['Open'].iloc[i] else 'green' for i in range(len(hist))]
    fig.add_trace(
        go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # MACD
    if 'MACD' in indicators and len(hist) >= 26:
        ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        fig.add_trace(
            go.Scatter(x=hist.index, y=macd, name='MACD', line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=signal, name='Signal', line=dict(color='red', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=hist.index, y=histogram, name='Histogram', marker_color='gray'),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=3, col=1)
    
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    return fig

def create_rsi_chart(hist, symbol):
    """Create RSI indicator chart"""
    if len(hist) < 15:
        return None
    
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=rsi, mode='lines', name='RSI', line=dict(color='purple', width=2)))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.update_layout(title=f'{symbol} - RSI (14)', yaxis_title='RSI', xaxis_title='Date', yaxis_range=[0, 100], height=300)
    return fig

# =============================
# FORMATTING FUNCTIONS
# =============================
def format_dataframe(df):
    """Format DataFrame for display"""
    df = df.copy()
    
    if "Price" in df.columns:
        df["Price"] = df["Price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    
    if "MarketCap" in df.columns:
        df["MarketCap"] = df["MarketCap"].apply(lambda x: format_market_cap(x) if pd.notna(x) else "N/A")
    
    percent_cols = ["DividendYield", "Week", "Month", "3Months", "6Months", "Year", "EPSGrowth", "RevenueGrowth", "Volatility"]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    ratio_cols = ["PE", "PB", "PS", "EV_EBITDA", "Beta", "RSI", "Stoch_K", "Stoch_D", "Sharpe"]
    for col in ratio_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    return df

def format_market_cap(mc):
    """Format market cap"""
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
    
    progress_container = st.container()
    with progress_container:
        st.info(f"Screening {len(universe)} stocks with advanced filters...")
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
        
        time.sleep(0.02)
    
    progress_bar.empty()
    status_text.empty()
    
    if results:
        df = pd.DataFrame(results)
        
        if "MarketCap" in df.columns:
            df = df.sort_values("MarketCap", ascending=False, na_position='last')
        
        st.session_state.screened_stocks = df
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Stocks Found", len(df))
        with col2:
            avg_pe = df["PE"].mean() if "PE" in df.columns and df["PE"].notna().any() else None
            st.metric("Avg P/E", f"{avg_pe:.2f}" if avg_pe else "N/A")
        with col3:
            avg_perf = df["Year"].mean() if "Year" in df.columns and df["Year"].notna().any() else None
            st.metric("Avg YTD Return", f"{avg_perf:.2f}%" if avg_perf else "N/A")
        with col4:
            avg_sharpe = df["Sharpe"].mean() if "Sharpe" in df.columns and df["Sharpe"].notna().any() else None
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}" if avg_sharpe else "N/A")
        with col5:
            total_mc = df["MarketCap"].sum() if "MarketCap" in df.columns and df["MarketCap"].notna().any() else None
            st.metric("Total Market Cap", format_market_cap(total_mc) if total_mc else "N/A")
        
        st.markdown("---")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Data Table", "Performance Analysis", "Sector Analysis", "Correlation Matrix", 
            "Stock Charts", "Watchlist"
        ])
        
        with tab1:
            # Sortable columns
            sort_col = st.selectbox("Sort by:", ["MarketCap", "Price", "PE", "Year", "RSI", "Sharpe"])
            sort_asc = st.checkbox("Ascending", value=False)
            df_sorted = df.sort_values(sort_col, ascending=sort_asc, na_position='last')
            
            df_display = format_dataframe(df_sorted)
            
            display_cols = ["Ticker", "Price", "MarketCap", "PE", "PB", "PS", "EV_EBITDA",
                          "DividendYield", "Beta", "EPSGrowth", "RevenueGrowth",
                          "RSI", "MACD_Bullish", "Sharpe", "Volatility",
                          "Week", "Month", "Year", "Sector"]
            available_cols = [col for col in display_cols if col in df_display.columns]
            df_display = df_display[available_cols]
            
            st.dataframe(df_display, use_container_width=True, height=600, hide_index=True)
            
            # Export buttons
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
            with col2:
                try:
                    import io
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Screener Results')
                    excel_data = output.getvalue()
                    st.download_button("Download Excel", excel_data, f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except:
                    st.info("Excel export requires openpyxl")
            
            # Add to watchlist
            selected_tickers = st.multiselect("Add to Watchlist:", df_sorted["Ticker"].tolist())
            if st.button("Add Selected to Watchlist"):
                for ticker in selected_tickers:
                    if ticker not in st.session_state.watchlist:
                        st.session_state.watchlist.append(ticker)
                st.success(f"Added {len(selected_tickers)} stocks to watchlist")
                st.rerun()
        
        with tab2:
            # Performance heatmap
            perf_cols = ["Week", "Month", "3Months", "6Months", "Year"]
            perf_df = df[["Ticker"] + [col for col in perf_cols if col in df.columns]].set_index("Ticker")
            perf_df = perf_df.replace([np.inf, -np.inf], np.nan)
            
            if not perf_df.empty:
                fig = px.imshow(
                    perf_df.T,
                    labels=dict(x="Stock", y="Period", color="Return (%)"),
                    title="Performance Heatmap (%)",
                    color_continuous_scale="RdYlGn",
                    aspect="auto"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance comparison
            if len(perf_df.columns) > 0:
                fig = px.bar(perf_df.T, title="Performance Comparison", labels={"value": "Return (%)", "index": "Period"})
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Sector analysis
            if "Sector" in df.columns:
                sector_stats = df.groupby("Sector").agg({
                    "Ticker": "count",
                    "Year": "mean",
                    "PE": "mean",
                    "MarketCap": "sum"
                }).round(2)
                sector_stats.columns = ["Count", "Avg Return (%)", "Avg P/E", "Total Market Cap"]
                sector_stats = sector_stats.sort_values("Count", ascending=False)
                
                st.subheader("Sector Breakdown")
                st.dataframe(sector_stats, use_container_width=True)
                
                # Sector performance chart
                fig = px.bar(
                    sector_stats.reset_index(),
                    x="Sector",
                    y="Avg Return (%)",
                    title="Average Performance by Sector",
                    color="Avg Return (%)",
                    color_continuous_scale="RdYlGn"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Correlation matrix
            numeric_cols = ["Price", "PE", "PB", "PS", "Beta", "RSI", "Year", "Month", "Week"]
            corr_cols = [col for col in numeric_cols if col in df.columns]
            if len(corr_cols) > 1:
                corr_df = df[corr_cols].corr()
                
                fig = px.imshow(
                    corr_df,
                    labels=dict(color="Correlation"),
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu",
                    aspect="auto",
                    text_auto=True
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("Advanced Stock Charts")
            
            if len(df) > 0:
                ticker_list = df["Ticker"].tolist()
                selected_ticker = st.selectbox("Select Stock:", ticker_list, key="chart_ticker")
                
                if selected_ticker:
                    chart_period = st.selectbox("Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3, key="period")
                    
                    hist_data = fetch_historical_data(selected_ticker, period=chart_period)
                    
                    if hist_data is not None and not hist_data.empty:
                        current_price = hist_data['Close'].iloc[-1]
                        price_change = current_price - hist_data['Close'].iloc[-2] if len(hist_data) > 1 else 0
                        price_change_pct = (price_change / hist_data['Close'].iloc[-2] * 100) if len(hist_data) > 1 and hist_data['Close'].iloc[-2] != 0 else 0
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                        with col3:
                            st.metric("High", f"${hist_data['High'].max():.2f}")
                        with col4:
                            st.metric("Low", f"${hist_data['Low'].min():.2f}")
                        with col5:
                            st.metric("Volume", f"{hist_data['Volume'].iloc[-1]:,.0f}")
                        
                        st.markdown("---")
                        
                        # Indicator selection
                        indicators = st.multiselect("Indicators:", ["SMA", "BB", "MACD"], default=["SMA", "MACD"])
                        
                        chart_fig = create_advanced_candlestick_chart(hist_data, selected_ticker, indicators)
                        st.plotly_chart(chart_fig, use_container_width=True)
                        
                        # RSI chart
                        rsi_fig = create_rsi_chart(hist_data, selected_ticker)
                        if rsi_fig:
                            st.plotly_chart(rsi_fig, use_container_width=True)
                    else:
                        st.warning(f"Could not fetch data for {selected_ticker}")
        
        with tab6:
            st.subheader("Watchlist")
            
            if st.session_state.watchlist:
                watchlist_df = df[df["Ticker"].isin(st.session_state.watchlist)] if st.session_state.screened_stocks is not None else pd.DataFrame()
                
                if not watchlist_df.empty:
                    watchlist_display = format_dataframe(watchlist_df)
                    st.dataframe(watchlist_display, use_container_width=True, height=400)
                    
                    # Remove from watchlist
                    remove_tickers = st.multiselect("Remove from Watchlist:", st.session_state.watchlist)
                    if st.button("Remove Selected"):
                        st.session_state.watchlist = [t for t in st.session_state.watchlist if t not in remove_tickers]
                        st.rerun()
                else:
                    st.info("No watchlist stocks found in current screening results.")
            else:
                st.info("Your watchlist is empty. Add stocks from the Data Table tab.")
            
            if st.button("Clear Watchlist"):
                st.session_state.watchlist = []
                st.rerun()
    else:
        st.warning("No stocks match your criteria. Try adjusting your filters.")
        if errors > 0:
            st.info(f"{errors} stocks could not be processed.")

else:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to Professional Stock Screener</h2>
        <p style="font-size: 1.1em;">Advanced stock screening and analysis platform with comprehensive technical and fundamental analysis tools.</p>
        <br>
        <h3>Key Features:</h3>
        <ul style="text-align: left; display: inline-block;">
        <li>Advanced fundamental filters (P/E, P/B, P/S, EV/EBITDA, Growth metrics)</li>
        <li>Technical indicators (RSI, MACD, Bollinger Bands, Stochastic, ATR)</li>
        <li>Risk metrics (Sharpe Ratio, Volatility)</li>
        <li>Performance analysis across multiple timeframes</li>
        <li>Interactive charts with multiple indicators</li>
        <li>Sector and correlation analysis</li>
        <li>Watchlist functionality</li>
        <li>Export capabilities</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "<center>Professional Stock Screener | Powered by Yahoo Finance<br>"
    "<small>For educational purposes only | Not financial advice</small></center>",
    unsafe_allow_html=True
)
