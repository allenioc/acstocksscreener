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
from theme import (
    get_global_css,
    render_top_nav,
    filter_panel_open,
    filter_panel_close,
    results_bar,
    kpi_strip,
    format_pct,
    delta_class,
    apply_dark_plotly_layout,
    THEME,
)
from filter_logic import build_filter_config, DEFAULT_BOUNDS
from screener_backend import (
    run_screener_pipeline,
    format_screener_results,
    UNIVERSE_SIZE_OPTIONS,
)
warnings.filterwarnings('ignore')

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Stock Screener",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================
# GLOBAL THEME
# =============================
st.markdown(f"<style>{get_global_css()}</style>", unsafe_allow_html=True)

# =============================
# SESSION STATE INITIALIZATION
# =============================
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'screened_stocks' not in st.session_state:
    st.session_state.screened_stocks = None

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
        
        # Get financial statements for ratio calculations
        try:
            balance_sheet = t.balance_sheet
            income_stmt = t.financials
            cashflow = t.cashflow
        except:
            balance_sheet = None
            income_stmt = None
            cashflow = None
        
        # Calculate financial ratios from financial statements
        roe = None
        roa = None
        current_ratio = None
        quick_ratio = None
        debt_to_equity = None
        debt_to_assets = None
        equity_ratio = None
        interest_coverage = None
        profit_margin = None
        operating_margin = None
        
        if balance_sheet is not None and not balance_sheet.empty:
            # Get most recent period (first column)
            try:
                # Total assets
                total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
                # Total equity
                total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else None
                if total_equity is None:
                    total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else None
                # Current assets
                current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else None
                # Current liabilities
                current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else None
                # Cash and equivalents
                cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance_sheet.index else None
                # Total debt
                total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else None
                if total_debt is None:
                    # Try to calculate from components
                    long_term_debt = balance_sheet.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance_sheet.index else 0
                    short_term_debt = balance_sheet.loc['Short Term Debt'].iloc[0] if 'Short Term Debt' in balance_sheet.index else 0
                    total_debt = long_term_debt + short_term_debt if (long_term_debt or short_term_debt) else None
                
                # Calculate ratios
                if total_equity and total_equity > 0:
                    equity_ratio = (total_equity / total_assets * 100) if total_assets and total_assets > 0 else None
                    debt_to_equity = (total_debt / total_equity) if total_debt is not None and total_debt > 0 else None
                
                if total_assets and total_assets > 0:
                    debt_to_assets = (total_debt / total_assets * 100) if total_debt is not None and total_debt > 0 else None
                
                if current_assets and current_liabilities and current_liabilities > 0:
                    current_ratio = current_assets / current_liabilities
                    if cash is not None:
                        quick_ratio = (current_assets - (balance_sheet.loc['Inventory'].iloc[0] if 'Inventory' in balance_sheet.index else 0)) / current_liabilities
            except:
                pass
        
        if income_stmt is not None and not income_stmt.empty:
            try:
                # Net income
                net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else None
                # Operating income
                operating_income = income_stmt.loc['Operating Income'].iloc[0] if 'Operating Income' in income_stmt.index else None
                if operating_income is None:
                    operating_income = income_stmt.loc['Operating Income Or Loss'].iloc[0] if 'Operating Income Or Loss' in income_stmt.index else None
                # Total revenue
                total_revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else None
                if total_revenue is None:
                    total_revenue = income_stmt.loc['Revenue'].iloc[0] if 'Revenue' in income_stmt.index else None
                # Interest expense
                interest_expense = income_stmt.loc['Interest Expense'].iloc[0] if 'Interest Expense' in income_stmt.index else None
                if interest_expense is None:
                    interest_expense = abs(income_stmt.loc['Interest And Dividend Income'].iloc[0]) if 'Interest And Dividend Income' in income_stmt.index else None
                
                # Calculate ROE
                if net_income and total_equity and total_equity > 0:
                    roe = (net_income / total_equity) * 100
                
                # Calculate ROA
                if net_income and total_assets and total_assets > 0:
                    roa = (net_income / total_assets) * 100
                
                # Calculate margins
                if total_revenue and total_revenue > 0:
                    if net_income:
                        profit_margin = (net_income / total_revenue) * 100
                    if operating_income:
                        operating_margin = (operating_income / total_revenue) * 100
                
                # Interest coverage
                if operating_income and interest_expense and interest_expense > 0:
                    interest_coverage = operating_income / abs(interest_expense)
            except:
                pass
        
        # Try to get ratios from info if not calculated
        if roe is None:
            roe = info.get('returnOnEquity') * 100 if info.get('returnOnEquity') else None
        if roa is None:
            roa = info.get('returnOnAssets') * 100 if info.get('returnOnAssets') else None
        if current_ratio is None:
            current_ratio = info.get('currentRatio')
        if debt_to_equity is None:
            debt_to_equity = info.get('debtToEquity')
        if profit_margin is None:
            profit_margin = info.get('profitMargins') * 100 if info.get('profitMargins') else None
        if operating_margin is None:
            operating_margin = info.get('operatingMargins') * 100 if info.get('operatingMargins') else None
        
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
            # Price Ratios
            "PE": info.get("trailingPE") or info.get("forwardPE"),
            "PB": info.get("priceToBook"),
            "PS": info.get("priceToSalesTrailing12Months"),
            "PEG": info.get("pegRatio"),
            "EV_EBITDA": info.get("enterpriseToEbitda"),
            "EV_Sales": info.get("enterpriseToRevenue"),
            "DividendYield": (info.get("dividendYield") or 0) * 100,
            # Profitability Ratios
            "ROE": roe,
            "ROA": roa,
            "ProfitMargin": profit_margin,
            "OperatingMargin": operating_margin,
            # Leverage Ratios
            "DebtToEquity": debt_to_equity,
            "DebtToAssets": debt_to_assets,
            "EquityRatio": equity_ratio,
            "InterestCoverage": interest_coverage,
            # Liquidity Ratios
            "CurrentRatio": current_ratio,
            "QuickRatio": quick_ratio,
            # Other
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
# CHART FUNCTIONS
# =============================
def create_advanced_candlestick_chart(hist, symbol, indicators=['SMA', 'BB', 'MACD']):
    """Create advanced candlestick chart with multiple indicators"""
    t = THEME
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
            name='Price',
            increasing_line_color=t['positive'],
            increasing_fillcolor=t['positive'],
            decreasing_line_color=t['negative'],
            decreasing_fillcolor=t['negative'],
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if 'SMA' in indicators:
        if len(hist) >= 20:
            sma20 = hist['Close'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(x=hist.index, y=sma20, name='SMA 20', line=dict(color='#38bdf8', width=1)),
                row=1, col=1
            )
        if len(hist) >= 50:
            sma50 = hist['Close'].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(x=hist.index, y=sma50, name='SMA 50', line=dict(color=t['accent_blue'], width=1)),
                row=1, col=1
            )
        if len(hist) >= 200:
            sma200 = hist['Close'].rolling(window=200).mean()
            fig.add_trace(
                go.Scatter(x=hist.index, y=sma200, name='SMA 200', line=dict(color='#facc15', width=1)),
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
            go.Scatter(x=hist.index, y=bb_upper, name='BB Upper', line=dict(color=t['neutral'], width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=bb_lower, name='BB Lower', line=dict(color=t['neutral'], width=1, dash='dash'), fill='tonexty'),
            row=1, col=1
        )
    
    # Volume
    colors = [t['negative'] if hist['Close'].iloc[i] < hist['Open'].iloc[i] else t['positive'] for i in range(len(hist))]
    fig.add_trace(
        go.Bar(x=hist.index, y=hist['Volume'], name='Volume', marker_color=colors, opacity=0.75),
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
            go.Scatter(x=hist.index, y=macd, name='MACD', line=dict(color=t['accent_blue'], width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=hist.index, y=signal, name='Signal', line=dict(color='#facc15', width=2)),
            row=3, col=1
        )
        hist_colors = [t['positive'] if v >= 0 else t['negative'] for v in histogram]
        fig.add_trace(
            go.Bar(x=hist.index, y=histogram, name='Histogram', marker_color=hist_colors, opacity=0.7),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color=t['border_default'], row=3, col=1)
    
    fig.update_xaxes(rangeslider_visible=False)
    apply_dark_plotly_layout(fig, height=900)
    fig.update_layout(showlegend=True, hovermode='x unified')
    for annotation in fig.layout.annotations:
        annotation.font.color = t['text_secondary']
        annotation.font.size = 12
    return fig

def create_rsi_chart(hist, symbol):
    """Create RSI indicator chart"""
    t = THEME
    if len(hist) < 15:
        return None
    
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=rsi, mode='lines', name='RSI', line=dict(color=t['accent_blue'], width=2)))
    fig.add_hline(y=70, line_dash="dash", line_color=t['negative'], annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color=t['positive'], annotation_text="Oversold (30)")
    apply_dark_plotly_layout(fig, title=f'{symbol} - RSI (14)', height=300)
    fig.update_layout(yaxis_title='RSI', xaxis_title='Date', yaxis_range=[0, 100])
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
    
    # Percentage columns
    percent_cols = ["DividendYield", "EPSGrowth", "RevenueGrowth", "Volatility", "ROE", "ROA", 
                   "ProfitMargin", "OperatingMargin", "DebtToAssets", "EquityRatio"]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: format_pct(x) if pd.notna(x) else "N/A")

    perf_cols = ["Week", "Month", "3Months", "6Months", "Year"]
    for col in perf_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: format_pct(x) if pd.notna(x) else "N/A")
    
    # Ratio columns (numeric values)
    ratio_cols = ["PE", "PB", "PS", "PEG", "EV_EBITDA", "EV_Sales", "Beta", "RSI", 
                 "Stoch_K", "Stoch_D", "Sharpe", "DebtToEquity", "InterestCoverage",
                 "CurrentRatio", "QuickRatio"]
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
current_page = render_top_nav()

if current_page == "Screener":
    st.markdown(filter_panel_open(), unsafe_allow_html=True)
    st.markdown('<div class="filter-zone">', unsafe_allow_html=True)

    ftab1, ftab2, ftab3, ftab4 = st.tabs(["Descriptive", "Fundamental", "Technical", "Performance"])

    with ftab1:
        r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
        with r1c1:
            markets = st.multiselect("Exchange", ["US", "Canada"], default=["US"], label_visibility="visible")
        with r1c2:
            universe_size = st.selectbox(
                "Universe",
                list(UNIVERSE_SIZE_OPTIONS.keys()),
                index=1,
                help="Limit symbols scanned for speed. Default Top 500.",
            )
        with r1c3:
            market_cap = st.selectbox("Market Cap", ["Any", "Mega (>$200B)", "Large ($10B-$200B)", "Mid ($2B-$10B)", "Small ($300M-$2B)", "Micro (<$300M)"])
        with r1c4:
            sector_filter = st.multiselect(
                "Sector",
                ["Any", "Technology", "Healthcare", "Financial Services",
                 "Consumer Cyclical", "Consumer Defensive", "Industrials",
                 "Energy", "Basic Materials", "Communication Services",
                 "Utilities", "Real Estate"],
                default=["Any"],
            )
        with r1c5:
            price_min, price_max = st.slider("Price ($)", 0.0, 5000.0, (0.0, 5000.0), step=0.1)
        r1b1, r1b2, r1b3, r1b4, r1b5 = st.columns(5)
        with r1b1:
            min_volume = st.selectbox("Avg Volume", [0, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000], index=0)

    with ftab2:
        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
        with r2c1:
            pe_min, pe_max = st.slider("P/E", DEFAULT_BOUNDS["pe"][0], DEFAULT_BOUNDS["pe"][1], DEFAULT_BOUNDS["pe"], step=0.1, key="pe_range_v3")
        with r2c2:
            pb_min, pb_max = st.slider("P/B", DEFAULT_BOUNDS["pb"][0], DEFAULT_BOUNDS["pb"][1], DEFAULT_BOUNDS["pb"], step=0.1, key="pb_range_v3")
        with r2c3:
            ps_min, ps_max = st.slider("P/S", DEFAULT_BOUNDS["ps"][0], DEFAULT_BOUNDS["ps"][1], DEFAULT_BOUNDS["ps"], step=0.1, key="ps_range_v3")
        with r2c4:
            ev_ebitda_min, ev_ebitda_max = st.slider("EV/EBITDA", DEFAULT_BOUNDS["ev_ebitda"][0], DEFAULT_BOUNDS["ev_ebitda"][1], DEFAULT_BOUNDS["ev_ebitda"], step=0.1, key="ev_range_v3")
        with r2c5:
            peg_min, peg_max = st.slider("PEG", DEFAULT_BOUNDS["peg"][0], DEFAULT_BOUNDS["peg"][1], DEFAULT_BOUNDS["peg"], step=0.1, key="peg_range_v3")

        r2b1, r2b2, r2b3, r2b4, r2b5 = st.columns(5)
        with r2b1:
            min_dividend = st.slider("Div Yield %", 0.0, 10.0, 0.0, step=0.1)
        with r2b2:
            roe_min, roe_max = st.slider("ROE %", DEFAULT_BOUNDS["roe"][0], DEFAULT_BOUNDS["roe"][1], DEFAULT_BOUNDS["roe"], step=0.1)
        with r2b3:
            roa_min, roa_max = st.slider("ROA %", DEFAULT_BOUNDS["roa"][0], DEFAULT_BOUNDS["roa"][1], DEFAULT_BOUNDS["roa"], step=0.1)
        with r2b4:
            profit_margin_min, profit_margin_max = st.slider("Profit Margin %", DEFAULT_BOUNDS["profit_margin"][0], DEFAULT_BOUNDS["profit_margin"][1], DEFAULT_BOUNDS["profit_margin"], step=0.1)
        with r2b5:
            operating_margin_min, operating_margin_max = st.slider("Oper. Margin %", DEFAULT_BOUNDS["operating_margin"][0], DEFAULT_BOUNDS["operating_margin"][1], DEFAULT_BOUNDS["operating_margin"], step=0.1)

        r2c1b, r2c2b, r2c3b, r2c4b, r2c5b = st.columns(5)
        with r2c1b:
            debt_to_equity_min, debt_to_equity_max = st.slider("Debt/Equity", DEFAULT_BOUNDS["debt_to_equity"][0], DEFAULT_BOUNDS["debt_to_equity"][1], DEFAULT_BOUNDS["debt_to_equity"], step=1.0, key="debt_equity_range_v3")
        with r2c2b:
            debt_to_assets_min, debt_to_assets_max = st.slider("Debt/Assets %", DEFAULT_BOUNDS["debt_to_assets"][0], DEFAULT_BOUNDS["debt_to_assets"][1], DEFAULT_BOUNDS["debt_to_assets"], step=0.1)
        with r2c3b:
            equity_ratio_min, equity_ratio_max = st.slider("Equity Ratio %", DEFAULT_BOUNDS["equity_ratio"][0], DEFAULT_BOUNDS["equity_ratio"][1], DEFAULT_BOUNDS["equity_ratio"], step=0.1)
        with r2c4b:
            interest_coverage_min, interest_coverage_max = st.slider("Int. Coverage", DEFAULT_BOUNDS["interest_coverage"][0], DEFAULT_BOUNDS["interest_coverage"][1], DEFAULT_BOUNDS["interest_coverage"], step=0.1)
        with r2c5b:
            current_ratio_min, current_ratio_max = st.slider("Current Ratio", DEFAULT_BOUNDS["current_ratio"][0], DEFAULT_BOUNDS["current_ratio"][1], DEFAULT_BOUNDS["current_ratio"], step=0.1, key="current_ratio_range_v3")

        r2d1, r2d2, r2d3, r2d4, r2d5 = st.columns(5)
        with r2d1:
            quick_ratio_min, quick_ratio_max = st.slider("Quick Ratio", DEFAULT_BOUNDS["quick_ratio"][0], DEFAULT_BOUNDS["quick_ratio"][1], DEFAULT_BOUNDS["quick_ratio"], step=0.1, key="quick_ratio_range_v3")
        with r2d2:
            beta_min, beta_max = st.slider("Beta", DEFAULT_BOUNDS["beta"][0], DEFAULT_BOUNDS["beta"][1], DEFAULT_BOUNDS["beta"], step=0.1)
        with r2d3:
            eps_growth_min = st.number_input("EPS Growth %", value=-100.0, step=1.0)
        with r2d4:
            revenue_growth_min = st.number_input("Rev Growth %", value=-100.0, step=1.0)

    with ftab3:
        r3c1, r3c2, r3c3, r3c4, r3c5 = st.columns(5)
        with r3c1:
            rsi_min, rsi_max = st.slider("RSI (14)", 0.0, 100.0, (0.0, 100.0), step=1.0)
        with r3c2:
            bb_position = st.selectbox("Bollinger Bands", ["Any", "Above Upper", "Below Lower", "Between Bands"])
        with r3c3:
            above_sma20 = st.checkbox("Above SMA 20")
        with r3c4:
            above_sma50 = st.checkbox("Above SMA 50")
        with r3c5:
            above_sma200 = st.checkbox("Above SMA 200")
        r3b1, r3b2, r3b3, r3b4, r3b5 = st.columns(5)
        with r3b1:
            macd_bullish = st.checkbox("MACD Bullish")

    with ftab4:
        r4c1, r4c2, r4c3, r4c4, r4c5 = st.columns(5)
        with r4c1:
            perf_period = st.selectbox("Perf Period", ["Any", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year"])
        perf_min, perf_max = -100.0, 300.0
        if perf_period != "Any":
            with r4c2:
                perf_min = st.slider("Min Perf %", -100.0, 300.0, -100.0, step=1.0)
            with r4c3:
                perf_max = st.slider("Max Perf %", -100.0, 300.0, 300.0, step=1.0)

    st.markdown('</div>', unsafe_allow_html=True)

    act1, act2, act3 = st.columns([1, 1, 8])
    with act1:
        st.markdown('<div class="fv-filter-btn">', unsafe_allow_html=True)
        run_button = st.button("▼  Filter", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    with act2:
        st.caption(f"Watchlist: {len(st.session_state.watchlist)}")

    st.markdown(filter_panel_close(), unsafe_allow_html=True)

    if run_button:
        filter_kwargs = dict(
            price_min=price_min, price_max=price_max,
            min_volume=min_volume, market_cap=market_cap, sector_filter=sector_filter,
            pe_min=pe_min, pe_max=pe_max, pb_min=pb_min, pb_max=pb_max,
            ps_min=ps_min, ps_max=ps_max,
            ev_ebitda_min=ev_ebitda_min, ev_ebitda_max=ev_ebitda_max,
            peg_min=peg_min, peg_max=peg_max, min_dividend=min_dividend,
            roe_min=roe_min, roe_max=roe_max, roa_min=roa_min, roa_max=roa_max,
            profit_margin_min=profit_margin_min, profit_margin_max=profit_margin_max,
            operating_margin_min=operating_margin_min, operating_margin_max=operating_margin_max,
            debt_to_equity_min=debt_to_equity_min, debt_to_equity_max=debt_to_equity_max,
            debt_to_assets_min=debt_to_assets_min, debt_to_assets_max=debt_to_assets_max,
            equity_ratio_min=equity_ratio_min, equity_ratio_max=equity_ratio_max,
            interest_coverage_min=interest_coverage_min, interest_coverage_max=interest_coverage_max,
            current_ratio_min=current_ratio_min, current_ratio_max=current_ratio_max,
            quick_ratio_min=quick_ratio_min, quick_ratio_max=quick_ratio_max,
            beta_min=beta_min, beta_max=beta_max,
            eps_growth_min=eps_growth_min, revenue_growth_min=revenue_growth_min,
            perf_period=perf_period, perf_min=perf_min, perf_max=perf_max,
            rsi_min=rsi_min, rsi_max=rsi_max,
            above_sma20=above_sma20, above_sma50=above_sma50, above_sma200=above_sma200,
            macd_bullish=macd_bullish, bb_position=bb_position,
        )

        progress_container = st.container()
        with progress_container:
            st.markdown(results_bar(message="Starting screener scan..."), unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()

        def _on_progress(pct, msg):
            progress_bar.progress(min(max(pct, 0.0), 1.0))
            status_text.text(msg)

        filtered_df, full_df, debug = run_screener_pipeline(
            markets=markets,
            universe_size=universe_size,
            filter_kwargs=filter_kwargs,
            progress_callback=_on_progress,
        )

        progress_bar.empty()
        status_text.empty()
        st.session_state.last_filter_debug = debug

        with st.expander("Filter debug", expanded=filtered_df.empty):
            st.write(f"**Total symbols loaded:** {debug['symbols_loaded']}")
            st.write(f"**Symbols scanned:** {debug['symbols_scanned']}")
            st.write(f"**Price data fetched:** {debug['price_fetched']}")
            st.write(f"**Fundamentals fetched:** {debug['fundamentals_fetched']}")
            st.write(f"**Fetch failures:** {debug['fetch_failures']}")
            st.write(f"**Final matches:** {debug['matched']}")
            st.markdown("**Removed by active filter:**")
            filter_rows = [{"Filter": k, "Removed": v} for k, v in debug.get("filter_rejections", {}).items() if v]
            if filter_rows:
                st.dataframe(pd.DataFrame(filter_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No filter rejections (aside from fetch failures).")
            if debug.get("excluded_samples"):
                st.markdown("**First excluded tickers:**")
                st.dataframe(pd.DataFrame(debug["excluded_samples"]), hide_index=True, use_container_width=True)

        if not filtered_df.empty:
            df = filtered_df.copy()

            if "MarketCap" in df.columns:
                df = df.sort_values("MarketCap", ascending=False, na_position='last')

            st.session_state.screened_stocks = df
        
            avg_pe = df["PE"].mean() if "PE" in df.columns and df["PE"].notna().any() else None
            avg_perf = df["Year"].mean() if "Year" in df.columns and df["Year"].notna().any() else None
            avg_sharpe = df["Sharpe"].mean() if "Sharpe" in df.columns and df["Sharpe"].notna().any() else None
            total_mc = df["MarketCap"].sum() if "MarketCap" in df.columns and df["MarketCap"].notna().any() else None

            st.markdown(results_bar(count=len(df)), unsafe_allow_html=True)
            st.markdown(kpi_strip([
                ("Matches", str(len(df))),
                ("Avg P/E", f"{avg_pe:.2f}" if avg_pe else "-"),
                ("Avg 1Y", format_pct(avg_perf) if avg_perf is not None else "-", delta_class(avg_perf)),
                ("Avg Sharpe", f"{avg_sharpe:.2f}" if avg_sharpe else "-"),
                ("Total Cap", format_market_cap(total_mc) if total_mc else "-"),
            ]), unsafe_allow_html=True)

            st.markdown('<div class="results-zone">', unsafe_allow_html=True)
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Overview", "Performance", "Sectors", "Correlation", "Charts", "Watchlist"
            ])
        
            with tab1:
                sort_col1, sort_col2, sort_col3, export_col1, export_col2 = st.columns([2, 1, 1, 1, 1])
                with sort_col1:
                    sort_col = st.selectbox("Order by", [
                        "MarketCap", "Price", "PE", "PB", "PS", "PEG", "EV_EBITDA",
                        "ROE", "ROA", "ProfitMargin", "OperatingMargin",
                        "DebtToEquity", "CurrentRatio", "QuickRatio",
                        "Year", "RSI", "Sharpe"
                    ], label_visibility="collapsed")
                with sort_col2:
                    sort_asc = st.checkbox("Asc", value=False)
                df_sorted = df.sort_values(sort_col, ascending=sort_asc, na_position='last')
            
                df_display = format_dataframe(df_sorted)
            
                display_cols = ["Ticker", "Price", "MarketCap", 
                              # Price Ratios
                              "PE", "PB", "PS", "PEG", "EV_EBITDA",
                              # Profitability
                              "ROE", "ROA", "ProfitMargin", "OperatingMargin",
                              # Leverage
                              "DebtToEquity", "DebtToAssets", "InterestCoverage",
                              # Liquidity
                              "CurrentRatio", "QuickRatio",
                              # Other
                              "DividendYield", "Beta", "EPSGrowth", "RevenueGrowth",
                              "RSI", "Sharpe", "Volatility",
                              "Week", "Month", "Year", "Sector"]
                available_cols = [col for col in display_cols if col in df_display.columns]
                df_display = df_display[available_cols]
            
                st.dataframe(df_display, use_container_width=True, height=520, hide_index=True)
            
                with export_col1:
                    csv = df.to_csv(index=False)
                    st.download_button("Export CSV", csv, f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
                with export_col2:
                    try:
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Screener Results')
                        excel_data = output.getvalue()
                        st.download_button("Export XLS", excel_data, f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    except Exception:
                        pass
            
                wl_col1, wl_col2 = st.columns([4, 1])
                with wl_col1:
                    selected_tickers = st.multiselect("Add to watchlist", df_sorted["Ticker"].tolist(), label_visibility="collapsed")
                with wl_col2:
                    if st.button("Add", key="add_watchlist"):
                        for ticker in selected_tickers:
                            if ticker not in st.session_state.watchlist:
                                st.session_state.watchlist.append(ticker)
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
                        color_continuous_scale=[
                            [0.0, THEME["negative"]],
                            [0.5, THEME["neutral"]],
                            [1.0, THEME["positive"]],
                        ],
                        aspect="auto"
                    )
                    apply_dark_plotly_layout(fig, height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
                # Performance comparison
                if len(perf_df.columns) > 0:
                    fig = px.bar(perf_df.T, title="Performance Comparison", labels={"value": "Return (%)", "index": "Period"})
                    apply_dark_plotly_layout(fig, height=600)
                    fig.update_layout(showlegend=True)
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
                
                    st.markdown('<div class="section-header">Sector Breakdown</div>', unsafe_allow_html=True)
                    st.dataframe(sector_stats, use_container_width=True)
                
                    # Sector performance chart
                    fig = px.bar(
                        sector_stats.reset_index(),
                        x="Sector",
                        y="Avg Return (%)",
                        title="Average Performance by Sector",
                        color="Avg Return (%)",
                        color_continuous_scale=[
                            [0.0, THEME["negative"]],
                            [0.5, THEME["neutral"]],
                            [1.0, THEME["positive"]],
                        ],
                    )
                    apply_dark_plotly_layout(fig, height=500)
                    st.plotly_chart(fig, use_container_width=True)
        
            with tab4:
                # Correlation matrix
                numeric_cols = ["Price", "PE", "PB", "PS", "PEG", "EV_EBITDA",
                              "ROE", "ROA", "ProfitMargin", "OperatingMargin",
                              "DebtToEquity", "CurrentRatio", "QuickRatio",
                              "Beta", "RSI", "Year", "Month", "Week", "Sharpe"]
                corr_cols = [col for col in numeric_cols if col in df.columns]
                if len(corr_cols) > 1:
                    corr_df = df[corr_cols].corr()
                
                    fig = px.imshow(
                        corr_df,
                        labels=dict(color="Correlation"),
                        title="Correlation Matrix",
                        color_continuous_scale=[
                            [0.0, THEME["negative"]],
                            [0.5, THEME["bg_surface_muted"]],
                            [1.0, THEME["accent_blue"]],
                        ],
                        aspect="auto",
                        text_auto=True
                    )
                    apply_dark_plotly_layout(fig, height=700)
                    st.plotly_chart(fig, use_container_width=True)
        
            with tab5:
                st.markdown('<div class="section-header">Stock Detail Charts</div>', unsafe_allow_html=True)
            
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
                                st.markdown(f'<div class="fv-kpi"><div class="fv-kpi-label">Price</div><div class="fv-kpi-val">${current_price:.2f}</div></div>', unsafe_allow_html=True)
                            with col2:
                                chg_cls = delta_class(price_change_pct)
                                st.markdown(f'<div class="fv-kpi"><div class="fv-kpi-label">Change</div><div class="fv-kpi-val {chg_cls}">{format_pct(price_change_pct)}</div></div>', unsafe_allow_html=True)
                            with col3:
                                st.markdown(f'<div class="fv-kpi"><div class="fv-kpi-label">High</div><div class="fv-kpi-val">${hist_data["High"].max():.2f}</div></div>', unsafe_allow_html=True)
                            with col4:
                                st.markdown(f'<div class="fv-kpi"><div class="fv-kpi-label">Low</div><div class="fv-kpi-val">${hist_data["Low"].min():.2f}</div></div>', unsafe_allow_html=True)
                            with col5:
                                st.markdown(f'<div class="fv-kpi"><div class="fv-kpi-label">Volume</div><div class="fv-kpi-val">{hist_data["Volume"].iloc[-1]:,.0f}</div></div>', unsafe_allow_html=True)
                        
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
                st.markdown('<div class="section-header">Watchlist</div>', unsafe_allow_html=True)
            
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

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No stocks match your criteria. Try adjusting your filters or check Filter debug above.")

    else:
        st.markdown(
            '<div class="fv-empty">Configure filters above and click <strong>▼ Filter</strong> to load results.</div>',
            unsafe_allow_html=True,
        )

# =============================
# FOOTER
# =============================
st.markdown(
    '<div class="fv-footer">Stock Screener · Powered by Yahoo Finance · For educational purposes only · Not financial advice</div>',
    unsafe_allow_html=True,
)
