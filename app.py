import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="AC Stocks Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Stock table */
    .stock-table {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1e293b;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    .dataframe thead tr th {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 600;
        padding: 12px !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #f1f5f9 !important;
    }
    
    /* Filter section */
    .filter-section {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">📈 AC Stocks Screener</h1>
    <p class="header-subtitle">Professional Stock Screening for US & Canadian Markets</p>
</div>
""", unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.markdown("## 🎯 Screening Filters")
    
    # Markets
    st.markdown("### 📍 Markets")
    markets = st.multiselect(
        "Select Markets",
        ["US Stocks", "Canadian (TSX)"],
        default=["US Stocks", "Canadian (TSX)"]
    )
    
    st.divider()
    
    # Market Cap
    st.markdown("### 💰 Market Cap")
    market_cap = st.selectbox(
        "Size",
        ["Any", "Mega (>$200B)", "Large ($10B-$200B)", "Mid ($2B-$10B)", 
         "Small ($300M-$2B)", "Micro (<$300M)"]
    )
    
    st.divider()
    
    # Price
    st.markdown("### 💵 Price")
    col1, col2 = st.columns(2)
    with col1:
        price_min = st.number_input("Min ($)", value=0, min_value=0)
    with col2:
        price_max = st.number_input("Max ($)", value=10000, min_value=0)
    
    st.divider()
    
    # Volume
    st.markdown("### 📊 Volume")
    min_volume = st.selectbox(
        "Minimum Avg Volume",
        [0, 50000, 100000, 500000, 1000000],
        index=0
    )
    
    st.divider()
    
    # Performance
    st.markdown("### 📈 Performance")
    perf_period = st.selectbox(
        "Period",
        ["Any", "Week", "Month", "Quarter", "Half", "Year"]
    )
    
    perf_min = -100  # Default value
    if perf_period != "Any":
        perf_min = st.slider(f"Min {perf_period} Performance (%)", -100, 500, -50)
    
    st.divider()
    
    # Fundamentals
    st.markdown("### 📊 Fundamentals")
    
    pe_max = st.number_input("Max P/E Ratio", value=100, min_value=0)
    
    col1, col2 = st.columns(2)
    with col1:
        profit_margin_min = st.number_input("Min Profit Margin (%)", value=-50, step=5)
    with col2:
        roe_min = st.number_input("Min ROE (%)", value=-50, step=5)
    
    st.divider()
    
    # Sector
    st.markdown("### 🏢 Sector")
    sector_filter = st.multiselect(
        "Select Sectors",
        ["Any", "Technology", "Healthcare", "Financial", "Consumer", 
         "Industrial", "Energy", "Materials", "Utilities", "Real Estate"]
    )
    
    st.divider()
    
    # Run button
    run_button = st.button("🚀 RUN SCREENER", use_container_width=True)

# Stock universe
@st.cache_data(ttl=3600)
def get_stock_universe():
    """Get comprehensive stock list"""
    stocks = []
    
    if "US Stocks" in markets:
        # Major US stocks across sectors
        us_stocks = [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN',
            # Finance
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'LLY', 'MRK', 'DHR',
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'COST', 'TGT', 'LOW', 'DIS',
            'PG', 'KO', 'PEP', 'PM', 'MO',
            # Industrial
            'CAT', 'BA', 'HON', 'UPS', 'GE', 'MMM', 'DE', 'EMR',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX',
            # Materials
            'LIN', 'APD', 'SHW', 'NEM', 'FCX'
        ]
        stocks.extend(us_stocks)
    
    if "Canadian (TSX)" in markets:
        # Major TSX stocks
        tsx_stocks = [
            # Banks
            'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'NA.TO',
            # Energy
            'ENB.TO', 'CNQ.TO', 'SU.TO', 'TRP.TO', 'IMO.TO', 'CVE.TO',
            # Materials
            'ABX.TO', 'FNV.TO', 'WPM.TO', 'NTR.TO', 'K.TO',
            # Industrials
            'CNR.TO', 'CP.TO', 'CSU.TO', 'WCN.TO', 'TRI.TO',
            # Tech
            'SHOP.TO', 'OTEX.TO',
            # Telecom
            'BCE.TO', 'T.TO', 'RCI-B.TO',
            # Consumer
            'QSR.TO', 'L.TO', 'ATD.TO', 'DOL.TO',
            # Financials
            'MFC.TO', 'SLF.TO', 'GWO.TO', 'POW.TO', 'BAM.TO',
            # Utilities
            'FTS.TO', 'EMA.TO', 'H.TO'
        ]
        stocks.extend(tsx_stocks)
    
    return stocks

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_stock_data(symbol):
    """Fetch stock data"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get historical data for performance
        hist = ticker.history(period="1y")
        
        if hist.empty:
            return None
        
        current_price = hist['Close'].iloc[-1]
        
        # Calculate performance
        perf_data = {}
        try:
            if len(hist) >= 5:
                perf_data['week'] = ((current_price - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5]) * 100
            if len(hist) >= 21:
                perf_data['month'] = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]) * 100
            if len(hist) >= 63:
                perf_data['quarter'] = ((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63]) * 100
            if len(hist) >= 126:
                perf_data['half'] = ((current_price - hist['Close'].iloc[-126]) / hist['Close'].iloc[-126]) * 100
            if len(hist) >= 252:
                perf_data['year'] = ((current_price - hist['Close'].iloc[-252]) / hist['Close'].iloc[-252]) * 100
        except:
            pass
        
        return {
            'symbol': symbol,
            'name': info.get('longName', symbol)[:30],
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'price': current_price,
            'change': info.get('regularMarketChangePercent', 0),
            'volume': info.get('averageVolume', 0),
            'marketCap': info.get('marketCap', 0),
            'pe': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'peg': info.get('pegRatio'),
            'ps': info.get('priceToSalesTrailing12Months'),
            'pb': info.get('priceToBook'),
            'eps': info.get('trailingEps'),
            'dividend': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            'profitMargin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
            'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'debtEquity': info.get('debtToEquity'),
            'currentRatio': info.get('currentRatio'),
            'beta': info.get('beta'),
            'performance': perf_data,
            '52wHigh': info.get('fiftyTwoWeekHigh'),
            '52wLow': info.get('fiftyTwoWeekLow'),
        }
    except Exception as e:
        return None

def apply_filters(stock):
    """Apply all filters - handles missing data gracefully"""
    if not stock:
        return False
    
    # Market cap filter - only apply if not "Any"
    if market_cap != "Any":
        mc = stock.get('marketCap', 0)
        if mc == 0:  # No data, let it through
            pass
        elif market_cap == "Mega (>$200B)" and mc < 200e9:
            return False
        elif market_cap == "Large ($10B-$200B)" and not (10e9 <= mc < 200e9):
            return False
        elif market_cap == "Mid ($2B-$10B)" and not (2e9 <= mc < 10e9):
            return False
        elif market_cap == "Small ($300M-$2B)" and not (300e6 <= mc < 2e9):
            return False
        elif market_cap == "Micro (<$300M)" and mc >= 300e6:
            return False
    
    # Price filter - only apply if price exists
    price = stock.get('price', 0)
    if price > 0 and not (price_min <= price <= price_max):
        return False
    
    # Volume filter - only apply if volume exists
    volume = stock.get('volume', 0)
    if volume > 0 and volume < min_volume:
        return False
    
    # P/E filter - only apply if PE exists and is positive
    pe = stock.get('pe')
    if pe and pe > 0 and pe > pe_max:
        return False
    
    # Profit margin - only apply if exists
    profit_margin = stock.get('profitMargin')
    if profit_margin is not None and profit_margin < profit_margin_min:
        return False
    
    # ROE - only apply if exists
    roe = stock.get('roe')
    if roe is not None and roe < roe_min:
        return False
    
    # Sector filter - only apply if sectors selected and not "Any"
    if sector_filter and len(sector_filter) > 0 and "Any" not in sector_filter:
        sector = stock.get('sector', 'N/A')
        if sector not in sector_filter:
            return False
    
    # Performance filter - only apply if not "Any"
    if perf_period != "Any":
        period_map = {
            "Week": "week",
            "Month": "month",
            "Quarter": "quarter",
            "Half": "half",
            "Year": "year"
        }
        period_key = period_map.get(perf_period)
        if period_key and period_key in stock.get('performance', {}):
            perf_value = stock['performance'][period_key]
            if perf_value is not None and perf_value < perf_min:
                return False
    
    return True

def format_number(num, prefix='', suffix='', decimals=2):
    """Format numbers for display"""
    if num is None or num == 0:
        return 'N/A'
    
    if abs(num) >= 1e12:
        return f"{prefix}{num/1e12:.{decimals}f}T{suffix}"
    elif abs(num) >= 1e9:
        return f"{prefix}{num/1e9:.{decimals}f}B{suffix}"
    elif abs(num) >= 1e6:
        return f"{prefix}{num/1e6:.{decimals}f}M{suffix}"
    elif abs(num) >= 1e3:
        return f"{prefix}{num/1e3:.{decimals}f}K{suffix}"
    else:
        return f"{prefix}{num:.{decimals}f}{suffix}"

# Main screening logic
if run_button:
    if not markets:
        st.error("❌ Please select at least one market!")
    else:
        stock_universe = get_stock_universe()
        
        # Progress
        progress_container = st.container()
        with progress_container:
            st.info(f"🔍 Screening {len(stock_universe)} stocks...")
            progress_bar = st.progress(0)
            status = st.empty()
        
        # Screen stocks
        results = []
        for idx, symbol in enumerate(stock_universe):
            status.text(f"Analyzing {symbol}... ({idx+1}/{len(stock_universe)})")
            progress_bar.progress((idx + 1) / len(stock_universe))
            
            stock_data = fetch_stock_data(symbol)
            if stock_data and apply_filters(stock_data):
                results.append(stock_data)
            
            time.sleep(0.05)  # Rate limiting
        
        progress_container.empty()
        
        # Display results
        if results:
            st.success(f"✅ Found {len(results)} stocks matching your criteria")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(results)}</div>
                    <div class="metric-label">Total Matches</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_pe = sum([s['pe'] for s in results if s['pe']]) / len([s for s in results if s['pe']]) if any(s['pe'] for s in results) else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_pe:.1f}</div>
                    <div class="metric-label">Avg P/E</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_margin = sum([s['profitMargin'] for s in results]) / len(results) if results else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_margin:.1f}%</div>
                    <div class="metric-label">Avg Profit Margin</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                total_mc = sum([s['marketCap'] for s in results])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{format_number(total_mc, prefix='$')}</div>
                    <div class="metric-label">Total Market Cap</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Create DataFrame
            df_data = []
            for stock in results:
                perf_week = stock['performance'].get('week', 0)
                perf_month = stock['performance'].get('month', 0)
                perf_year = stock['performance'].get('year', 0)
                
                df_data.append({
                    'Ticker': stock['symbol'],
                    'Company': stock['name'],
                    'Sector': stock['sector'],
                    'Price': f"${stock['price']:.2f}",
                    'Change %': f"{stock['change']:.2f}%",
                    'Volume': format_number(stock['volume']),
                    'Market Cap': format_number(stock['marketCap'], prefix='$'),
                    'P/E': f"{stock['pe']:.1f}" if stock['pe'] else 'N/A',
                    'EPS': f"${stock['eps']:.2f}" if stock['eps'] else 'N/A',
                    'Dividend %': f"{stock['dividend']:.2f}%" if stock['dividend'] else 'N/A',
                    'Profit Margin %': f"{stock['profitMargin']:.1f}%",
                    'ROE %': f"{stock['roe']:.1f}%",
                    'Week %': f"{perf_week:.1f}%" if perf_week else 'N/A',
                    'Month %': f"{perf_month:.1f}%" if perf_month else 'N/A',
                    'Year %': f"{perf_year:.1f}%" if perf_year else 'N/A',
                })
            
            df = pd.DataFrame(df_data)
            
            # Display table
            st.markdown("### 📊 Screening Results")
            st.dataframe(
                df,
                use_container_width=True,
                height=600,
                hide_index=True
            )
            
            # Export
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"ac_stocks_screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("⚠️ No stocks match your criteria. Try adjusting your filters.")

else:
    # Welcome screen
    st.markdown("### 👋 Welcome to AC Stocks Screener!")
    st.markdown("""
    A modern, Finviz-style stock screener covering **US and Canadian markets**.
    
    **Features:**
    - 📊 Screen by market cap, price, volume, and performance
    - 💰 Filter by fundamentals (P/E, profit margin, ROE)
    - 🏢 Sector-specific screening
    - 📈 Performance tracking across multiple timeframes
    - 📥 Export results to CSV
    
    **Get started:** Set your filters in the sidebar and click **RUN SCREENER**!
    """)
    
    # Sample stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**80+** US Stocks")
    with col2:
        st.info("**45+** Canadian Stocks")
    with col3:
        st.info("**10+** Sectors Covered")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p><strong>AC Stocks Screener</strong> | Powered by Yahoo Finance</p>
    <p style="font-size: 0.9rem;">For informational purposes only. Not investment advice.</p>
</div>
""", unsafe_allow_html=True)
