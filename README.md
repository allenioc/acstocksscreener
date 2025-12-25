# Stock Screener 

An enhanced stock screener built with Streamlit, inspired by Finviz's powerful screening capabilities.

##  Enhanced Features

### New Additions vs Original:

1. **Advanced Fundamental Filters**
   - P/E, P/B, P/S ratio filters with ranges
   - Beta filtering
   - EPS Growth (YoY) filtering
   - Enhanced dividend yield filtering

2. **Technical Indicators**
   - RSI (Relative Strength Index) calculation and filtering
   - Moving Average filters (SMA 20, 50, 200)
   - Price position relative to moving averages

3. **Extended Performance Metrics**
   - Multiple timeframes: 1 Week, 1 Month, 3 Months, 6 Months, 1 Year
   - Performance range filtering (min/max)
   - Enhanced performance calculations

4. **Data Visualization**
   - Interactive performance comparison charts (Plotly)
   - Fundamentals scatter plots (P/E vs P/B, sized by P/S)
   - Tabbed interface for different views

5. **Better Data Display**
   - Formatted market cap (B, M, T notation)
   - Summary metrics dashboard
   - Better column ordering and formatting
   - Enhanced CSV export

6. **Improved User Experience**
   - Expanded stock universe (more tickers)
   - Better progress tracking
   - Welcome screen with features overview
   - More organized sidebar filters with sections

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📋 Requirements

- streamlit>=1.28.0
- yfinance>=0.2.28
- pandas>=2.0.0
- numpy>=1.24.0
- plotly>=5.17.0

## 🎯 Usage

```bash
streamlit run stock_screener_enhanced.py
```

## 📊 Features Breakdown

### Filter Categories

1. **Market Selection**: US and/or Canada markets
2. **Market Cap**: Mega, Large, Mid, Small, Micro caps
3. **Price Filters**: Price range with slider
4. **Volume**: Minimum average volume filtering
5. **Fundamentals**: P/E, P/B, P/S, Beta, EPS Growth, Dividend Yield
6. **Performance**: Multiple timeframes with range filtering
7. **Technical**: RSI, Moving Average positions
8. **Sectors**: Industry sector filtering

### Data Views

- **Table View**: Comprehensive data table with all metrics
- **Performance Chart**: Bar chart comparing performance across timeframes
- **Fundamentals Chart**: Scatter plot visualization of valuation metrics

## 🔧 Technical Improvements

- Better error handling
- Caching for improved performance
- Rate limiting to avoid API issues
- More robust data fetching
- Enhanced data validation

## 📝 Notes

- Data is cached for 30 minutes (stocks) and 24 hours (universe)
- Rate limiting added to prevent API throttling
- Some metrics may be unavailable for certain stocks
- For educational purposes only - not financial advice

