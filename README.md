# 📈 AC Stocks Screener

A modern, professional stock screening platform inspired by Finviz, covering both **US and Canadian markets**.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.40-red.svg)

---

##  Features

###  Comprehensive Screening
- **Market Cap Filters**: From micro-cap to mega-cap stocks
- **Price Ranges**: Custom minimum and maximum price filters
- **Volume Filters**: Screen by average trading volume
- **Performance Tracking**: Week, month, quarter, half-year, and annual returns

###  Fundamental Analysis
- **Valuation Metrics**: P/E, P/S, P/B, PEG ratios
- **Profitability**: Profit margins, ROE, ROA
- **Financial Health**: Debt-to-equity, current ratio
- **Dividends**: Yield and payout information

###  Multi-Market Coverage
- **80+ US Stocks**: Major companies across all sectors
- **45+ Canadian Stocks**: TSX-listed companies
- **10+ Sectors**: Technology, Healthcare, Finance, Consumer, Industrial, Energy, Materials, Utilities, Real Estate, and more

###  Modern, Clean UI
- Gradient header design
- Professional color scheme (purple/blue theme)
- Responsive table layout
- Interactive filters
- Real-time data updates

---

##  Quick Start

### Local Deployment

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/acstocksscreener.git
cd acstocksscreener
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```

4. **Open in browser**
Navigate to `http://localhost:8501`

---

### Deploy to Streamlit Cloud

1. **Create GitHub Repository**
   - Go to [github.com](https://github.com)
   - Create new repository named `acstocksscreener`
   - Make it public

2. **Upload Files**
   - Upload `app.py`
   - Upload `requirements.txt`

3. **Deploy on Streamlit**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file to `app.py`
   - Click "Deploy!"

4. **Access Your Live Site**
   - Your URL: `https://acstocksscreener.streamlit.app`

---

## 📖 How to Use

### 1. Select Markets
Choose between US Stocks and Canadian (TSX) markets, or both.

### 2. Set Filters

**Market Cap:**
- Mega (>$200B)
- Large ($10B-$200B)
- Mid ($2B-$10B)
- Small ($300M-$2B)
- Micro (<$300M)

**Price:**
- Set minimum and maximum price range

**Volume:**
- Filter by average daily trading volume

**Performance:**
- Screen by weekly, monthly, quarterly, or annual performance

**Fundamentals:**
- Maximum P/E ratio
- Minimum profit margin
- Minimum ROE

**Sector:**
- Filter by specific industries

### 3. Run Screener
Click the **"🚀 RUN SCREENER"** button to analyze stocks.

### 4. View Results
- See matching stocks in a clean table format
- Sort by any column
- View key metrics at a glance

### 5. Export Data
Download results as CSV for further analysis.

---

##  Available Metrics

| Metric | Description |
|--------|-------------|
| **Price** | Current stock price |
| **Change %** | Daily price change percentage |
| **Volume** | Average trading volume |
| **Market Cap** | Total market capitalization |
| **P/E Ratio** | Price-to-earnings ratio |
| **EPS** | Earnings per share |
| **Dividend %** | Dividend yield |
| **Profit Margin** | Net profit margin |
| **ROE** | Return on equity |
| **Performance** | Returns over various periods |

---

##  Use Cases

### For Long-Term Investors
- Find undervalued stocks with strong fundamentals
- Screen for dividend-paying stocks
- Identify growth stocks with consistent performance

### For Value Investors
- Low P/E ratios
- High profit margins
- Strong balance sheets

### For Growth Investors
- High ROE
- Strong year-over-year performance
- Expanding market cap

### For Dividend Investors
- High dividend yields
- Consistent payout history
- Financial stability

---

##  Technical Details

### Built With
- **Python 3.8+**
- **Streamlit** - Web framework
- **yfinance** - Market data API
- **Pandas** - Data manipulation

### Data Source
- **Yahoo Finance API** via yfinance library
- Real-time stock quotes
- Historical performance data
- Fundamental metrics

### Update Frequency
- Prices: Real-time (delayed 15 minutes)
- Fundamentals: Updated quarterly
- Performance: Calculated from historical data

---

## ⚙️ Customization

### Adding More Stocks

Edit the `get_stock_universe()` function in `app.py`:

```python
# Add to US stocks list
us_stocks = [
    'AAPL', 'MSFT', 'YOUR_SYMBOL_HERE'
]

# Add to Canadian stocks list
tsx_stocks = [
    'RY.TO', 'YOUR_SYMBOL.TO'
]
```

### Changing Color Scheme

Modify the CSS in the `st.markdown()` section at the top of `app.py`:

```css
.header-container {
    background: linear-gradient(135deg, #YOUR_COLOR1, #YOUR_COLOR2);
}
```

---

##  Troubleshooting

### Issue: No stocks showing
**Solution:** Lower your filter criteria. Try setting:
- Market Cap: Any
- P/E: 100
- Profit Margin: -50
- ROE: -50

### Issue: Slow performance
**Solution:** 
- Reduce number of stocks in universe
- Use caching (already implemented)
- Deploy to Streamlit Cloud for better resources

### Issue: Data not loading
**Solution:**
- Check internet connection
- Yahoo Finance API may be rate-limiting
- Wait a few minutes and try again

---

##  Limitations

1. **Data Accuracy**: Relies on Yahoo Finance API
2. **Real-time Data**: 15-minute delay on free tier
3. **Coverage**: Limited to stocks with Yahoo Finance data
4. **Rate Limits**: May encounter API limits with frequent use

---

##  Disclaimer

**Important:** This tool is for informational and educational purposes only.

- Not investment advice
- Always do your own research
- Consult financial professionals before investing
- Past performance does not guarantee future results
- Stock investing carries risk of loss

---

##  License

MIT License - feel free to use and modify for your own projects.

---

##  Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Check yfinance docs: [github.com/ranaroussi/yfinance](https://github.com/ranaroussi/yfinance)

---

## 🎉 Acknowledgments

- **Finviz** - Inspiration for design and features
- **Yahoo Finance** - Market data
- **Streamlit** - Amazing web framework
- **Python Community** - Open source libraries

---

<div align="center">
    <p><strong>AC Stocks Screener</strong></p>
    <p>Built with ❤️ for the investing community</p>
</div>
