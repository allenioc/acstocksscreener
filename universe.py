"""Stock universe definitions shared across app sections."""

import streamlit as st


@st.cache_data(ttl=86400)
def get_stock_universe(markets):
    stocks = []

    if "US" in markets:
        stocks += ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MU']
        stocks += ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'PYPL', 'SCHW', 'BLK', 'COF']
        stocks += ['JNJ', 'UNH', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'DHR', 'CVS', 'CI', 'HUM', 'ELV']
        stocks += ['WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'PG', 'KO', 'PEP', 'CL', 'UL', 'DIS', 'CMCSA']
        stocks += ['CAT', 'GE', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'ETN', 'EMR']
        stocks += ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO']
        stocks += ['ADBE', 'CRM', 'ORCL', 'IBM', 'CSCO', 'INTU', 'NOW', 'SNPS', 'CDNS', 'ANSS']
        stocks += ['VZ', 'T', 'TMUS', 'LUMN']
        stocks += ['NEE', 'DUK', 'SO', 'AEP']
        stocks += ['AMT', 'PLD', 'EQIX', 'SPG']
        stocks += ['LIN', 'APD', 'ECL', 'SHW']

    if "Canada" in markets:
        stocks += [
            'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'NA.TO',
            'ENB.TO', 'CNQ.TO', 'SU.TO', 'IMO.TO', 'TRP.TO',
            'SHOP.TO', 'OTEX.TO', 'WCN.TO', 'CGI.TO',
            'BCE.TO', 'T.TO', 'RCI-B.TO', 'QBR-B.TO',
            'CNR.TO', 'CP.TO',
            'ATD.TO', 'L.TO', 'MRU.TO',
        ]

    return list(set(stocks))
