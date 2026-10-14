"""Scalable stock screener backend using public symbol lists and yfinance."""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from filter_logic import build_filter_config, apply_screener_filters, any_filter_active

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BACKUP_PATH = os.path.join(DATA_DIR, "tickers_backup.csv")
CACHE_PATH = os.path.join(DATA_DIR, "stock_cache.csv")

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Canadian symbols kept separate for future expansion.
CANADIAN_TICKERS = [
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    "ENB.TO", "CNQ.TO", "SU.TO", "IMO.TO", "TRP.TO",
    "SHOP.TO", "OTEX.TO", "WCN.TO", "CGI.TO",
    "BCE.TO", "T.TO", "RCI-B.TO", "QBR-B.TO",
    "CNR.TO", "CP.TO", "ATD.TO", "L.TO", "MRU.TO",
    "ABX.TO", "FNV.TO", "WPM.TO", "NTR.TO",
    "MFC.TO", "SLF.TO", "GWO.TO", "POW.TO", "BAM.TO",
]

EXCLUDE_SYMBOL_PATTERN = re.compile(r"[\^=\$\+]")
# Nasdaq Trader dotted suffixes for units, warrants and rights (class shares like .A/.B are kept).
NON_SHARE_SUFFIXES = {"U", "W", "WS", "WT", "R", "RT"}
EXCLUDE_NAME_PATTERN = re.compile(
    r"\b(warrants?|rights?|preferred|units?|notes due|debentures?)\b", re.IGNORECASE
)
NON_US_SUFFIXES = (".TO", ".V", ".NE", ".CN")

UNIVERSE_SIZE_OPTIONS = {
    "Top 100": 100,
    "Top 500": 500,
    "Top 1000": 1000,
    "Full universe": None,
}

# ~15 months of calendar days gives >252 trading sessions for 1Y return and the 200-day SMA.
PRICE_LOOKBACK_DAYS = 460
RISK_FREE_RATE = 0.04
MAX_LIVE_FUNDAMENTALS_FETCH = 1000
FUNDAMENTAL_WORKERS = 6
FUNDAMENTALS_CACHE_MAX_AGE_DAYS = 7

FUNDAMENTAL_NUMERIC_FIELDS = [
    "market_cap", "pe", "forward_pe", "pb", "ps", "ev_ebitda", "peg", "dividend_yield",
    "beta", "roe", "roa", "profit_margin", "operating_margin", "debt_to_equity",
    "current_ratio", "quick_ratio", "revenue_growth", "earnings_growth",
    "fifty_two_week_high", "fifty_two_week_low",
]
FUNDAMENTAL_TEXT_FIELDS = ["company_name", "exchange", "country", "sector", "industry"]

# Successful fundamentals for this server process; failures are never stored.
_FUNDAMENTALS_MEMORY: dict[str, dict] = {}


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if v in {"", "-", "N/A", "nan", "None"}:
            return None
        try:
            value = float(v.replace(",", ""))
        except ValueError:
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def normalize_ticker(symbol) -> str:
    """Uppercase, strip, and convert US class-share dots to Yahoo's dash format (BRK.B -> BRK-B)."""
    s = str(symbol).strip().upper()
    if s.endswith(NON_US_SUFFIXES):
        return s
    return s.replace(".", "-")


def _is_common_stock_symbol(symbol) -> bool:
    if symbol is None or (isinstance(symbol, float) and np.isnan(symbol)):
        return False
    symbol = str(symbol).strip()
    if not symbol or symbol == "nan":
        return False
    if EXCLUDE_SYMBOL_PATTERN.search(symbol):
        return False
    if "." in symbol and symbol.rsplit(".", 1)[1].upper() in NON_SHARE_SUFFIXES:
        return False
    if len(symbol) > 7:
        return False
    return symbol.replace(".", "").isalnum()


def _is_excluded_security_name(name) -> bool:
    text = str(name or "")
    if "common units" in text.lower():
        return False
    return bool(EXCLUDE_NAME_PATTERN.search(text))


def load_us_symbols_from_nasdaq() -> pd.DataFrame:
    """Load US common-stock symbols from Nasdaq Trader public files."""
    frames = []

    try:
        nasdaq_resp = requests.get(NASDAQ_LISTED_URL, timeout=20)
        nasdaq_resp.raise_for_status()
        nasdaq_df = pd.read_csv(StringIO(nasdaq_resp.text), sep="|", keep_default_na=False)
        nasdaq_df = nasdaq_df.rename(columns={
            "Symbol": "ticker",
            "Security Name": "company_name",
        })
        nasdaq_df["exchange"] = "NASDAQ"
        nasdaq_df["country"] = "US"
        frames.append(nasdaq_df[["ticker", "company_name", "exchange", "country", "ETF", "Test Issue"]])
    except Exception:
        pass

    try:
        other_resp = requests.get(OTHER_LISTED_URL, timeout=20)
        other_resp.raise_for_status()
        other_df = pd.read_csv(StringIO(other_resp.text), sep="|", keep_default_na=False)
        other_df = other_df.rename(columns={
            "ACT Symbol": "ticker",
            "Security Name": "company_name",
            "Exchange": "exchange",
        })
        other_df["country"] = "US"
        frames.append(other_df[["ticker", "company_name", "exchange", "country", "ETF", "Test Issue"]])
    except Exception:
        pass

    if not frames:
        return pd.DataFrame(columns=["ticker", "company_name", "exchange", "country", "priority"])

    df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[df["ticker"] != ""]

    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].astype(str).str.upper() != "Y"]
    if "ETF" in df.columns:
        df = df[df["ETF"].astype(str).str.upper() != "Y"]

    df = df[df["ticker"].apply(_is_common_stock_symbol)]
    df = df[~df["company_name"].apply(_is_excluded_security_name)]
    df["ticker"] = df["ticker"].apply(normalize_ticker)
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    df["priority"] = 5000
    return df[["ticker", "company_name", "exchange", "country", "priority"]]


def load_local_symbols_backup() -> pd.DataFrame:
    """Load curated fallback symbol list."""
    if not os.path.exists(BACKUP_PATH):
        return pd.DataFrame(columns=["ticker", "company_name", "exchange", "country", "priority"])
    df = pd.read_csv(BACKUP_PATH, keep_default_na=False, na_values={"priority": [""]})
    df = df[df["ticker"].apply(_is_common_stock_symbol)]
    df["ticker"] = df["ticker"].apply(normalize_ticker)
    if "priority" not in df.columns:
        df["priority"] = 9999
    return df.drop_duplicates(subset=["ticker"], keep="first")


@st.cache_data(ttl=86400, show_spinner=False)
def load_symbol_universe(markets: tuple, universe_size: str) -> pd.DataFrame:
    """Load merged symbol universe with optional size limit."""
    frames = []

    if "US" in markets:
        us_df = load_us_symbols_from_nasdaq()
        if us_df.empty:
            us_df = load_local_symbols_backup()
            us_df = us_df[us_df["country"] == "US"] if "country" in us_df.columns else us_df
        else:
            backup = load_local_symbols_backup()
            if not backup.empty:
                priority_map = backup.set_index("ticker")["priority"].to_dict()
                us_df["priority"] = us_df["ticker"].map(priority_map).fillna(us_df["priority"])
        frames.append(us_df)

    if "Canada" in markets:
        ca_backup = load_local_symbols_backup()
        ca_rows = ca_backup[ca_backup["ticker"].isin(CANADIAN_TICKERS)] if not ca_backup.empty else pd.DataFrame()
        if ca_rows.empty:
            ca_rows = pd.DataFrame({
                "ticker": CANADIAN_TICKERS,
                "company_name": CANADIAN_TICKERS,
                "exchange": "TSX",
                "country": "CA",
                "priority": 3000,
            })
        frames.append(ca_rows)

    if not frames:
        return pd.DataFrame(columns=["ticker", "company_name", "exchange", "country", "priority"])

    universe = pd.concat(frames, ignore_index=True)
    universe["ticker"] = universe["ticker"].apply(normalize_ticker)
    universe = universe.drop_duplicates(subset=["ticker"], keep="first")
    universe = universe.sort_values(["priority", "ticker"], ascending=[True, True])

    limit = UNIVERSE_SIZE_OPTIONS.get(universe_size)
    if limit is not None:
        universe = universe.head(limit)

    return universe.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Price history and price-derived metrics
# ---------------------------------------------------------------------------

def _extract_ticker_frame(raw: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
    """Pull one ticker's OHLCV frame out of a yf.download result (single or multi-index)."""
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        frame = None
        for level in range(raw.columns.nlevels):
            if symbol in raw.columns.get_level_values(level):
                frame = raw.xs(symbol, axis=1, level=level)
                break
        if frame is None:
            return None
    else:
        frame = raw
    if "Close" not in frame.columns:
        return None
    frame = frame.dropna(subset=["Close"])
    return frame if not frame.empty else None


def _trailing_return(close: pd.Series, lookback: int) -> Optional[float]:
    """(Close[-1] / Close[-lookback]) - 1, as a percent."""
    if len(close) < lookback:
        return None
    base = float(close.iloc[-lookback])
    if base <= 0:
        return None
    return (float(close.iloc[-1]) / base - 1) * 100


def _last_or_none(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    return _safe_float(series.iloc[-1])


def compute_price_metrics(symbol: str, frame: pd.DataFrame) -> dict:
    """Quote fields, trailing returns, risk metrics and technicals from adjusted OHLCV history."""
    close = frame["Close"].astype(float)
    volume = frame["Volume"].astype(float) if "Volume" in frame.columns else None
    n = len(close)

    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if n >= 2 else None
    change_pct = ((price - prev) / prev * 100) if prev else None

    returns = close.pct_change().dropna().tail(252)
    volatility = float(returns.std() * np.sqrt(252)) if len(returns) >= 20 else None

    one_year = _trailing_return(close, 252)
    if one_year is not None:
        annual_return = one_year / 100
    elif n >= 60 and float(close.iloc[0]) > 0:
        annual_return = (price / float(close.iloc[0])) ** (252 / (n - 1)) - 1
    else:
        annual_return = None
    sharpe = None
    if annual_return is not None and volatility is not None and volatility > 0:
        sharpe = (annual_return - RISK_FREE_RATE) / volatility

    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = None
    if n > 14:
        gain, loss = _last_or_none(avg_gain), _last_or_none(avg_loss)
        if gain is not None and loss is not None:
            rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)

    sma20 = _last_or_none(close.rolling(20).mean()) if n >= 20 else None
    sma50 = _last_or_none(close.rolling(50).mean()) if n >= 50 else None
    sma200 = _last_or_none(close.rolling(200).mean()) if n >= 200 else None

    macd_bullish = None
    if n >= 35:
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_bullish = bool(macd.iloc[-1] > signal.iloc[-1])

    bb_position = None
    if n >= 20:
        std20 = _last_or_none(close.rolling(20).std())
        if sma20 is not None and std20 is not None:
            upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
            bb_position = "Above Upper" if price > upper else "Below Lower" if price < lower else "Between Bands"

    clean_vol = volume.dropna() if volume is not None else pd.Series(dtype=float)
    last_year = close.tail(252)

    return {
        "ticker": symbol,
        "price": price,
        "previous_close": prev,
        "open": _last_or_none(frame["Open"]) if "Open" in frame.columns else None,
        "day_high": _last_or_none(frame["High"]) if "High" in frame.columns else None,
        "day_low": _last_or_none(frame["Low"]) if "Low" in frame.columns else None,
        "volume": float(clean_vol.iloc[-1]) if not clean_vol.empty else None,
        "average_volume": float(clean_vol.tail(20).mean()) if not clean_vol.empty else None,
        "change_percent": change_pct,
        "week": _trailing_return(close, 5),
        "month": _trailing_return(close, 21),
        "quarter": _trailing_return(close, 63),
        "six_month": _trailing_return(close, 126),
        "year": one_year,
        "volatility": volatility * 100 if volatility is not None else None,
        "sharpe": sharpe,
        "rsi": rsi,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "macd_bullish": macd_bullish,
        "bb_position": bb_position,
        "hist_52w_high": float(last_year.max()),
        "hist_52w_low": float(last_year.min()),
    }


def _download_prices(tickers: list[str], start: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception:
        return rows
    for sym in tickers:
        try:
            frame = _extract_ticker_frame(raw, sym)
            if frame is not None:
                rows[sym] = compute_price_metrics(sym, frame)
        except Exception:
            continue
    return rows


def fetch_price_data_batch(tickers: list[str]) -> pd.DataFrame:
    """Batch download ~15 months of adjusted prices and compute price-derived metrics."""
    if not tickers:
        return pd.DataFrame()

    start = (date.today() - timedelta(days=PRICE_LOOKBACK_DAYS)).isoformat()
    rows: dict[str, dict] = {}
    chunk_size = 100
    for i in range(0, len(tickers), chunk_size):
        rows.update(_download_prices(tickers[i : i + chunk_size], start))

    missing = [sym for sym in tickers if sym not in rows]
    for i in range(0, len(missing), chunk_size):
        rows.update(_download_prices(missing[i : i + chunk_size], start))

    return pd.DataFrame(list(rows.values()))


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def _has_fundamentals(row: dict) -> bool:
    return any(_safe_float(row.get(field)) is not None for field in FUNDAMENTAL_NUMERIC_FIELDS if field != "market_cap")


def _parse_info(symbol: str, info: dict) -> dict:
    def num(*keys):
        for key in keys:
            val = _safe_float(info.get(key))
            if val is not None:
                return val
        return None

    def pct(*keys):
        val = num(*keys)
        return val * 100 if val is not None else None

    # Yahoo's dividendYield units changed across yfinance versions; rate / price is unambiguous.
    dividend_rate = num("dividendRate", "trailingAnnualDividendRate")
    quote_price = num("currentPrice", "regularMarketPrice", "previousClose")
    if dividend_rate is not None and quote_price:
        dividend_yield = dividend_rate / quote_price * 100
    else:
        dividend_yield = num("dividendYield")

    return {
        "ticker": symbol,
        "company_name": info.get("longName") or info.get("shortName"),
        "exchange": info.get("exchange"),
        "country": info.get("country"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": num("marketCap"),
        "pe": num("trailingPE", "forwardPE"),
        "forward_pe": num("forwardPE"),
        "pb": num("priceToBook"),
        "ps": num("priceToSalesTrailing12Months"),
        "ev_ebitda": num("enterpriseToEbitda"),
        "peg": num("pegRatio", "trailingPegRatio"),
        "dividend_yield": dividend_yield,
        "beta": num("beta"),
        "roe": pct("returnOnEquity"),
        "roa": pct("returnOnAssets"),
        "profit_margin": pct("profitMargins"),
        "operating_margin": pct("operatingMargins"),
        "debt_to_equity": num("debtToEquity"),
        "current_ratio": num("currentRatio"),
        "quick_ratio": num("quickRatio"),
        "revenue_growth": pct("revenueGrowth"),
        "earnings_growth": pct("earningsGrowth", "earningsQuarterlyGrowth"),
        "fifty_two_week_high": num("fiftyTwoWeekHigh"),
        "fifty_two_week_low": num("fiftyTwoWeekLow"),
    }


def _fetch_info_row(symbol: str, rate_limited: threading.Event) -> tuple[dict, str, str]:
    """Return (row, status, error). Retries transient errors once; stops the scan's live fetches on rate limits."""
    error = ""
    for attempt in range(2):
        if rate_limited.is_set():
            return {"ticker": symbol}, "failed", "Yahoo rate limit hit during this scan; will retry next scan"
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:120]}"
            if "RateLimit" in type(exc).__name__:
                rate_limited.set()
                return {"ticker": symbol}, "failed", error
            time.sleep(1.0 * (attempt + 1))
            continue
        try:
            row = _parse_info(symbol, info)
        except Exception as exc:
            return {"ticker": symbol}, "failed", f"parse error: {exc}"
        if _has_fundamentals(row):
            return row, "fetched", ""
        return row, "failed", "no fundamental fields returned by Yahoo"
    return {"ticker": symbol}, "failed", error or "unknown error"


def _read_disk_cache() -> dict[str, dict]:
    """Fresh, non-empty fundamentals rows from disk keyed by ticker."""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        df = pd.read_csv(CACHE_PATH, keep_default_na=False, na_values=[""])
    except Exception:
        return {}
    if df.empty or "ticker" not in df.columns:
        return {}

    cutoff = date.today() - timedelta(days=FUNDAMENTALS_CACHE_MAX_AGE_DAYS)
    rows: dict[str, dict] = {}
    for record in df.to_dict("records"):
        try:
            cached_on = datetime.strptime(str(record.get("cache_date")), "%Y-%m-%d").date()
        except ValueError:
            continue
        if cached_on < cutoff or not _has_fundamentals(record):
            continue
        rows[normalize_ticker(record["ticker"])] = record
    return rows


def _write_disk_cache(new_rows: list[dict]) -> None:
    if not new_rows:
        return
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        today = str(date.today())
        merged = _read_disk_cache()
        for row in new_rows:
            merged[row["ticker"]] = {**row, "cache_date": today}
        pd.DataFrame(list(merged.values())).to_csv(CACHE_PATH, index=False)
    except Exception:
        pass


def fetch_fundamentals_cached(tickers, progress_callback=None) -> pd.DataFrame:
    """
    Fundamentals for each ticker from memory cache, disk cache, or live yfinance `.info`.

    Only successful rows are cached, so a rate-limited or failed call is retried on the next scan
    instead of being stored as permanent N/A. Adds `fundamentals_status`
    (cached / fetched / failed / skipped) and `fundamentals_error` columns.
    """
    tickers = [normalize_ticker(t) for t in tickers]
    disk = _read_disk_cache()

    results: dict[str, dict] = {}
    to_fetch = []
    for sym in tickers:
        cached = _FUNDAMENTALS_MEMORY.get(sym) or disk.get(sym)
        if cached is not None:
            results[sym] = {**cached, "fundamentals_status": "cached", "fundamentals_error": ""}
        else:
            to_fetch.append(sym)

    skipped = to_fetch[MAX_LIVE_FUNDAMENTALS_FETCH:]
    to_fetch = to_fetch[:MAX_LIVE_FUNDAMENTALS_FETCH]
    for sym in skipped:
        results[sym] = {
            "ticker": sym,
            "fundamentals_status": "skipped",
            "fundamentals_error": f"over per-scan live fetch limit ({MAX_LIVE_FUNDAMENTALS_FETCH})",
        }

    fetched_rows = []
    rate_limited = threading.Event()
    if to_fetch:
        with ThreadPoolExecutor(max_workers=FUNDAMENTAL_WORKERS) as pool:
            futures = {pool.submit(_fetch_info_row, sym, rate_limited): sym for sym in to_fetch}
            for done, future in enumerate(as_completed(futures), start=1):
                sym = futures[future]
                try:
                    row, status, error = future.result()
                except Exception as exc:
                    row, status, error = {"ticker": sym}, "failed", str(exc)
                results[sym] = {**row, "fundamentals_status": status, "fundamentals_error": error}
                if status == "fetched":
                    _FUNDAMENTALS_MEMORY[sym] = row
                    fetched_rows.append(row)
                if progress_callback:
                    progress_callback(done, len(to_fetch))

    _write_disk_cache(fetched_rows)

    df = pd.DataFrame([results[sym] for sym in tickers])
    return df.drop(columns=["cache_date"], errors="ignore")


# ---------------------------------------------------------------------------
# Normalize, format, pipeline
# ---------------------------------------------------------------------------

LEGACY_ALIASES = {
    "Ticker": "ticker",
    "Price": "price",
    "MarketCap": "market_cap",
    "PE": "pe",
    "PB": "pb",
    "PS": "ps",
    "PEG": "peg",
    "EV_EBITDA": "ev_ebitda",
    "DividendYield": "dividend_yield",
    "Beta": "beta",
    "ROE": "roe",
    "ROA": "roa",
    "ProfitMargin": "profit_margin",
    "OperatingMargin": "operating_margin",
    "DebtToEquity": "debt_to_equity",
    "CurrentRatio": "current_ratio",
    "QuickRatio": "quick_ratio",
    "EPSGrowth": "earnings_growth",
    "RevenueGrowth": "revenue_growth",
    "Sector": "sector",
    "Week": "week",
    "Month": "month",
    "3Months": "quarter",
    "6Months": "six_month",
    "Year": "year",
    "Volatility": "volatility",
    "Sharpe": "sharpe",
    "RSI": "rsi",
    "SMA20": "sma20",
    "SMA50": "sma50",
    "SMA200": "sma200",
    "MACD_Bullish": "macd_bullish",
    "BB_Position": "bb_position",
}

PRICE_NUMERIC_FIELDS = [
    "price", "previous_close", "open", "day_high", "day_low", "volume", "average_volume",
    "change_percent", "week", "month", "quarter", "six_month", "year", "volatility",
    "sharpe", "rsi", "sma20", "sma50", "sma200", "hist_52w_high", "hist_52w_low",
]


def normalize_stock_dataframe(
    universe_df: pd.DataFrame,
    price_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge universe, price, and fundamentals into a normalized screener DataFrame."""
    base = universe_df.copy()
    base["ticker"] = base["ticker"].apply(normalize_ticker)
    base = base.drop_duplicates(subset=["ticker"])

    if price_df is not None and not price_df.empty:
        prices = price_df.copy()
        prices["ticker"] = prices["ticker"].apply(normalize_ticker)
        prices = prices.drop_duplicates(subset=["ticker"])
        base = base.merge(prices, on="ticker", how="left")

    if fundamentals_df is not None and not fundamentals_df.empty:
        fund = fundamentals_df.copy()
        fund["ticker"] = fund["ticker"].apply(normalize_ticker)
        fund = fund.drop_duplicates(subset=["ticker"])
        overlap = [c for c in fund.columns if c != "ticker" and c in base.columns]
        fund = fund.rename(columns={c: f"{c}_fund" for c in overlap})
        base = base.merge(fund, on="ticker", how="left")
        for col in overlap:
            # Yahoo's values (e.g. longName) are preferred over the symbol-file text.
            base[col] = base[f"{col}_fund"].where(base[f"{col}_fund"].notna() & (base[f"{col}_fund"] != ""), base[col])
            base = base.drop(columns=[f"{col}_fund"])

    for col in FUNDAMENTAL_NUMERIC_FIELDS + PRICE_NUMERIC_FIELDS:
        if col not in base.columns:
            base[col] = np.nan
        base[col] = pd.to_numeric(base[col], errors="coerce")
    for col in FUNDAMENTAL_TEXT_FIELDS + ["macd_bullish", "bb_position", "fundamentals_status", "fundamentals_error"]:
        if col not in base.columns:
            base[col] = None

    base["fifty_two_week_high"] = base["fifty_two_week_high"].fillna(base["hist_52w_high"])
    base["fifty_two_week_low"] = base["fifty_two_week_low"].fillna(base["hist_52w_low"])
    base["company_name"] = base["company_name"].fillna(base["ticker"])

    for alias, source in LEGACY_ALIASES.items():
        base[alias] = base[source]

    return base


def format_screener_results(df: pd.DataFrame) -> pd.DataFrame:
    """Format normalized results for display."""
    out = df.copy()

    money_cols = ["price", "market_cap", "fifty_two_week_high", "fifty_two_week_low", "Price", "MarketCap"]
    for col in money_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

    pct_cols = [
        "change_percent", "dividend_yield", "roe", "roa", "profit_margin", "operating_margin",
        "revenue_growth", "earnings_growth", "week", "month", "quarter", "six_month", "year",
        "volatility", "DividendYield", "ROE", "ROA", "ProfitMargin", "OperatingMargin",
        "EPSGrowth", "RevenueGrowth", "Week", "Month", "3Months", "6Months", "Year", "Volatility",
    ]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            )

    ratio_cols = [
        "pe", "forward_pe", "pb", "ps", "ev_ebitda", "peg", "beta", "debt_to_equity",
        "current_ratio", "quick_ratio", "sharpe", "rsi", "PE", "PB", "PS", "PEG", "EV_EBITDA",
        "Beta", "DebtToEquity", "CurrentRatio", "QuickRatio", "Sharpe", "RSI",
    ]
    for col in ratio_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    return out


def _status_tickers(fundamentals_df: pd.DataFrame, status: str) -> list[str]:
    if fundamentals_df.empty or "fundamentals_status" not in fundamentals_df.columns:
        return []
    return fundamentals_df.loc[fundamentals_df["fundamentals_status"] == status, "ticker"].tolist()


def run_screener_pipeline(
    markets: list,
    universe_size: str,
    filter_kwargs: dict,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full screener pipeline.
    Returns: (filtered_df, full_normalized_df, debug_dict)
    """
    debug = {
        "symbols_loaded": 0,
        "symbols_scanned": 0,
        "price_fetched": 0,
        "price_failed_samples": [],
        "fundamentals_fetched": 0,
        "fundamentals_ok": 0,
        "fundamentals_cached": 0,
        "fundamentals_live": 0,
        "fundamentals_failed": 0,
        "fundamentals_skipped": 0,
        "fundamentals_failed_samples": [],
        "fetch_failures": 0,
        "matched": 0,
        "filter_rejections": {},
        "excluded_samples": [],
        "error": None,
    }

    def report(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    try:
        universe_df = load_symbol_universe(tuple(markets), universe_size)
        debug["symbols_loaded"] = len(universe_df)
        if universe_df.empty:
            return pd.DataFrame(), pd.DataFrame(), debug

        tickers = universe_df["ticker"].tolist()
        debug["symbols_scanned"] = len(tickers)

        report(0.05, f"Downloading ~15 months of prices for {len(tickers)} symbols...")
        price_df = fetch_price_data_batch(tickers)
        debug["price_fetched"] = len(price_df)
        priced = set(price_df["ticker"]) if not price_df.empty else set()
        debug["price_failed_samples"] = [t for t in tickers if t not in priced][:10]

        report(0.35, "Loading fundamentals (cache first, then live)...")

        def fundamentals_progress(done: int, total: int) -> None:
            report(0.35 + 0.5 * done / max(total, 1), f"Fetching fundamentals live: {done}/{total}")

        fundamentals_df = fetch_fundamentals_cached(tickers, progress_callback=fundamentals_progress)
        debug["fundamentals_cached"] = len(_status_tickers(fundamentals_df, "cached"))
        debug["fundamentals_live"] = len(_status_tickers(fundamentals_df, "fetched"))
        debug["fundamentals_ok"] = debug["fundamentals_cached"] + debug["fundamentals_live"]
        debug["fundamentals_fetched"] = debug["fundamentals_ok"]
        debug["fundamentals_failed"] = len(_status_tickers(fundamentals_df, "failed"))
        debug["fundamentals_skipped"] = len(_status_tickers(fundamentals_df, "skipped"))
        failed = fundamentals_df[fundamentals_df["fundamentals_status"] == "failed"]
        debug["fundamentals_failed_samples"] = [
            {"ticker": r["ticker"], "reason": r.get("fundamentals_error") or "unknown"}
            for r in failed.head(10).to_dict("records")
        ]

        report(0.9, "Normalizing and filtering...")
        normalized = normalize_stock_dataframe(universe_df, price_df, fundamentals_df)
        debug["fetch_failures"] = len(tickers) - int(normalized["price"].notna().sum())

        filter_cfg = build_filter_config(**filter_kwargs)
        filtered, rejections, excluded_samples = apply_screener_filters(normalized, filter_cfg)

        if filtered.empty and not any_filter_active(filter_cfg):
            filtered = normalized[normalized["price"].notna()].copy()

        debug["matched"] = len(filtered)
        debug["filter_rejections"] = rejections
        debug["excluded_samples"] = excluded_samples[:10]

        print(
            f"[screener] universe={universe_size} scanned={len(tickers)} "
            f"prices_ok={debug['price_fetched']} prices_failed={len(tickers) - debug['price_fetched']} "
            f"fundamentals_ok={debug['fundamentals_ok']} (cached={debug['fundamentals_cached']}, "
            f"live={debug['fundamentals_live']}) fundamentals_failed={debug['fundamentals_failed']} "
            f"skipped={debug['fundamentals_skipped']} matched={debug['matched']}",
            flush=True,
        )
        if debug["fundamentals_failed_samples"]:
            print(f"[screener] fundamentals failures (first 10): {debug['fundamentals_failed_samples']}", flush=True)

        report(1.0, "Done")
        return filtered, normalized, debug
    except Exception as exc:
        debug["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[screener] pipeline error: {debug['error']}", flush=True)
        return pd.DataFrame(), pd.DataFrame(), debug
