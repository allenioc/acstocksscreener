"""Scalable stock screener backend using public symbol lists and yfinance."""

from __future__ import annotations

import os
import re
from datetime import date, datetime
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

UNIVERSE_SIZE_OPTIONS = {
    "Top 100": 100,
    "Top 500": 500,
    "Top 1000": 1000,
    "Full universe": None,
}


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


def _is_common_stock_symbol(symbol) -> bool:
    if symbol is None or (isinstance(symbol, float) and np.isnan(symbol)):
        return False
    symbol = str(symbol).strip()
    if not symbol or symbol == "nan":
        return False
    if EXCLUDE_SYMBOL_PATTERN.search(symbol):
        return False
    if len(symbol) > 6:
        return False
    return symbol.replace(".", "").isalnum()


def load_us_symbols_from_nasdaq() -> pd.DataFrame:
    """Load US common-stock symbols from Nasdaq Trader public files."""
    frames = []

    try:
        nasdaq_resp = requests.get(NASDAQ_LISTED_URL, timeout=20)
        nasdaq_resp.raise_for_status()
        nasdaq_df = pd.read_csv(StringIO(nasdaq_resp.text), sep="|")
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
        other_df = pd.read_csv(StringIO(other_resp.text), sep="|")
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
    df = df[df["ticker"].notna() & (df["ticker"] != "nan")]

    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].astype(str).str.upper() != "Y"]
    if "ETF" in df.columns:
        df = df[df["ETF"].astype(str).str.upper() != "Y"]

    df = df[df["ticker"].apply(_is_common_stock_symbol)]
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    df["priority"] = 5000
    return df[["ticker", "company_name", "exchange", "country", "priority"]]


def load_local_symbols_backup() -> pd.DataFrame:
    """Load curated fallback symbol list."""
    if not os.path.exists(BACKUP_PATH):
        return pd.DataFrame(columns=["ticker", "company_name", "exchange", "country", "priority"])
    df = pd.read_csv(BACKUP_PATH)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "priority" not in df.columns:
        df["priority"] = 9999
    return df


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

    universe = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker"], keep="first")
    universe = universe.sort_values(["priority", "ticker"], ascending=[True, True])

    limit = UNIVERSE_SIZE_OPTIONS.get(universe_size)
    if limit is not None:
        universe = universe.head(limit)

    return universe.reset_index(drop=True)


MAX_LIVE_FUNDAMENTALS_FETCH = 200


def _fetch_single_price(symbol: str, period: str = "6mo") -> Optional[dict]:
    try:
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        return _price_row_from_series(symbol, hist["Close"], hist.get("Volume"))
    except Exception:
        return None


def fetch_price_data_batch(tickers: list[str], period: str = "6mo") -> pd.DataFrame:
    """Batch download price/volume data via yfinance."""
    if not tickers:
        return pd.DataFrame()

    rows = []
    chunk_size = 100

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            raw = yf.download(
                tickers=chunk,
                period=period,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
        except Exception:
            raw = None

        if raw is None or raw.empty:
            continue

        if len(chunk) == 1:
            sym = chunk[0]
            try:
                close = raw["Close"].dropna()
                if close.empty:
                    continue
                rows.append(_price_row_from_series(sym, close, raw.get("Volume")))
            except Exception:
                continue
            continue

        for sym in chunk:
            try:
                if sym not in raw.columns.get_level_values(0):
                    continue
                close = raw[sym]["Close"].dropna()
                if close.empty:
                    continue
                vol = raw[sym]["Volume"] if "Volume" in raw[sym] else None
                rows.append(_price_row_from_series(sym, close, vol))
            except Exception:
                continue

    fetched = {row["ticker"] for row in rows}
    missing = [sym for sym in tickers if sym not in fetched]
    for sym in missing:
        row = _fetch_single_price(sym, period)
        if row:
            rows.append(row)

    return pd.DataFrame(rows)


def _price_row_from_series(symbol: str, close: pd.Series, volume: Optional[pd.Series]) -> dict:
    price = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else price
    change_pct = ((price - prev) / prev * 100) if prev else None

    def perf(days: int) -> Optional[float]:
        if len(close) > days:
            old = float(close.iloc[-days - 1])
            return ((price - old) / old * 100) if old else None
        return None

    avg_vol = float(volume.dropna().tail(20).mean()) if volume is not None and not volume.dropna().empty else None
    cur_vol = float(volume.iloc[-1]) if volume is not None and not volume.dropna().empty else None

    return {
        "ticker": symbol,
        "price": price,
        "previous_close": prev,
        "open": None,
        "day_high": None,
        "day_low": None,
        "volume": cur_vol,
        "average_volume": avg_vol,
        "change_percent": change_pct,
        "week": perf(5),
        "month": perf(21),
        "quarter": perf(63),
        "year": perf(252),
    }


def _read_disk_cache(max_age_days: int = 7) -> Optional[pd.DataFrame]:
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        df = pd.read_csv(CACHE_PATH)
        if "cache_date" not in df.columns or df.empty:
            return None
        cache_day = datetime.strptime(str(df["cache_date"].iloc[0]), "%Y-%m-%d").date()
        if (date.today() - cache_day).days > max_age_days:
            return None
        return df.drop(columns=["cache_date"], errors="ignore")
    except Exception:
        return None


def _merge_disk_cache(new_rows: pd.DataFrame) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = _read_disk_cache(max_age_days=365) or pd.DataFrame()
    if not existing.empty and "ticker" in existing.columns:
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker"], keep="last")
    else:
        combined = new_rows.copy()
    combined["cache_date"] = str(date.today())
    combined.to_csv(CACHE_PATH, index=False)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals_cached(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Fetch fundamentals per ticker with in-memory and incremental disk cache."""
    cached = _read_disk_cache()
    cached_by_ticker: dict[str, dict] = {}
    if cached is not None and not cached.empty:
        for _, row in cached.iterrows():
            cached_by_ticker[str(row["ticker"])] = row.to_dict()

    to_fetch = [sym for sym in tickers if sym not in cached_by_ticker]
    if len(to_fetch) > MAX_LIVE_FUNDAMENTALS_FETCH:
        to_fetch = to_fetch[:MAX_LIVE_FUNDAMENTALS_FETCH]
    rows = []

    for sym in to_fetch:
        row = {"ticker": sym}
        try:
            t = yf.Ticker(sym)
            info = {}
            try:
                info = t.info or {}
            except Exception:
                info = {}
            try:
                fast = t.fast_info
            except Exception:
                fast = {}

            row.update({
                "company_name": info.get("longName") or info.get("shortName"),
                "exchange": info.get("exchange"),
                "country": info.get("country"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": _safe_float(info.get("marketCap") or getattr(fast, "market_cap", None)),
                "pe": _safe_float(info.get("trailingPE") or info.get("forwardPE")),
                "forward_pe": _safe_float(info.get("forwardPE")),
                "pb": _safe_float(info.get("priceToBook")),
                "ps": _safe_float(info.get("priceToSalesTrailing12Months")),
                "ev_ebitda": _safe_float(info.get("enterpriseToEbitda")),
                "peg": _safe_float(info.get("pegRatio")),
                "dividend_yield": (_safe_float(info.get("dividendYield")) * 100)
                if info.get("dividendYield") is not None else None,
                "beta": _safe_float(info.get("beta")),
                "roe": (_safe_float(info.get("returnOnEquity")) * 100) if info.get("returnOnEquity") is not None else None,
                "roa": (_safe_float(info.get("returnOnAssets")) * 100) if info.get("returnOnAssets") is not None else None,
                "profit_margin": (_safe_float(info.get("profitMargins")) * 100) if info.get("profitMargins") is not None else None,
                "operating_margin": (_safe_float(info.get("operatingMargins")) * 100) if info.get("operatingMargins") is not None else None,
                "debt_to_equity": _safe_float(info.get("debtToEquity")),
                "current_ratio": _safe_float(info.get("currentRatio")),
                "quick_ratio": _safe_float(info.get("quickRatio")),
                "revenue_growth": (_safe_float(info.get("revenueGrowth")) * 100) if info.get("revenueGrowth") is not None else None,
                "earnings_growth": (
                    _safe_float(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")) * 100
                )
                if (info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")) is not None
                else None,
                "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
                "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
            })
        except Exception:
            pass
        rows.append(row)

    if rows:
        _merge_disk_cache(pd.DataFrame(rows))

    result_rows = []
    for sym in tickers:
        if sym in cached_by_ticker:
            result_rows.append(cached_by_ticker[sym])
        else:
            fetched = next((r for r in rows if r.get("ticker") == sym), {"ticker": sym})
            result_rows.append(fetched)

    return pd.DataFrame(result_rows)


def normalize_stock_dataframe(
    universe_df: pd.DataFrame,
    price_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge universe, price, and fundamentals into a normalized screener DataFrame."""
    base = universe_df.copy()
    if not price_df.empty:
        base = base.merge(price_df, on="ticker", how="left", suffixes=("", "_px"))
    if not fundamentals_df.empty:
        cols = [c for c in fundamentals_df.columns if c != "ticker"]
        base = base.merge(fundamentals_df[["ticker"] + cols], on="ticker", how="left", suffixes=("", "_fund"))

    for col in [
        "price", "previous_close", "open", "day_high", "day_low", "volume", "average_volume",
        "market_cap", "change_percent", "pe", "forward_pe", "pb", "ps", "ev_ebitda", "peg",
        "dividend_yield", "beta", "roe", "roa", "profit_margin", "operating_margin",
        "debt_to_equity", "current_ratio", "quick_ratio", "revenue_growth", "earnings_growth",
        "fifty_two_week_high", "fifty_two_week_low", "week", "month", "quarter", "year",
    ]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    if "company_name" in base.columns:
        base["company_name"] = base["company_name"].fillna(base.get("company_name_fund")).fillna(base["ticker"])

    # Legacy uppercase aliases for existing UI components
    base["Ticker"] = base["ticker"]
    base["Price"] = base["price"]
    base["MarketCap"] = base["market_cap"]
    base["PE"] = base["pe"]
    base["PB"] = base["pb"]
    base["PS"] = base["ps"]
    base["PEG"] = base["peg"]
    base["EV_EBITDA"] = base["ev_ebitda"]
    base["DividendYield"] = base["dividend_yield"]
    base["Beta"] = base["beta"]
    base["ROE"] = base["roe"]
    base["ROA"] = base["roa"]
    base["ProfitMargin"] = base["profit_margin"]
    base["OperatingMargin"] = base["operating_margin"]
    base["DebtToEquity"] = base["debt_to_equity"]
    base["CurrentRatio"] = base["current_ratio"]
    base["QuickRatio"] = base["quick_ratio"]
    base["EPSGrowth"] = base["earnings_growth"]
    base["RevenueGrowth"] = base["revenue_growth"]
    base["Sector"] = base["sector"] if "sector" in base.columns else None
    base["Week"] = base["week"] if "week" in base.columns else None
    base["Month"] = base["month"] if "month" in base.columns else None
    base["Year"] = base["year"] if "year" in base.columns else None

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
        "revenue_growth", "earnings_growth", "week", "month", "year", "DividendYield",
        "ROE", "ROA", "ProfitMargin", "OperatingMargin", "EPSGrowth", "RevenueGrowth",
        "Week", "Month", "Year",
    ]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A"
            )

    ratio_cols = [
        "pe", "forward_pe", "pb", "ps", "ev_ebitda", "peg", "beta", "debt_to_equity",
        "current_ratio", "quick_ratio", "PE", "PB", "PS", "PEG", "EV_EBITDA", "Beta",
        "DebtToEquity", "CurrentRatio", "QuickRatio",
    ]
    for col in ratio_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    return out


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
        "fundamentals_fetched": 0,
        "fetch_failures": 0,
        "matched": 0,
        "filter_rejections": {},
        "excluded_samples": [],
        "error": None,
    }

    try:
        universe_df = load_symbol_universe(tuple(markets), universe_size)
        debug["symbols_loaded"] = len(universe_df)
        if universe_df.empty:
            return pd.DataFrame(), pd.DataFrame(), debug

        tickers = universe_df["ticker"].tolist()
        debug["symbols_scanned"] = len(tickers)

        if progress_callback:
            progress_callback(0.1, "Fetching price data (batch)...")
        price_df = fetch_price_data_batch(tickers)
        debug["price_fetched"] = len(price_df)

        if progress_callback:
            progress_callback(0.45, "Fetching fundamentals (cached)...")
        fundamentals_df = fetch_fundamentals_cached(tuple(tickers))
        debug["fundamentals_fetched"] = int(fundamentals_df["ticker"].nunique()) if not fundamentals_df.empty else 0

        if progress_callback:
            progress_callback(0.75, "Normalizing and filtering...")

        normalized = normalize_stock_dataframe(universe_df, price_df, fundamentals_df)
        debug["fetch_failures"] = len(tickers) - int(normalized["price"].notna().sum()) if "price" in normalized.columns else len(tickers)

        filter_cfg = build_filter_config(**filter_kwargs)
        filtered, rejections, excluded_samples = apply_screener_filters(normalized, filter_cfg)

        if filtered.empty and not any_filter_active(filter_cfg):
            if "price" in normalized.columns:
                filtered = normalized[normalized["price"].notna()].copy()
            else:
                filtered = normalized.copy()

        debug["matched"] = len(filtered)
        debug["filter_rejections"] = rejections
        debug["excluded_samples"] = excluded_samples[:10]

        if progress_callback:
            progress_callback(1.0, "Done")

        return filtered, normalized, debug
    except Exception as exc:
        debug["error"] = str(exc)
        return pd.DataFrame(), pd.DataFrame(), debug
