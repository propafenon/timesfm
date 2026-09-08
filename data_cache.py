"""Local OHLCV cache.

Two reasons this exists, both of which matter more for backtesting than for
plotting a single chart:

  Reproducibility - yfinance silently revises history. A backtest that
  re-downloads on every run is comparing against a moving target, and results
  stop being reproducible from one day to the next.

  Leakage - yfinance 1.x defaults to auto_adjust=True, so the 'Close' it hands
  back is split- and dividend-adjusted using corporate actions that happened
  AFTER the bar. Harmless when forecasting forward from today, a genuine
  look-ahead when backtesting 2023 with 2025's dividends folded in. The cache
  stores raw prices plus 'Adj Close' so the caller chooses explicitly.
"""

import os
import json
import datetime

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pyarrow  # noqa: F401
    _PARQUET = True
except ImportError:
    _PARQUET = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _slug(ticker, interval, period):
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in ticker)
    return f"{safe}__{interval}__{period}"


def _paths(ticker, interval, period):
    base = os.path.join(CACHE_DIR, _slug(ticker, interval, period))
    return (base + ('.parquet' if _PARQUET else '.csv'), base + '.meta.json')


def cache_entries():
    """Everything currently cached, newest fetch first."""
    if not os.path.isdir(CACHE_DIR):
        return []
    entries = []
    for name in os.listdir(CACHE_DIR):
        if not name.endswith('.meta.json'):
            continue
        try:
            with open(os.path.join(CACHE_DIR, name), encoding='utf-8') as handle:
                entries.append(json.load(handle))
        except (OSError, ValueError):
            continue
    return sorted(entries, key=lambda e: e.get('fetched_at', ''), reverse=True)


def _read(data_path):
    if _PARQUET:
        return pd.read_parquet(data_path)
    frame = pd.read_csv(data_path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def _write(frame, data_path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if _PARQUET:
        frame.to_parquet(data_path)
    else:
        frame.to_csv(data_path)


def _download(ticker, period, interval):
    if yf is None:
        raise ImportError("yfinance library is missing.")

    # auto_adjust=False keeps raw prices AND the adjusted close side by side,
    # so the adjustment decision moves to the caller instead of being implicit.
    frame = yf.download(ticker, period=period, interval=interval,
                        progress=False, timeout=30, auto_adjust=False)

    if frame is None or frame.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame[[c for c in OHLCV_COLUMNS if c in frame.columns]].copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    return frame[~frame.index.duplicated(keep='last')].sort_index()


def apply_adjustment(frame):
    """Scale OHLC by the Adj Close / Close ratio.

    Volume is left raw: adjusting it correctly needs the split factor alone,
    and dividends must not touch it.
    """
    if "Adj Close" not in frame.columns or "Close" not in frame.columns:
        return frame

    frame = frame.copy()
    ratio = frame["Adj Close"] / frame["Close"].replace(0, pd.NA)
    ratio = ratio.astype(float).ffill().bfill().fillna(1.0)
    for column in ("Open", "High", "Low", "Close"):
        if column in frame.columns:
            frame[column] = frame[column] * ratio
    return frame


def get_ohlcv(ticker, period, interval, adjusted=True, force_refresh=False,
              max_age_hours=12.0):
    """Cached OHLCV. Returns (frame, meta) where meta says where it came from.

    Set adjusted=False for backtesting to avoid the look-ahead described above.
    """
    ticker = ticker.strip().upper()
    data_path, meta_path = _paths(ticker, interval, period)

    meta = None
    frame = None
    if not force_refresh and os.path.exists(data_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, encoding='utf-8') as handle:
                meta = json.load(handle)
            age_hours = (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.datetime.fromisoformat(meta['fetched_at'])
            ).total_seconds() / 3600.0
            if age_hours <= max_age_hours:
                frame = _read(data_path)
                meta = dict(meta, source='cache', age_hours=round(age_hours, 2))
            else:
                meta = None
        except (OSError, ValueError, KeyError):
            meta = None
            frame = None

    if frame is None:
        frame = _download(ticker, period, interval)
        meta = {
            'ticker': ticker, 'interval': interval, 'period': period,
            'rows': int(len(frame)),
            'first': frame.index[0].isoformat(),
            'last': frame.index[-1].isoformat(),
            'fetched_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'format': 'parquet' if _PARQUET else 'csv',
        }
        _write(frame, data_path)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(meta_path, 'w', encoding='utf-8') as handle:
            json.dump(meta, handle, indent=2)
        meta = dict(meta, source='download', age_hours=0.0)

    if adjusted:
        frame = apply_adjustment(frame)
    meta = dict(meta, adjusted=bool(adjusted))
    return frame, meta


def clear_cache():
    """Delete every cached series. Returns how many files were removed."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    removed = 0
    for name in os.listdir(CACHE_DIR):
        if name.endswith(('.parquet', '.csv', '.meta.json')):
            os.remove(os.path.join(CACHE_DIR, name))
            removed += 1
    return removed
