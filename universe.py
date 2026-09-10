"""Multi-ticker fetching: turn one-name forecasting into a cross-section.

The single structural change. Ranking many names against each other needs only
the ORDER to be slightly better than random, and market-wide moves cancel out
between the top and bottom of the ranking. Forecasting one name needs the
absolute prediction to be right, which is far harder and gives 252 bets a year
instead of thousands.

SURVIVORSHIP WARNING, because it is the most common way a backtest lies: the
lists below are today's members. Backtesting them over ten years silently
excludes every name that was delisted, merged or demoted in between - exactly
the losers - and inflates every result. `coverage_report` surfaces what is
missing so the distortion is at least visible. Point-in-time constituent data is
the only real fix and it is not free.
"""

import numpy as np
import pandas as pd

import data_cache

# Liquid Borsa Istanbul names. Written from knowledge, NOT verified against
# current index membership: expect a few to fail or to have been renamed, and
# check them before trusting any result. Failures are skipped, not fatal.
BIST_LIQUID = [
    "ASELS.IS", "THYAO.IS", "GARAN.IS", "AKBNK.IS", "ISCTR.IS", "YKBNK.IS",
    "KCHOL.IS", "SAHOL.IS", "TUPRS.IS", "EREGL.IS", "BIMAS.IS", "SISE.IS",
    "PETKM.IS", "FROTO.IS", "TOASO.IS", "ARCLK.IS", "TCELL.IS", "TTKOM.IS",
    "HEKTS.IS", "KOZAL.IS", "KOZAA.IS", "PGSUS.IS", "ENKAI.IS", "VESTL.IS",
    "SASA.IS", "TAVHL.IS", "ALARK.IS", "EKGYO.IS", "HALKB.IS", "VAKBN.IS",
    "TSKB.IS", "MGROS.IS", "ULKER.IS", "AKSEN.IS", "GUBRF.IS", "KRDMD.IS",
    "OYAKC.IS", "DOHOL.IS", "SOKM.IS", "CIMSA.IS",
]

PRESETS = {"BIST liquid (~40)": BIST_LIQUID}


def fetch_panel(tickers, period="max", interval="1d", adjusted=False,
                column="Close", force_refresh=False, progress_cb=None):
    """Wide price frame: rows are dates, columns are tickers.

    One cached fetch per ticker, so a failure loses that name and not the run.
    Returns (prices, failures). Dates are the union across names, so a name that
    listed late simply has NaN before it existed - which is correct, and the
    portfolio layer must skip it there rather than fill it.
    """
    series_by_ticker = {}
    failures = {}

    for position, ticker in enumerate(tickers, start=1):
        if progress_cb is not None:
            progress_cb(position, len(tickers), ticker)
        try:
            frame, _meta = data_cache.get_ohlcv(
                ticker, period, interval, adjusted=adjusted,
                force_refresh=force_refresh,
            )
            if column not in frame.columns:
                raise ValueError(f"no '{column}' column")
            values = pd.to_numeric(frame[column], errors="coerce")
            values = values[values > 0]
            if values.empty:
                raise ValueError("no positive prices")
            series_by_ticker[ticker] = values
        except Exception as error:
            failures[ticker] = str(error)[:80]

    if not series_by_ticker:
        raise ValueError("No tickers could be fetched.")

    prices = pd.DataFrame(series_by_ticker).sort_index()
    prices.index = pd.DatetimeIndex(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    return prices[~prices.index.duplicated(keep="last")], failures


def coverage_report(prices):
    """Where the panel is thin, and how badly survivorship may be biting."""
    counts = prices.notna().sum()
    first_valid = prices.apply(lambda column: column.first_valid_index())
    last_valid = prices.apply(lambda column: column.last_valid_index())
    panel_end = prices.index[-1]

    # A name whose last quote is well before the panel ends has stopped trading:
    # delisted, suspended, or renamed. Its absence from a current-membership list
    # is exactly the survivorship hole.
    stale = [
        ticker for ticker in prices.columns
        if last_valid[ticker] is not None
        and (panel_end - last_valid[ticker]).days > 30
    ]

    return {
        "n_tickers": int(prices.shape[1]),
        "n_dates": int(prices.shape[0]),
        "start": prices.index[0].date().isoformat(),
        "end": panel_end.date().isoformat(),
        "median_history": int(counts.median()),
        "min_history": int(counts.min()),
        "names_at_start": int(prices.iloc[0].notna().sum()),
        "names_at_end": int(prices.iloc[-1].notna().sum()),
        "stale_names": stale,
        "first_valid": {t: (d.date().isoformat() if d is not None else None)
                        for t, d in first_valid.items()},
    }


def survivorship_note(report):
    """One line the user should read before believing any backtest number."""
    return (
        f"{report['n_tickers']} names, {report['names_at_start']} present at "
        f"{report['start']} and {report['names_at_end']} at {report['end']}. "
        "This list is CURRENT membership, so names delisted along the way are "
        "absent and results are optimistic by an unknown amount."
    )
