"""Covariate catalogue and feature engineering.

Two halves:

  MACRO_CATALOG - external series (FRED and Yahoo) covering policy rates, FX,
  energy, metals, credit spreads and risk proxies. These are what actually move
  a Borsa Istanbul name: the CBRT policy rate, USDTRY, Brent, and global risk
  appetite.

  build_features - technical and statistical columns derived from OHLCV.

Everything here is causal. Rolling and ewm windows only ever look backwards,
external series are forward-filled from their own release dates and never
back-filled, and standardisation happens later against context-window
statistics only. Back-filling a macro series is not a cosmetic choice: it takes
the first known value and writes it into every earlier date, which is future
information sitting in the training window.
"""

import numpy as np
import pandas as pd

# label -> (source, symbol, note)
MACRO_CATALOG = {
    # --- Policy rates -----------------------------------------------------
    "TR policy rate (CBRT)":      ("fred", "IRSTCI01TRM156N", "Immediate rate, central bank of Turkiye"),
    "TR 3-month interbank":       ("fred", "IR3TIB01TRM156N", "Short-term TRY funding cost"),
    "US fed funds (effective)":   ("fred", "DFF", "Daily effective federal funds rate"),
    "US 3-month T-bill":          ("fred", "DTB3", "Short end of the US curve"),
    "US 2-year treasury":         ("fred", "DGS2", "Policy expectations"),
    "US 10-year treasury":        ("fred", "DGS10", "Global discount rate"),
    "US 10y-2y spread":           ("fred", "T10Y2Y", "Curve slope, recession proxy"),

    # --- Inflation --------------------------------------------------------
    "TR CPI":                     ("fred", "TURCPIALLMINMEI", "Turkish consumer prices"),
    "US CPI":                     ("fred", "CPIAUCSL", "US consumer prices"),
    "US 10y breakeven inflation": ("fred", "T10YIE", "Market-implied inflation"),

    # --- FX ---------------------------------------------------------------
    "USD/TRY":                    ("yahoo", "USDTRY=X", "The single biggest driver for a BIST name"),
    "EUR/TRY":                    ("yahoo", "EURTRY=X", "Trade-weighted second leg"),
    "EUR/USD":                    ("yahoo", "EURUSD=X", "Dollar direction"),
    "Dollar index (DXY)":         ("yahoo", "DX-Y.NYB", "Broad dollar strength"),

    # --- Energy and commodities ------------------------------------------
    "Brent crude":                ("yahoo", "BZ=F", "Turkiye is a net energy importer"),
    "WTI crude":                  ("yahoo", "CL=F", "US benchmark crude"),
    "Natural gas":                ("yahoo", "NG=F", "Import cost and inflation input"),
    "Gold":                       ("yahoo", "GC=F", "TRY hedge and store of value"),
    "Silver":                     ("yahoo", "SI=F", "Industrial and monetary metal"),
    "Copper":                     ("yahoo", "HG=F", "Global industrial demand"),

    # --- Risk and equity beta --------------------------------------------
    "BIST 100 index":             ("yahoo", "XU100.IS", "Domestic market beta"),
    "Turkiye ETF (TUR)":          ("yahoo", "TUR", "Foreign investor flow proxy"),
    "S&P 500":                    ("yahoo", "^GSPC", "Global equity beta"),
    "VIX":                        ("yahoo", "^VIX", "Global risk appetite"),
    "EM equities (EEM)":          ("yahoo", "EEM", "Emerging market flows"),
    "US high-yield OAS":          ("fred", "BAMLH0A0HYM2", "Global credit stress"),
    "EM corporate OAS":           ("fred", "BAMLEMCBPIOAS", "EM credit stress"),
}

# A sensible starting set of roughly 15, spanning rates, FX, energy and risk.
DEFAULT_MACRO_SELECTION = (
    "TR policy rate (CBRT)", "US fed funds (effective)", "US 10-year treasury",
    "US 10y-2y spread", "TR CPI", "USD/TRY", "EUR/TRY", "Dollar index (DXY)",
    "Brent crude", "Natural gas", "Gold", "BIST 100 index", "S&P 500", "VIX",
    "US high-yield OAS",
)


def _safe_div(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def build_features(data):
    """Add technical and statistical columns to an OHLCV frame.

    Everything is computed from past observations only, so a value at time t
    could have been known at time t.
    """
    data = data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.loc[:, ~data.columns.duplicated()]

    close = pd.to_numeric(data.get("Close"), errors="coerce")
    if close is None or close.dropna().empty:
        raise ValueError("The downloaded data does not contain a usable Close series.")
    high = pd.to_numeric(data.get("High", close), errors="coerce")
    low = pd.to_numeric(data.get("Low", close), errors="coerce")
    open_ = pd.to_numeric(data.get("Open", close), errors="coerce")
    volume = pd.to_numeric(data.get("Volume"), errors="coerce")

    # --- returns ----------------------------------------------------------
    data["Return_1D"] = close.pct_change()
    positive = close.where(close > 0)
    data["LogReturn_1D"] = np.log(positive).diff()

    # --- trend / momentum -------------------------------------------------
    data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]
    data["SMA_20"] = close.rolling(20, min_periods=20).mean()
    data["SMA_50"] = close.rolling(50, min_periods=50).mean()
    data["SMA_200"] = close.rolling(200, min_periods=200).mean()
    # Ratios rather than levels: scale-free, so they stay comparable over years.
    data["Price_over_SMA20"] = _safe_div(close, data["SMA_20"])
    data["Price_over_SMA200"] = _safe_div(close, data["SMA_200"])
    data["SMA20_over_SMA50"] = _safe_div(data["SMA_20"], data["SMA_50"])
    for window in (5, 20, 60, 120):
        data[f"Momentum_{window}D"] = close.pct_change(window)
    # Classic 12-1: a year of momentum excluding the last month's reversal.
    data["Momentum_12_1"] = close.shift(21) / close.shift(252) - 1.0

    # --- oscillators ------------------------------------------------------
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["RSI_14"] = 100 - (100 / (1 + _safe_div(gain, loss)))
    lowest_low = low.rolling(14, min_periods=14).min()
    highest_high = high.rolling(14, min_periods=14).max()
    data["Stochastic_K"] = 100 * _safe_div(close - lowest_low, highest_high - lowest_low)

    # --- volatility -------------------------------------------------------
    rolling_std = close.rolling(20, min_periods=20).std()
    data["Bollinger_Middle"] = data["SMA_20"]
    data["Bollinger_Upper"] = data["SMA_20"] + 2 * rolling_std
    data["Bollinger_Lower"] = data["SMA_20"] - 2 * rolling_std
    # %B is the normalised form: where price sits inside the band, 0 to 1.
    band_width = (data["Bollinger_Upper"] - data["Bollinger_Lower"])
    data["Bollinger_PctB"] = _safe_div(close - data["Bollinger_Lower"], band_width)
    data["Bollinger_Width"] = _safe_div(band_width, data["SMA_20"])

    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    data["ATR_14"] = true_range.rolling(14, min_periods=14).mean()
    data["ATR_Pct"] = _safe_div(data["ATR_14"], close)

    for window in (5, 20, 60):
        data[f"RealizedVol_{window}D"] = data["LogReturn_1D"].rolling(window, min_periods=window).std() * np.sqrt(252)

    # Parkinson and Garman-Klass use the intrabar range and are far more
    # efficient than close-to-close at the same sample size.
    hl = np.log(_safe_div(high, low).where(lambda s: s > 0))
    data["Parkinson_20D"] = np.sqrt(
        (hl ** 2).rolling(20, min_periods=20).mean() / (4 * np.log(2))
    ) * np.sqrt(252)
    co = np.log(_safe_div(close, open_).where(lambda s: s > 0))
    data["GarmanKlass_20D"] = np.sqrt(
        (0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2).rolling(20, min_periods=20).mean().clip(lower=0)
    ) * np.sqrt(252)

    # --- shape and drawdown ----------------------------------------------
    data["Return_Skew_60D"] = data["LogReturn_1D"].rolling(60, min_periods=60).skew()
    data["Return_Kurt_60D"] = data["LogReturn_1D"].rolling(60, min_periods=60).kurt()
    rolling_peak = close.rolling(252, min_periods=20).max()
    data["Drawdown_252D"] = _safe_div(close, rolling_peak) - 1.0
    data["Dist_From_52W_High"] = data["Drawdown_252D"]

    # --- volume and liquidity --------------------------------------------
    if volume is not None and not volume.dropna().empty:
        data["OBV"] = (np.sign(close.diff()).fillna(0) * volume.fillna(0)).cumsum()
        dollar_volume = close * volume
        data["DollarVolume"] = dollar_volume
        volume_mean = volume.rolling(20, min_periods=20).mean()
        volume_std = volume.rolling(20, min_periods=20).std()
        data["Volume_Z_20D"] = _safe_div(volume - volume_mean, volume_std)
        # Amihud: absolute return per unit of traded value, the standard
        # illiquidity proxy.
        data["Amihud_Illiquidity"] = _safe_div(
            data["Return_1D"].abs(), dollar_volume
        ).rolling(20, min_periods=20).mean()

    return data


def align_external(series, market_index, label):
    """Reindex an external series onto the market calendar, causally.

    Forward fill carries the last released value forward, which is what an
    observer would actually have known. Leading gaps stay NaN: filling them
    backwards would write a later value into earlier dates.
    """
    series = pd.Series(series).astype(float)
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index))
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series = series[~series.index.duplicated(keep="last")].sort_index()

    combined = series.reindex(series.index.union(market_index)).ffill()
    aligned = combined.reindex(market_index)
    aligned.name = label
    return aligned


def standardize_context(matrix):
    """Z-score each covariate against the context window itself.

    Policy rates sit near 50 while a share price sits near 400; handing both to
    the model unscaled lets magnitude stand in for information. Using only the
    context window's own mean and standard deviation keeps this leak-free,
    because those statistics are recomputed at every forecast origin from data
    the model can already see.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0:
        return matrix

    mean = np.nanmean(matrix, axis=1, keepdims=True)
    std = np.nanstd(matrix, axis=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    scaled = (matrix - np.where(np.isfinite(mean), mean, 0.0)) / std
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
