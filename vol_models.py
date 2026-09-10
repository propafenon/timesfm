"""Volatility forecasting: HAR-RV and its baselines.

Volatility is the part of a price series that IS predictable. It clusters:
turbulent days follow turbulent days. Returns do not behave that way, which is
why every returns model in this app lands near MASE 1.0 while these do not.

Corsi's HAR-RV captures the clustering with an ordinary least-squares regression
on three lookbacks - yesterday, the last week, the last month - standing in for
traders acting on different horizons. It is a linear model with four
coefficients and it remains hard to beat.

Fitted on log volatility: volatility is positive and right-skewed, so OLS on the
raw level is heteroskedastic and can predict negative values.
"""

import numpy as np

TRADING_DAYS = 252
DAILY_WINDOW = 1
WEEKLY_WINDOW = 5
MONTHLY_WINDOW = 22


def realized_volatility(prices, window=DAILY_WINDOW, annualize=True):
    """A realized-volatility series from prices, computed causally.

    window=1 gives the daily proxy |r_t|, which is noisy but unbiased and is the
    honest choice without intraday data. Larger windows smooth it, but note that
    overlapping windows make consecutive values mechanically correlated, which
    flatters any model that predicts them - including this one.
    """
    prices = np.asarray(prices, dtype=float).reshape(-1)
    positive = prices > 0
    log_prices = np.full(prices.shape, np.nan)
    log_prices[positive] = np.log(prices[positive])
    returns = np.diff(log_prices)

    squared = returns ** 2
    if window <= 1:
        variance = squared
    else:
        variance = np.full(squared.shape, np.nan)
        for i in range(squared.size):
            start = max(0, i - window + 1)
            chunk = squared[start:i + 1]
            if np.isfinite(chunk).any():
                variance[i] = np.nanmean(chunk)

    volatility = np.sqrt(np.maximum(variance, 0.0))
    if annualize:
        volatility = volatility * np.sqrt(TRADING_DAYS)
    # One shorter than prices, matching np.diff.
    return volatility


def _har_design(series, horizon):
    """Daily / weekly / monthly averages, and cumulative-ahead targets."""
    n = series.size
    first = MONTHLY_WINDOW
    last = n - horizon
    if last <= first:
        return None, None

    rows, targets = [], [[] for _ in range(horizon)]
    for t in range(first, last):
        daily = series[t - 1]
        weekly = float(np.mean(series[t - WEEKLY_WINDOW:t]))
        monthly = float(np.mean(series[t - MONTHLY_WINDOW:t]))
        if not np.isfinite([daily, weekly, monthly]).all():
            continue
        rows.append([1.0, daily, weekly, monthly])
        for step in range(horizon):
            targets[step].append(series[t + step])

    if not rows:
        return None, None
    return np.asarray(rows, dtype=float), [np.asarray(t, dtype=float) for t in targets]


def make_har_forecaster(floor=1e-8):
    """A model_fn for the walk-forward engine, over a VOLATILITY series.

    The context must already be realized volatility, not price: `values` passed
    to walk_forward is the RV series, so the naive reference is "tomorrow's
    volatility equals today's", which is the benchmark HAR has to beat.
    """
    def _forecast(context, horizon):
        series = np.asarray(context, dtype=float).reshape(-1)
        usable = np.isfinite(series) & (series > 0)
        if usable.sum() < MONTHLY_WINDOW + horizon + 10:
            last = float(series[usable][-1]) if usable.any() else floor
            return np.full(horizon, last), None

        # Fit in logs, so predictions cannot come back negative.
        logged = np.log(np.maximum(series, floor))
        logged[~np.isfinite(logged)] = np.nanmean(logged[np.isfinite(logged)])

        design, targets = _har_design(logged, horizon)
        if design is None or design.shape[0] < 30:
            return np.full(horizon, float(series[usable][-1])), None

        latest = np.array([[
            1.0,
            logged[-1],
            float(np.mean(logged[-WEEKLY_WINDOW:])),
            float(np.mean(logged[-MONTHLY_WINDOW:])),
        ]], dtype=float)

        out = np.empty(horizon, dtype=float)
        for step in range(horizon):
            coefficients, *_ = np.linalg.lstsq(design, targets[step], rcond=None)
            # (1, 4) @ (4,) is a 1-element array; numpy 2 refuses float() on it.
            out[step] = float((latest @ coefficients)[0])

        return np.exp(out), None

    _forecast.__name__ = "har_rv"
    return _forecast


def ewma_volatility_forecaster(lam=0.94):
    """RiskMetrics EWMA, the other standard volatility benchmark."""
    def _forecast(context, horizon):
        series = np.asarray(context, dtype=float).reshape(-1)
        series = series[np.isfinite(series)]
        if series.size == 0:
            return np.full(horizon, np.nan), None
        variance = float(series[0] ** 2)
        for value in series[1:]:
            variance = lam * variance + (1 - lam) * float(value) ** 2
        return np.full(horizon, np.sqrt(max(variance, 0.0))), None
    _forecast.__name__ = f"ewma_{lam}"
    return _forecast


def available_models():
    return {"har_rv": make_har_forecaster(), "ewma_0.94": ewma_volatility_forecaster()}
