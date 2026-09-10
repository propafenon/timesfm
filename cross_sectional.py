"""Cross-sectional ranking backtest.

Rank a universe at each rebalance, hold the top of the ranking against the
bottom, and let market-wide moves cancel between the two legs. Only the ORDER
has to be better than random.

Every factor here is built from shifts and rolling windows of past prices, so a
value dated t could have been computed at t. The portfolio then holds those
weights from t to t+1, which is the invariant tests/test_cross_sectional.py
pins: nothing dated t may use a price after t.
"""

import numpy as np
import pandas as pd
from scipy import stats

import metrics
import vol_models

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Factors. Higher rank = more attractive.
# --------------------------------------------------------------------------

def to_usd(prices, fx):
    """Restate a TRY price panel in dollars.

    IMPORTANT, and not what you would guess: this does NOT change the ranking
    produced by a ratio momentum factor. Momentum is p(t-21)/p(t-252); dividing
    every price by the same FX series gives

        momentum_usd = (momentum_try + 1) / G - 1

    where G is FX growth over the window. That is the same monotonic map applied
    to every name, so the order cannot change. Measured rank correlation between
    the two currencies is exactly 1.000 for momentum_12_1, momentum_6_1 and
    short_term_reversal.

    What it DOES change:
      - volatility-based factors, because a return series minus a common FX
        return series has a different variance per name depending on how each
        name co-moves with the currency (rank correlation ~0.88 for
        low_volatility)
      - trend_vs_200d, slightly, since the mean of p/fx is not mean(p)/fx
      - the realised RETURN STREAM, which is what a dollar-based investor
        actually earns, and therefore Sharpe, drawdown and volatility targeting

    `fx` is USDTRY (lira per dollar), forward-filled onto the equity calendar
    and never back-filled, so no rate is used before it was published.
    """
    rate = pd.Series(fx).astype(float)
    rate.index = pd.DatetimeIndex(rate.index)
    if rate.index.tz is not None:
        rate.index = rate.index.tz_localize(None)
    rate = rate[~rate.index.duplicated(keep="last")].sort_index()

    combined = rate.reindex(rate.index.union(prices.index)).ffill()
    aligned = combined.reindex(prices.index)
    return prices.div(aligned, axis=0)


def _overnight_log_returns(prices, panels):
    """log(open_t / close_{t-1}): the gap while the exchange was shut."""
    opens = panels["Open"].reindex_like(prices)
    ratio = opens / prices.shift(1)
    return np.log(ratio.where(ratio > 0))


def _intraday_log_returns(prices, panels):
    """log(close_t / open_t): the move during the session."""
    opens = panels["Open"].reindex_like(prices)
    ratio = prices / opens.where(opens > 0)
    return np.log(ratio.where(ratio > 0))


def overnight_momentum_12_1(prices, panels=None, lookback=252, skip=21):
    """Momentum built only from the overnight gaps.

    Most equity return accrues overnight rather than during the session, and the
    two are driven by different participants - news and foreign flow while the
    market is closed, domestic retail and market-making while it is open. On
    BIST, where foreign flow is a large overnight driver, the split may separate
    more cleanly than it does in the US.
    """
    if panels is None or "Open" not in panels:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    logs = _overnight_log_returns(prices, panels)
    return logs.rolling(lookback - skip, min_periods=lookback - skip).sum().shift(skip)


def intraday_momentum_12_1(prices, panels=None, lookback=252, skip=21):
    """The same window, built only from open-to-close moves."""
    if panels is None or "Open" not in panels:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    logs = _intraday_log_returns(prices, panels)
    return logs.rolling(lookback - skip, min_periods=lookback - skip).sum().shift(skip)


def overnight_share(prices, panels=None, lookback=252):
    """How much of a name's recent move happened while the market was closed.

    Not a return forecast: a character measure. Names whose gains arrive
    overnight behave differently from names that grind higher intraday.
    """
    if panels is None or "Open" not in panels:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    overnight = _overnight_log_returns(prices, panels).rolling(
        lookback, min_periods=lookback // 2).sum()
    intraday = _intraday_log_returns(prices, panels).rolling(
        lookback, min_periods=lookback // 2).sum()
    total = overnight.abs() + intraday.abs()
    return overnight / total.where(total > 0)


def momentum_12_1(prices, panels=None):
    """The classic: a year of momentum, skipping the last month's reversal.

    The most replicated anomaly in equities - and it lives at a monthly-to-
    annual horizon, not the one-day horizon this project has been testing.
    """
    return prices.shift(21) / prices.shift(252) - 1.0


def momentum_6_1(prices, panels=None):
    return prices.shift(21) / prices.shift(126) - 1.0


def short_term_reversal(prices, panels=None):
    """Last month's losers tend to bounce, so the sign is inverted."""
    return -(prices / prices.shift(21) - 1.0)


def low_volatility(prices, panels=None, window=60):
    """Low-volatility names have historically outperformed on a risk-adjusted
    basis; negated so that calmer ranks higher."""
    returns = np.log(prices).diff()
    return -returns.rolling(window, min_periods=window).std()


def trend_vs_200d(prices, panels=None):
    return prices / prices.rolling(200, min_periods=200).mean() - 1.0


FACTORS = {
    "momentum_12_1": momentum_12_1,
    "momentum_6_1": momentum_6_1,
    "short_term_reversal": short_term_reversal,
    "low_volatility": low_volatility,
    "trend_vs_200d": trend_vs_200d,
    # Need an Open panel; return all-NaN without one, which the portfolio layer
    # reports as "not enough eligible names" rather than silently ranking noise.
    "overnight_momentum_12_1": overnight_momentum_12_1,
    "intraday_momentum_12_1": intraday_momentum_12_1,
    "overnight_share": overnight_share,
}

# Factors that cannot be computed from closing prices alone.
NEEDS_OPEN = {"overnight_momentum_12_1", "intraday_momentum_12_1", "overnight_share"}

# Factors whose RANKING is provably unchanged by a currency conversion, because
# the conversion applies the same monotonic map to every name. Ranking these in
# USD is a no-op; only the resulting return stream differs. See to_usd.
FX_RANK_INVARIANT = {
    "momentum_12_1", "momentum_6_1", "short_term_reversal",
    "overnight_momentum_12_1", "intraday_momentum_12_1",
}


# --------------------------------------------------------------------------
# Portfolio construction
# --------------------------------------------------------------------------

def rebalance_dates(index, frequency="M"):
    """Last trading day of each period present in the index."""
    frame = pd.Series(index, index=index)
    if frequency in ("D", "daily"):
        return list(index)
    rule = {"W": "W", "weekly": "W", "M": "ME", "monthly": "ME",
            "Q": "QE", "quarterly": "QE"}.get(frequency, "ME")
    return list(frame.resample(rule).last().dropna())


def target_weights(factor_row, prices_row, top_quantile=0.2, bottom_quantile=0.2,
                   long_only=False, min_names=10):
    """Equal-weight the top of the ranking against the bottom.

    Weights are normalised so gross exposure is 1.0, which keeps the cost and
    volatility scaling below interpretable.
    """
    eligible = factor_row[factor_row.notna() & prices_row.notna() & (prices_row > 0)]
    if eligible.size < min_names:
        return None

    ordered = eligible.sort_values(ascending=False)
    count = max(int(round(len(ordered) * top_quantile)), 1)
    longs = ordered.index[:count]

    weights = pd.Series(0.0, index=factor_row.index)
    if long_only:
        weights[longs] = 1.0 / len(longs)
        return weights

    short_count = max(int(round(len(ordered) * bottom_quantile)), 1)
    shorts = ordered.index[-short_count:]
    if set(longs) & set(shorts):
        return None
    weights[longs] = 0.5 / len(longs)
    weights[shorts] = -0.5 / len(shorts)
    return weights


def run(prices, factor=None, factor_name="momentum_12_1", frequency="M",
        top_quantile=0.2, bottom_quantile=0.2, long_only=False,
        cost_bps=20.0, vol_target=None, max_leverage=3.0, min_names=10,
        vol_lookback=750, panels=None, n_trials=1):
    """Walk the rebalance dates and accrue daily portfolio returns.

    Returns a dict with the daily net/gross return series, the weight history
    and diagnostics. `vol_target` (annualised, e.g. 0.15) turns on volatility
    scaling driven by HAR-RV on the strategy's own past returns.
    """
    prices = prices.sort_index()
    if factor is None:
        factor = FACTORS[factor_name](prices, panels)
    asset_returns = prices.pct_change()

    schedule = [d for d in rebalance_dates(prices.index, frequency)
                if d in prices.index]
    if len(schedule) < 6:
        raise ValueError(
            f"Only {len(schedule)} rebalance dates; fetch a longer period or "
            "rebalance more often."
        )

    dates = prices.index
    held = pd.Series(0.0, index=prices.columns)
    scale = 1.0

    daily_gross, daily_net, daily_index = [], [], []
    weight_history, turnover_history, scale_history = {}, [], []
    skipped = 0

    for position, rebalance_date in enumerate(schedule[:-1]):
        desired = target_weights(
            factor.loc[rebalance_date], prices.loc[rebalance_date],
            top_quantile, bottom_quantile, long_only, min_names,
        )
        if desired is None:
            skipped += 1
            continue

        if vol_target is not None:
            scale = _volatility_scale(
                daily_net, vol_target, max_leverage, vol_lookback
            )
        desired = desired * scale

        # Turnover against what is actually held after drifting, not against
        # last period's target: that is what a broker would charge.
        turnover = float((desired - held).abs().sum())
        turnover_history.append(turnover)
        scale_history.append(scale)
        weight_history[rebalance_date] = desired

        cost = turnover * cost_bps / 10_000.0
        held = desired.copy()

        # Hold to the next rebalance. Returns are strictly AFTER the date the
        # weights were formed on.
        window = dates[(dates > rebalance_date) & (dates <= schedule[position + 1])]
        for day_number, day in enumerate(window):
            day_returns = asset_returns.loc[day].reindex(held.index).fillna(0.0)
            gross = float((held * day_returns).sum())
            daily_gross.append(gross)
            daily_net.append(gross - (cost if day_number == 0 else 0.0))
            daily_index.append(day)
            # Positions drift with prices between rebalances.
            held = held * (1.0 + day_returns)

    if not daily_net:
        raise ValueError("No holding periods produced returns.")

    net = pd.Series(daily_net, index=pd.DatetimeIndex(daily_index))
    gross = pd.Series(daily_gross, index=pd.DatetimeIndex(daily_index))
    return {
        "net_returns": net,
        "gross_returns": gross,
        "weights": pd.DataFrame(weight_history).T.sort_index(),
        "turnover": turnover_history,
        "scale": scale_history,
        "rebalances": len(turnover_history),
        "skipped_rebalances": skipped,
        "summary": summarize_portfolio(net, gross, turnover_history, frequency,
                                       cost_bps, scale_history, n_trials),
    }


def _volatility_scale(past_returns, vol_target, max_leverage, lookback):
    """Scale exposure by target / predicted volatility, using HAR-RV.

    This is where the volatility result earns its keep. It does not improve
    returns; it stabilises them, and a steadier return stream is a higher
    Sharpe for the same edge.
    """
    if len(past_returns) < 120:
        return 1.0
    recent = np.asarray(past_returns[-lookback:], dtype=float)
    daily_vol = np.abs(recent) * np.sqrt(TRADING_DAYS)
    daily_vol = daily_vol[np.isfinite(daily_vol) & (daily_vol > 0)]
    if daily_vol.size < 60:
        return 1.0

    predicted, _ = vol_models.make_har_forecaster()(daily_vol, 1)
    predicted = float(predicted[0])
    if not np.isfinite(predicted) or predicted <= 1e-6:
        return 1.0
    return float(np.clip(vol_target / predicted, 0.0, max_leverage))


def summarize_portfolio(net, gross, turnover, frequency, cost_bps, scale_history,
                        n_trials=1):
    periods = TRADING_DAYS
    net_values = net.to_numpy(float)
    gross_values = gross.to_numpy(float)

    annual_turnover = 0.0
    if turnover:
        per_year = {"D": TRADING_DAYS, "W": 52, "weekly": 52,
                    "M": 12, "monthly": 12, "Q": 4}.get(frequency, 12)
        annual_turnover = float(np.mean(turnover)) * per_year

    years = len(net) / periods if len(net) else float("nan")
    total = float(np.prod(1.0 + net_values)) if net_values.size else float("nan")
    cagr = (total ** (1 / years) - 1.0) if years and years > 0 and total > 0 else float("nan")

    # Deflate the Sharpe by how many configurations have been tried. A strategy
    # search is a maximisation over noise: try enough factors, frequencies and
    # quantiles and the best one looks good whether or not anything is there.
    # The formula wants a PER-PERIOD Sharpe and the return distribution's shape,
    # not an annualised number.
    annual_sharpe = metrics.sharpe_ratio(net_values, periods)
    deflated = float("nan")
    probabilistic = float("nan")
    if np.isfinite(annual_sharpe) and net_values.size > 8:
        per_period = annual_sharpe / np.sqrt(periods)
        skew = float(stats.skew(net_values))
        kurtosis = float(stats.kurtosis(net_values, fisher=False))
        deflated = metrics.deflated_sharpe_ratio(
            per_period, n_trials=max(int(n_trials), 1), n_obs=net_values.size,
            skew=skew, kurtosis=kurtosis,
        )
        # The same statistic against a single trial: the gap between them is
        # exactly what the search cost you.
        probabilistic = metrics.deflated_sharpe_ratio(
            per_period, n_trials=1, n_obs=net_values.size,
            skew=skew, kurtosis=kurtosis,
        )

    return {
        "n_days": int(len(net)),
        "years": years,
        "cagr_net": cagr,
        "sharpe_net": annual_sharpe,
        "deflated_sharpe": deflated,
        "probabilistic_sharpe": probabilistic,
        "n_trials": int(n_trials),
        "return_skew": float(stats.skew(net_values)) if net_values.size > 8 else float("nan"),
        "return_kurtosis": float(stats.kurtosis(net_values, fisher=False)) if net_values.size > 8 else float("nan"),
        "sharpe_gross": metrics.sharpe_ratio(gross_values, periods),
        "max_drawdown": metrics.max_drawdown(net_values),
        "hit_rate": metrics.hit_rate(net_values),
        "profit_factor": metrics.profit_factor(net_values),
        "turnover_per_rebalance": float(np.mean(turnover)) if turnover else float("nan"),
        "turnover_annual": annual_turnover,
        "cost_bps": float(cost_bps),
        "avg_leverage": float(np.mean(scale_history)) if scale_history else 1.0,
        "cost_drag_annual": annual_turnover * cost_bps / 10_000.0,
    }


def information_coefficient(prices, factor, frequency="M"):
    """Rank correlation between the factor and the return that follows it.

    The cleanest read on whether a factor knows anything, independent of how a
    portfolio is built from it.
    """
    schedule = [d for d in rebalance_dates(prices.index, frequency) if d in prices.index]
    coefficients = []
    for position, date in enumerate(schedule[:-1]):
        nxt = schedule[position + 1]
        signal = factor.loc[date]
        realised = prices.loc[nxt] / prices.loc[date] - 1.0
        both = pd.concat([signal, realised], axis=1).dropna()
        if len(both) >= 8:
            value = metrics.information_coefficient(
                both.iloc[:, 0].to_numpy(float), both.iloc[:, 1].to_numpy(float))
            if np.isfinite(value):
                coefficients.append(value)
    if not coefficients:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "ic_t_stat": float("nan"), "n_periods": 0}
    array = np.asarray(coefficients, dtype=float)
    return {
        "ic_mean": float(array.mean()),
        "ic_std": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
        # An IC that is positive on average but noisy is not a factor.
        "ic_t_stat": float(array.mean() / (array.std(ddof=1) / np.sqrt(array.size)))
                     if array.size > 1 and array.std(ddof=1) > 0 else float("nan"),
        "n_periods": int(array.size),
    }
