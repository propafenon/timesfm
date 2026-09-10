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

import metrics
import vol_models

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Factors. Higher rank = more attractive.
# --------------------------------------------------------------------------

def momentum_12_1(prices):
    """The classic: a year of momentum, skipping the last month's reversal.

    The most replicated anomaly in equities - and it lives at a monthly-to-
    annual horizon, not the one-day horizon this project has been testing.
    """
    return prices.shift(21) / prices.shift(252) - 1.0


def momentum_6_1(prices):
    return prices.shift(21) / prices.shift(126) - 1.0


def short_term_reversal(prices):
    """Last month's losers tend to bounce, so the sign is inverted."""
    return -(prices / prices.shift(21) - 1.0)


def low_volatility(prices, window=60):
    """Low-volatility names have historically outperformed on a risk-adjusted
    basis; negated so that calmer ranks higher."""
    returns = np.log(prices).diff()
    return -returns.rolling(window, min_periods=window).std()


def trend_vs_200d(prices):
    return prices / prices.rolling(200, min_periods=200).mean() - 1.0


FACTORS = {
    "momentum_12_1": momentum_12_1,
    "momentum_6_1": momentum_6_1,
    "short_term_reversal": short_term_reversal,
    "low_volatility": low_volatility,
    "trend_vs_200d": trend_vs_200d,
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
        vol_lookback=750):
    """Walk the rebalance dates and accrue daily portfolio returns.

    Returns a dict with the daily net/gross return series, the weight history
    and diagnostics. `vol_target` (annualised, e.g. 0.15) turns on volatility
    scaling driven by HAR-RV on the strategy's own past returns.
    """
    prices = prices.sort_index()
    if factor is None:
        factor = FACTORS[factor_name](prices)
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
                                       cost_bps, scale_history),
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


def summarize_portfolio(net, gross, turnover, frequency, cost_bps, scale_history):
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

    return {
        "n_days": int(len(net)),
        "years": years,
        "cagr_net": cagr,
        "sharpe_net": metrics.sharpe_ratio(net_values, periods),
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
