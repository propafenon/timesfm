"""Forecast evaluation metrics.

Split into three groups, deliberately:

  statistical  - is the forecast close to the truth, relative to a baseline
  distributional - is the predicted *distribution* honest
  economic     - would trading it have made money after costs

A model can win on the first and lose on the last, which is why all three are
reported side by side. Everything here is pure numpy/scipy: no dependency on
statsforecast, arch or vectorbt.
"""

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def _clean_pair(y_true, y_pred):
    """Drop positions where either side is missing."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


# --------------------------------------------------------------------------
# Statistical accuracy
# --------------------------------------------------------------------------

def mae(y_true, y_pred):
    y_true, y_pred = _clean_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if y_true.size else float('nan')


def rmse(y_true, y_pred):
    y_true, y_pred = _clean_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size else float('nan')


def smape(y_true, y_pred):
    """Symmetric MAPE in percent. Undefined where both sides are zero."""
    y_true, y_pred = _clean_pair(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def naive_scale(y_train, seasonality=1):
    """Denominator for MASE: in-sample MAE of the seasonal naive forecast.

    This is the number that makes an error scale-free and comparable across
    tickers. Returns nan when the training window is too short to define it.
    """
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_train = y_train[np.isfinite(y_train)]
    if y_train.size <= seasonality:
        return float('nan')
    diffs = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = float(np.mean(diffs))
    return scale if scale > 0 else float('nan')


def mase(y_true, y_pred, y_train, seasonality=1):
    """Mean Absolute Scaled Error.

    < 1 beats the naive forecast, >= 1 does not. On daily equity closes a value
    at or just above 1.0 is the norm, and is the result you should expect until
    proven otherwise.
    """
    scale = naive_scale(y_train, seasonality)
    if not np.isfinite(scale):
        return float('nan')
    error = mae(y_true, y_pred)
    return float(error / scale) if np.isfinite(error) else float('nan')


def directional_accuracy(y_true, y_pred, anchor):
    """Fraction of steps where the predicted direction from `anchor` was right.

    Flat predictions count as misses. 0.5 is a coin flip; anything at or below
    that is worthless as a trading signal no matter how small the MAE.
    """
    y_true, y_pred = _clean_pair(y_true, y_pred)
    if not y_true.size or not np.isfinite(anchor):
        return float('nan')
    true_dir = np.sign(y_true - anchor)
    pred_dir = np.sign(y_pred - anchor)
    scored = true_dir != 0
    if not scored.any():
        return float('nan')
    return float(np.mean(true_dir[scored] == pred_dir[scored]))


def information_coefficient(pred_returns, true_returns):
    """Spearman rank correlation between predicted and realised returns."""
    pred_returns, true_returns = _clean_pair(pred_returns, true_returns)
    if pred_returns.size < 3:
        return float('nan')
    if np.all(pred_returns == pred_returns[0]) or np.all(true_returns == true_returns[0]):
        return float('nan')
    return float(stats.spearmanr(pred_returns, true_returns).statistic)


# --------------------------------------------------------------------------
# Distributional quality
# --------------------------------------------------------------------------

def pinball_loss(y_true, q_pred, level):
    """Quantile (pinball) loss at one level in (0, 1). Lower is better."""
    y_true, q_pred = _clean_pair(y_true, q_pred)
    if not y_true.size:
        return float('nan')
    delta = y_true - q_pred
    return float(np.mean(np.maximum(level * delta, (level - 1.0) * delta)))


def crps_from_quantiles(y_true, quantiles, levels):
    """CRPS approximated from a discrete quantile spread.

    CRPS = 2 * integral of pinball loss over levels; with K evenly spaced levels
    that integral is approximated by the mean, giving 2 * mean(pinball).
    """
    levels = np.asarray(levels, dtype=float).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim != 2 or quantiles.shape[1] != levels.size:
        raise ValueError("quantiles must be (n_obs, n_levels) matching levels")
    losses = [pinball_loss(y_true, quantiles[:, i], level)
              for i, level in enumerate(levels)]
    losses = [l for l in losses if np.isfinite(l)]
    return float(2.0 * np.mean(losses)) if losses else float('nan')


def interval_coverage(y_true, lower, upper):
    """Realised coverage of a predicted interval.

    Compare against the nominal level. An 80% interval covering 45% of the time
    means the model is overconfident and any sizing built on it is wrong.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if not mask.any():
        return float('nan')
    inside = (y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])
    return float(np.mean(inside))


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------

def diebold_mariano(errors_a, errors_b, horizon=1, loss="squared"):
    """Test whether model A's forecast errors differ from model B's.

    Uses the Harvey-Leybourne-Newbold small-sample correction and a Student-t
    reference distribution, which matters at the sample sizes a walk-forward
    over a few years actually produces.

    Returns (statistic, p_value). Negative statistic favours A.
    """
    errors_a, errors_b = _clean_pair(errors_a, errors_b)
    n = errors_a.size
    if n < 8:
        return float('nan'), float('nan')

    if loss == "squared":
        d = errors_a ** 2 - errors_b ** 2
    elif loss == "absolute":
        d = np.abs(errors_a) - np.abs(errors_b)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d_bar = float(np.mean(d))
    demeaned = d - d_bar

    # Newey-West long-run variance, truncated at horizon - 1 lags: multi-step
    # forecast errors are autocorrelated by construction.
    gamma0 = float(np.mean(demeaned ** 2))
    lrv = gamma0
    for lag in range(1, int(horizon)):
        if lag >= n:
            break
        gamma = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
        lrv += 2.0 * gamma
    if lrv <= 0:
        return float('nan'), float('nan')

    dm = d_bar / np.sqrt(lrv / n)

    h = int(horizon)
    correction = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if correction <= 0:
        return float('nan'), float('nan')
    dm_star = dm * np.sqrt(correction)

    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_star), df=n - 1))
    return float(dm_star), float(p)


def deflated_sharpe_ratio(sharpe_observed, n_trials, n_obs, skew=0.0,
                          kurtosis=3.0, sharpe_std=None):
    """Probability the observed Sharpe is real given how many configs were tried.

    The point of this program is trying configurations until one looks good, so
    the naive Sharpe of the winner is biased upward. Bailey & Lopez de Prado's
    deflation asks: given `n_trials` attempts, how likely is a Sharpe this high
    under the null of no skill? Below ~0.95 means "probably backtest noise".

    IMPORTANT: `sharpe_observed` must be per-period, matching `n_obs` - not
    annualised. Pass an annualised figure and the answer is meaningless.

    `sharpe_std` is the spread of Sharpe estimates across the trials. When it is
    not supplied, the estimator's own standard error is used, which assumes the
    trials are independent draws under the null.
    """
    if n_trials < 1 or n_obs < 4 or not np.isfinite(sharpe_observed):
        return float('nan')

    # Standard error of the Sharpe estimator, adjusted for non-normal returns.
    variance = (1.0
                - skew * sharpe_observed
                + ((kurtosis - 1.0) / 4.0) * sharpe_observed ** 2) / (n_obs - 1)
    if variance <= 0:
        return float('nan')
    se = np.sqrt(variance)

    spread = se if sharpe_std is None else float(sharpe_std)
    if spread <= 0:
        return float('nan')

    if n_trials == 1:
        sr0 = 0.0
    else:
        # Expected maximum of n_trials independent null Sharpes, on the scale of
        # the estimates themselves. Omitting this scaling compares a Sharpe to a
        # raw z-quantile, which deflates everything to zero.
        q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr0 = spread * ((1 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2)

    return float(stats.norm.cdf((sharpe_observed - sr0) / se))


# --------------------------------------------------------------------------
# Economic
# --------------------------------------------------------------------------

PERIODS_PER_YEAR = {
    "1m": 252 * 390, "5m": 252 * 78, "15m": 252 * 26, "30m": 252 * 13,
    "1h": 252 * 7, "1d": 252, "1wk": 52, "1mo": 12,
}


def sharpe_ratio(returns, periods_per_year=252):
    returns = np.asarray(returns, dtype=float).reshape(-1)
    returns = returns[np.isfinite(returns)]
    if returns.size < 2:
        return float('nan')
    sd = float(np.std(returns, ddof=1))
    # An exact == 0 test misses constant series: floating-point residue leaves
    # sd around 1e-19, which turns the ratio into a nonsense 1e15 Sharpe.
    if not np.isfinite(sd) or sd <= 1e-15 * max(1.0, float(np.max(np.abs(returns)))):
        return float('nan')
    return float(np.mean(returns) / sd * np.sqrt(periods_per_year))


def max_drawdown(returns):
    """Worst peak-to-trough decline of the cumulative curve, as a fraction."""
    returns = np.asarray(returns, dtype=float).reshape(-1)
    returns = returns[np.isfinite(returns)]
    if not returns.size:
        return float('nan')
    curve = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve / peak - 1.0))


def profit_factor(returns):
    returns = np.asarray(returns, dtype=float).reshape(-1)
    returns = returns[np.isfinite(returns)]
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float('inf') if gains > 0 else float('nan')
    return float(gains / losses)


def hit_rate(returns):
    returns = np.asarray(returns, dtype=float).reshape(-1)
    returns = returns[np.isfinite(returns)]
    nonzero = returns[returns != 0]
    return float(np.mean(nonzero > 0)) if nonzero.size else float('nan')


def signal_pnl(positions, realised_returns, cost_bps=0.0):
    """Net returns of a position series, charging cost on turnover.

    Costs are not a rounding error on Borsa Istanbul, so a strategy is only
    reported net. `positions` must already be lagged: position[i] is what you
    held going into the bar that produced realised_returns[i].
    """
    positions = np.asarray(positions, dtype=float).reshape(-1)
    realised_returns = np.asarray(realised_returns, dtype=float).reshape(-1)
    if positions.shape != realised_returns.shape:
        raise ValueError("positions and returns must align")

    gross = positions * realised_returns
    turnover = np.abs(np.diff(np.concatenate(([0.0], positions))))
    costs = turnover * (cost_bps / 10_000.0)
    return gross - costs


def turnover(positions):
    positions = np.asarray(positions, dtype=float).reshape(-1)
    if not positions.size:
        return float('nan')
    return float(np.mean(np.abs(np.diff(np.concatenate(([0.0], positions))))))
