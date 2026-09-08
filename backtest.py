"""Rolling-origin backtesting.

One forecast from one origin is an anecdote. This walks an origin through
history, forecasting `horizon` steps from each, and produces one row per
(origin, step). Every metric downstream is a group-by over those rows.

The invariant that makes it a backtest rather than a demo: the context handed
to a model is always `values[:origin + 1]`, never a slice that reaches past the
origin. Forecast dates come from the observed index itself, so real trading
sessions are used and exchange holidays never appear as predicted bars.
"""

import numpy as np
import pandas as pd

import metrics

DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# --------------------------------------------------------------------------
# Baseline forecasters
#
# model_fn(context, horizon) -> (point_forecast, quantiles or None)
#
# If TimesFM cannot beat these, that is the finding, and it is worth far more
# than a low MAE with nothing to compare it to.
# --------------------------------------------------------------------------

def naive(context, horizon):
    """Tomorrow equals today. The benchmark everything is scaled against."""
    return np.full(horizon, float(context[-1])), None


def seasonal_naive(season_length):
    def _forecast(context, horizon):
        if len(context) < season_length:
            return naive(context, horizon)
        season = np.asarray(context[-season_length:], dtype=float)
        return season[np.arange(horizon) % season_length], None
    _forecast.__name__ = f"seasonal_naive_{season_length}"
    return _forecast


def drift(context, horizon):
    """Extrapolate the straight line through the first and last observation."""
    context = np.asarray(context, dtype=float)
    if context.size < 2:
        return naive(context, horizon)
    slope = (context[-1] - context[0]) / (context.size - 1)
    return context[-1] + slope * np.arange(1, horizon + 1), None


def ses(alpha=0.2):
    """Simple exponential smoothing at a fixed smoothing level."""
    def _forecast(context, horizon):
        context = np.asarray(context, dtype=float)
        level = context[0]
        for value in context[1:]:
            level = alpha * value + (1 - alpha) * level
        return np.full(horizon, float(level)), None
    _forecast.__name__ = f"ses_{alpha}"
    return _forecast


def theta(alpha=0.2):
    """Theta method, via its equivalence to SES with half the trend as drift.

    Hyndman & Billah (2003) showed the classic Theta method is SES plus a drift
    of b/2, where b is the slope of an OLS fit on time. That is what this is.
    """
    def _forecast(context, horizon):
        context = np.asarray(context, dtype=float)
        if context.size < 3:
            return naive(context, horizon)

        time_index = np.arange(context.size, dtype=float)
        slope = float(np.polyfit(time_index, context, 1)[0])

        level = context[0]
        for value in context[1:]:
            level = alpha * value + (1 - alpha) * level

        return level + (slope / 2.0) * np.arange(1, horizon + 1), None
    _forecast.__name__ = f"theta_{alpha}"
    return _forecast


BASELINES = {
    "naive": naive,
    "drift": drift,
    "ses": ses(0.2),
    "theta": theta(0.2),
    "seasonal_naive_5": seasonal_naive(5),
}


def try_statsforecast_baselines(season_length=5):
    """Upgrade to Nixtla's implementations when they are installed.

    Optional on purpose: the built-ins above cover the same ground with no
    dependency, but statsforecast adds AutoARIMA and AutoETS.
    """
    try:
        from statsforecast.models import AutoARIMA, AutoETS
    except ImportError:
        return {}

    def _wrap(model_cls, name):
        def _forecast(context, horizon):
            model = model_cls(season_length=season_length)
            model.fit(np.asarray(context, dtype=float))
            return np.asarray(model.predict(h=horizon)["mean"], dtype=float), None
        _forecast.__name__ = name
        return _forecast

    return {"auto_arima": _wrap(AutoARIMA, "auto_arima"),
            "auto_ets": _wrap(AutoETS, "auto_ets")}


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

def plan_origins(n_observations, context_len, horizon, step, min_context=None):
    """Origin indices with a full horizon of truth available after each."""
    min_context = min_context or min(context_len, 64)
    first = max(min_context - 1, 0)
    last = n_observations - horizon - 1
    if last < first:
        return []
    return list(range(first, last + 1, max(int(step), 1)))


def walk_forward(values, dates, model_fn, context_len, horizon, step,
                 mode="sliding", min_context=None, quantile_levels=None,
                 progress_cb=None, should_stop=None):
    """Roll an origin through history. Returns a list of per-step dicts."""
    values = np.asarray(values, dtype=float).reshape(-1)
    dates = pd.DatetimeIndex(dates)
    if values.size != dates.size:
        raise ValueError("values and dates must be the same length")

    origins = plan_origins(values.size, context_len, horizon, step, min_context)
    if not origins:
        raise ValueError(
            f"Not enough data: {values.size} bars cannot support a "
            f"{context_len}-bar context and a {horizon}-step horizon."
        )

    rows = []
    for position, origin in enumerate(origins):
        if should_stop is not None and should_stop():
            break

        context = values[:origin + 1]          # never reaches past the origin
        if mode == "sliding":
            context = context[-context_len:]

        point, quantiles = model_fn(context, horizon)
        point = np.asarray(point, dtype=float).reshape(-1)[:horizon]

        truth = values[origin + 1: origin + 1 + horizon]
        target_dates = dates[origin + 1: origin + 1 + horizon]
        anchor_value = float(values[origin])

        # MASE denominator from the context only, so it is also leak-free.
        scale = metrics.naive_scale(context, seasonality=1)

        for step_index in range(min(horizon, truth.size, point.size)):
            row = {
                "origin_index": int(origin),
                "origin_date": dates[origin].isoformat(),
                "step": step_index + 1,
                "target_date": target_dates[step_index].isoformat(),
                "anchor_value": anchor_value,
                "y_true": float(truth[step_index]),
                "y_pred": float(point[step_index]),
                "naive_pred": anchor_value,
                "naive_scale": float(scale) if np.isfinite(scale) else None,
            }
            if quantiles is not None:
                levels = quantile_levels or DEFAULT_QUANTILE_LEVELS
                matrix = np.asarray(quantiles, dtype=float)
                if matrix.ndim == 2 and step_index < matrix.shape[0]:
                    row["quantiles"] = {
                        str(level): float(matrix[step_index, i])
                        for i, level in enumerate(levels)
                        if i < matrix.shape[1]
                    }
            rows.append(row)

        if progress_cb is not None:
            progress_cb(position + 1, len(origins))

    return rows


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def summarize(rows, interval="1d", cost_bps=10.0, n_trials=1):
    """Turn per-step rows into the numbers that decide whether this works."""
    if not rows:
        return {}

    frame = pd.DataFrame(rows)
    y_true = frame["y_true"].to_numpy(float)
    y_pred = frame["y_pred"].to_numpy(float)
    naive_pred = frame["naive_pred"].to_numpy(float)

    scales = frame["naive_scale"].dropna().to_numpy(float)
    scale = float(np.mean(scales)) if scales.size else float("nan")

    model_mae = metrics.mae(y_true, y_pred)
    naive_mae = metrics.mae(y_true, naive_pred)

    summary = {
        "n_origins": int(frame["origin_index"].nunique()),
        "n_points": int(len(frame)),
        "mae": model_mae,
        "rmse": metrics.rmse(y_true, y_pred),
        "smape": metrics.smape(y_true, y_pred),
        "naive_mae": naive_mae,
        # The headline. >= 1 means the model does not beat "no change".
        "mase": float(model_mae / scale) if np.isfinite(scale) and scale > 0 else float("nan"),
        "naive_mase": float(naive_mae / scale) if np.isfinite(scale) and scale > 0 else float("nan"),
        "skill_vs_naive": float(1.0 - model_mae / naive_mae) if naive_mae > 0 else float("nan"),
        # Per-row anchors differ, so this cannot use the scalar-anchor helper.
        "directional_accuracy": _directional(frame),
    }

    # Is the difference from naive real, or sampling noise?
    horizon = int(frame["step"].max())
    stat, p_value = metrics.diebold_mariano(
        y_pred - y_true, naive_pred - y_true, horizon=horizon, loss="squared"
    )
    summary["dm_stat_vs_naive"] = stat
    summary["dm_pvalue_vs_naive"] = p_value

    # Pooling horizons hides the only thing that matters about them: error
    # grows with h. On a random walk a naive forecast pooled over h=1..5 scores
    # MASE ~1.6 purely because the h-step error grows like sqrt(h), while the
    # MASE denominator is a one-step scale. Always read the per-step table.
    summary["by_step"] = per_step(frame)
    if summary["by_step"]:
        summary["mase_step1"] = summary["by_step"][0]["mase"]
        summary["skill_step1"] = summary["by_step"][0]["skill_vs_naive"]

    summary.update(_distributional(frame))
    summary.update(_economic(frame, interval, cost_bps, n_trials))
    return summary


def per_step(frame):
    """Break every accuracy number out by horizon step."""
    out = []
    for step_value in sorted(frame["step"].unique()):
        chunk = frame[frame["step"] == step_value]
        y_true = chunk["y_true"].to_numpy(float)
        y_pred = chunk["y_pred"].to_numpy(float)
        y_naive = chunk["naive_pred"].to_numpy(float)

        scales = chunk["naive_scale"].dropna().to_numpy(float)
        scale = float(np.mean(scales)) if scales.size else float("nan")

        step_mae = metrics.mae(y_true, y_pred)
        naive_step_mae = metrics.mae(y_true, y_naive)
        out.append({
            "step": int(step_value),
            "n": int(len(chunk)),
            "mae": step_mae,
            "naive_mae": naive_step_mae,
            "mase": float(step_mae / scale) if np.isfinite(scale) and scale > 0 else float("nan"),
            "skill_vs_naive": float(1.0 - step_mae / naive_step_mae) if naive_step_mae > 0 else float("nan"),
            "directional_accuracy": _directional(chunk),
        })
    return out


def _directional(frame):
    anchors = frame["anchor_value"].to_numpy(float)
    true_dir = np.sign(frame["y_true"].to_numpy(float) - anchors)
    pred_dir = np.sign(frame["y_pred"].to_numpy(float) - anchors)
    scored = true_dir != 0
    if not scored.any():
        return float("nan")
    return float(np.mean(true_dir[scored] == pred_dir[scored]))


def _distributional(frame):
    if "quantiles" not in frame.columns:
        return {}
    present = frame["quantiles"].dropna()
    if present.empty:
        return {}

    levels = sorted(float(k) for k in present.iloc[0].keys())
    matrix = np.array([[row[str(level)] for level in levels] for row in present])
    y_true = frame.loc[present.index, "y_true"].to_numpy(float)

    out = {"crps": metrics.crps_from_quantiles(y_true, matrix, levels)}
    for low, high, nominal in ((0.1, 0.9, 0.8), (0.2, 0.8, 0.6)):
        if low in levels and high in levels:
            out[f"coverage_{int(nominal * 100)}"] = metrics.interval_coverage(
                y_true, matrix[:, levels.index(low)], matrix[:, levels.index(high)]
            )
            out[f"coverage_{int(nominal * 100)}_nominal"] = nominal
    return out


def _economic(frame, interval, cost_bps, n_trials):
    """Trade the one-step forecast and see whether it survives costs."""
    step_one = frame[frame["step"] == 1].sort_values("origin_index")
    if len(step_one) < 8:
        return {}

    anchors = step_one["anchor_value"].to_numpy(float)
    truth = step_one["y_true"].to_numpy(float)
    predicted = step_one["y_pred"].to_numpy(float)

    valid = anchors > 0
    if valid.sum() < 8:
        return {}
    anchors, truth, predicted = anchors[valid], truth[valid], predicted[valid]

    realised = truth / anchors - 1.0
    expected = predicted / anchors - 1.0
    positions = np.sign(expected)          # long / flat / short, no sizing

    net = metrics.signal_pnl(positions, realised, cost_bps=cost_bps)
    periods = metrics.PERIODS_PER_YEAR.get(interval, 252)
    sharpe = metrics.sharpe_ratio(net, periods_per_year=periods)

    per_period_sharpe = (
        sharpe / np.sqrt(periods) if np.isfinite(sharpe) else float("nan")
    )
    deflated = metrics.deflated_sharpe_ratio(
        per_period_sharpe, n_trials=max(int(n_trials), 1), n_obs=net.size
    ) if np.isfinite(per_period_sharpe) else float("nan")

    return {
        "ic": metrics.information_coefficient(expected, realised),
        "sharpe_net": sharpe,
        "sharpe_gross": metrics.sharpe_ratio(
            metrics.signal_pnl(positions, realised, cost_bps=0.0),
            periods_per_year=periods),
        "max_drawdown": metrics.max_drawdown(net),
        "hit_rate": metrics.hit_rate(net),
        "profit_factor": metrics.profit_factor(net),
        "turnover": metrics.turnover(positions),
        "cost_bps": float(cost_bps),
        "deflated_sharpe": deflated,
        "n_trials_assumed": int(n_trials),
    }


def compare_to_baselines(values, dates, model_rows, context_len, horizon, step,
                         mode="sliding", min_context=None, baselines=None,
                         interval="1d", cost_bps=10.0):
    """Run each baseline over the same origins and score them side by side."""
    table = {}
    if model_rows:
        table["timesfm"] = summarize(model_rows, interval=interval, cost_bps=cost_bps)

    chosen = baselines if baselines is not None else BASELINES
    for name, model_fn in chosen.items():
        rows = walk_forward(values, dates, model_fn, context_len, horizon, step,
                            mode=mode, min_context=min_context)
        table[name] = summarize(rows, interval=interval, cost_bps=cost_bps)
    return table
