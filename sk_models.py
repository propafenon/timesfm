"""Support-vector and other scikit-learn forecasters.

The one design decision that matters here: THE MODEL IS FITTED ON RETURNS, NOT
PRICES. An SVR trained on lagged price levels to predict the next price scores a
spectacular R-squared and draws a chart that tracks the market beautifully. It
has learned that tomorrow's price is roughly today's price, which is the naive
forecast wearing a kernel. It is the single most common way this gets done
wrong, and the resulting picture is convincing enough that people ship it.

Fitting on log returns removes that crutch: the model has to predict the change,
which is the thing that is actually hard, and MASE against a naive baseline then
means what it says.

Everything is fitted from the context window alone, refitted at every origin, so
these slot into the walk-forward engine with the same leak-free guarantee as
TimesFM.
"""

import numpy as np

try:
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import TransformedTargetRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DEFAULT_LAGS = 10
DEFAULT_MAX_TRAIN = 1500


def _supervised(returns, covariates, lags, horizon):
    """Lagged-return design matrix and cumulative-return targets.

    X[t] holds the `lags` most recent returns up to t (plus covariate values at
    t); y[h][t] is the cumulative log return from t to t + h + 1. Targeting the
    cumulative change is direct multi-step: each horizon gets its own model
    rather than one model iterated, which stops errors compounding.
    """
    n = returns.size
    last = n - horizon
    if last <= lags:
        return None, None

    rows, targets = [], [[] for _ in range(horizon)]
    for t in range(lags, last):
        features = list(returns[t - lags:t])
        if covariates is not None:
            features.extend(covariates[:, t])
        rows.append(features)
        cumulative = 0.0
        for step in range(horizon):
            cumulative += returns[t + step]
            targets[step].append(cumulative)

    return np.asarray(rows, dtype=float), [np.asarray(t, dtype=float) for t in targets]


def make_svr_forecaster(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale",
                        lags=DEFAULT_LAGS, max_train=DEFAULT_MAX_TRAIN,
                        use_covariates=True):
    """A model_fn for the walk-forward engine. Takes prices, returns prices.

    Always consumes a price context and reconstructs a price path, whatever the
    app's "Model in" setting says, because the differencing happens internally.
    Wrapping it in `in_return_space` as well would difference twice.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is not installed; run: pip install scikit-learn")

    def _forecast(context, horizon, covariate_window=None):
        prices = np.asarray(context, dtype=float).reshape(-1)
        anchor = float(prices[-1])

        positive = prices[prices > 0]
        if positive.size < prices.size or prices.size < lags + horizon + 20:
            # Not enough usable history to fit anything; fall back to no change.
            return np.full(horizon, anchor), None

        returns = np.diff(np.log(prices))

        aligned_covariates = None
        if use_covariates and covariate_window is not None and covariate_window.size:
            matrix = np.asarray(covariate_window, dtype=float)
            # Returns are one shorter than prices, so drop the first column.
            if matrix.shape[1] == prices.size:
                matrix = matrix[:, 1:]
            if matrix.shape[1] == returns.size:
                aligned_covariates = np.nan_to_num(matrix, nan=0.0,
                                                   posinf=0.0, neginf=0.0)

        design, targets = _supervised(returns, aligned_covariates, lags, horizon)
        if design is None or design.shape[0] < 30:
            return np.full(horizon, anchor), None

        if design.shape[0] > max_train:
            # Keep the most recent rows: the nearest regime is the relevant one,
            # and an RBF SVR is roughly quadratic in sample count.
            design = design[-max_train:]
            targets = [t[-max_train:] for t in targets]

        latest = list(returns[-lags:])
        if aligned_covariates is not None:
            latest.extend(aligned_covariates[:, -1])
        latest = np.asarray(latest, dtype=float).reshape(1, -1)

        cumulative = np.zeros(horizon, dtype=float)
        for step in range(horizon):
            # The TARGET is scaled as well as the features. SVR's epsilon is an
            # insensitive tube measured in the units of y, and daily log returns
            # have a standard deviation around 0.01 - so a raw epsilon anywhere
            # near that swallows the entire signal and the model predicts a
            # constant. Scaling y means epsilon is expressed in standard
            # deviations, which is what its default is meant to be.
            model = TransformedTargetRegressor(
                regressor=make_pipeline(
                    StandardScaler(),
                    SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma),
                ),
                transformer=StandardScaler(),
            )
            model.fit(design, targets[step])
            cumulative[step] = float(model.predict(latest)[0])

        # Cumulative log returns -> a price path from the anchor.
        return anchor * np.exp(cumulative), None

    _forecast.wants_covariates = True
    _forecast.__name__ = f"svr_{kernel}_C{C}"
    return _forecast


def available_models(use_covariates=True):
    """Registry for the backtest tab. Empty when scikit-learn is absent."""
    if not SKLEARN_AVAILABLE:
        return {}
    return {
        "svr_rbf": make_svr_forecaster(kernel="rbf", C=1.0,
                                       use_covariates=use_covariates),
        "svr_linear": make_svr_forecaster(kernel="linear", C=1.0,
                                          use_covariates=use_covariates),
    }
