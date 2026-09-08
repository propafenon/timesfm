"""Target-space transforms.

Forecasting price levels is the wrong problem, for three reasons:

  Stationarity - prices wander, returns do not. A model fitted on levels spends
  its capacity tracking the trend.
  Comparability - a MASE computed on levels is contaminated by drift: the scale
  is measured on an old window while prices move away from it, so a naive
  forecast can score 1.33 instead of 1.0 without anything being wrong.
  Relevance - the trading decision depends on the change, not the level.

So: model returns, then reconstruct a price path. Reconstruction is what keeps
level models and return models comparable, because both are finally scored in
price space against the same actuals.

The naive forecast survives the round trip unchanged: predicting a zero return
reconstructs to "price stays put", which is exactly the price-space naive. That
equivalence is what makes the two spaces' skill numbers mean the same thing.
"""

import numpy as np

PRICE = "price"
LOG_RETURN = "log_return"
SIMPLE_RETURN = "simple_return"

TARGET_SPACES = (PRICE, LOG_RETURN, SIMPLE_RETURN)

SPACE_LABELS = {
    PRICE: "Price level",
    LOG_RETURN: "Log returns",
    SIMPLE_RETURN: "Simple returns",
}


def encode(prices, space=LOG_RETURN):
    """Price context -> model context. Returns are one element shorter."""
    values = np.asarray(prices, dtype=float).reshape(-1)

    if space == PRICE:
        return values

    if values.size < 2:
        raise ValueError("at least two observations are needed to form returns")

    if space == LOG_RETURN:
        if np.any(values <= 0):
            raise ValueError(
                "log returns need strictly positive prices; use simple returns "
                "for series that touch or cross zero"
            )
        return np.diff(np.log(values))

    if space == SIMPLE_RETURN:
        previous = values[:-1]
        if np.any(previous == 0):
            raise ValueError("simple returns are undefined across a zero price")
        return values[1:] / previous - 1.0

    raise ValueError(f"unknown target space: {space}")


def decode(forecast, anchor_price, space=LOG_RETURN):
    """Model output -> price path, compounding forward from `anchor_price`."""
    values = np.asarray(forecast, dtype=float).reshape(-1)

    if space == PRICE:
        return values

    anchor_price = float(anchor_price)

    if space == LOG_RETURN:
        return anchor_price * np.exp(np.cumsum(values))

    if space == SIMPLE_RETURN:
        return anchor_price * np.cumprod(1.0 + values)

    raise ValueError(f"unknown target space: {space}")


def realised(prices, anchor_price, space=LOG_RETURN):
    """The realised path expressed in `space`, for scoring without reconstructing."""
    values = np.asarray(prices, dtype=float).reshape(-1)
    if space == PRICE:
        return values
    full = np.concatenate(([float(anchor_price)], values))
    return encode(full, space)


def naive_forecast(horizon, space=LOG_RETURN, last_price=None):
    """What "no change" looks like in each space.

    Zero in return space; the last price repeated in price space. These
    reconstruct to the same path, which is the invariant the round trip relies
    on and which tests/test_transforms.py pins.
    """
    if space == PRICE:
        if last_price is None:
            raise ValueError("price space needs last_price")
        return np.full(horizon, float(last_price))
    return np.zeros(horizon)
