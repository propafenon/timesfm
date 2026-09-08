import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import transforms as tf, backtest as bt, metrics as m

rng = np.random.default_rng(11)
prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, 400)))

# ---- round trip is exact in both return spaces -------------------------
for space in (tf.LOG_RETURN, tf.SIMPLE_RETURN):
    enc = tf.encode(prices, space)
    assert enc.size == prices.size - 1, (space, enc.size)
    back = tf.decode(enc, prices[0], space)
    np.testing.assert_allclose(back, prices[1:], rtol=1e-11)
    print(f"round trip exact: {space}")

# price space is a no-op
np.testing.assert_allclose(tf.decode(tf.encode(prices, tf.PRICE), 0.0, tf.PRICE), prices)
print("round trip exact: price")

# ---- THE invariant: naive is the same forecast in every space ----------
last = float(prices[-1])
flat = tf.decode(tf.naive_forecast(5, tf.PRICE, last_price=last), last, tf.PRICE)
for space in (tf.LOG_RETURN, tf.SIMPLE_RETURN):
    via_returns = tf.decode(tf.naive_forecast(5, space), last, space)
    np.testing.assert_allclose(via_returns, flat, rtol=1e-12)
np.testing.assert_allclose(flat, np.full(5, last))
print("INVARIANT: zero-return naive reconstructs to price-space naive, exactly")

# ---- guards ------------------------------------------------------------
for bad in ([100.0, 0.0, 50.0], [100.0, -5.0]):
    try:
        tf.encode(bad, tf.LOG_RETURN); raise AssertionError("should have refused")
    except ValueError: pass
print("log returns refuse non-positive prices")

# ---- wrapper keeps the engine in price space ---------------------------
dates = pd.bdate_range("2023-01-02", periods=400)
level_rows = bt.walk_forward(prices, dates, bt.naive, 128, 5, 7)
wrapped = bt.in_return_space(bt.naive, tf.LOG_RETURN)
ret_rows = bt.walk_forward(prices, dates, wrapped, 128, 5, 7)

assert len(level_rows) == len(ret_rows)
# Same origins, same truths, same naive reference: only y_pred may differ.
for a, b in zip(level_rows, ret_rows):
    assert a["origin_index"] == b["origin_index"] and a["target_date"] == b["target_date"]
    assert a["y_true"] == b["y_true"] and a["naive_pred"] == b["naive_pred"]
print("wrapper preserves origins, truths and the naive reference")

# Wrapped naive is momentum, so it must NOT equal the level naive.
assert not np.allclose([r["y_pred"] for r in level_rows], [r["y_pred"] for r in ret_rows])
print("wrapped naive is momentum, as documented - not the same model")

# ---- but a zero-return model DOES reproduce price naive exactly --------
def zero_return(context, horizon):
    return np.zeros(horizon), None
zero_rows = bt.walk_forward(prices, dates, bt.in_return_space(zero_return, tf.LOG_RETURN), 128, 5, 7)
np.testing.assert_allclose([r["y_pred"] for r in zero_rows],
                           [r["y_pred"] for r in level_rows], rtol=1e-11)
print("INVARIANT holds end to end: zero-return model == price naive through the engine")

# ---- return-space metrics --------------------------------------------
s = bt.summarize(zero_rows, interval="1d")
assert abs(s["skill_returns"]) < 1e-9, s["skill_returns"]
print(f"zero-return model has exactly zero return-space skill: {s['skill_returns']:.1e}")

# Price-space MASE is drift-contaminated; return-space skill is not.
print(f"  price MASE h1={s['mase_step1']:.3f} (inflated by drift)  "
      f"return skill={s['skill_returns']:+.1e} (clean)")

# ---- quantiles reconstruct per path -----------------------------------
levels = list(bt.DEFAULT_QUANTILE_LEVELS)
def qzero(context, horizon):
    from scipy import stats as st
    q = np.array([[st.norm.ppf(l) * 0.01 * np.sqrt(h + 1) for l in levels] for h in range(horizon)])
    return np.zeros(horizon), q
qrows = bt.walk_forward(prices, dates, bt.in_return_space(qzero, tf.LOG_RETURN),
                        128, 5, 7, quantile_levels=levels)
first = qrows[0]["quantiles"]
vals = [first[str(l)] for l in levels]
assert vals == sorted(vals), "reconstructed quantiles crossed"
assert all(v > 0 for v in vals), "reconstructed prices must be positive"
qs = bt.summarize(qrows, interval="1d")
print(f"quantiles reconstruct monotonically; CRPS={qs['crps']:.4f}, "
      f"80% coverage={qs['coverage_80']:.1%}")

print("\nALL TRANSFORM TESTS PASSED")
