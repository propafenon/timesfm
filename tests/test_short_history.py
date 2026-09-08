"""Regression: a context longer than the available history.

ASELS.IS over period=1y is ~254 bars against a default context of 1056. The old
code padded time_series up to 1056, then used positions in that padded array to
index a 254-row DatetimeIndex, giving

    IndexError: index 843 is out of bounds for axis 0 with size 254

Padding was also the wrong answer on its own terms: 802 of 1056 values would
have been one repeated edge value, so the model would see a mostly flat line.
"""

import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    _root = tk.Tk(); _root.withdraw()
except Exception as exc:
    print(f"SKIPPED: Tk unavailable ({exc})"); raise SystemExit(0)

import db_manager as db
db.DB_DIR = tempfile.mkdtemp(); db.DB_PATH = os.path.join(db.DB_DIR, 'short.db')
import timesfm_forecaster as forecaster
import transforms as tf

app = forecaster.TimesFMApp(_root)

# Exactly the reported shape: one year of daily bars.
n = 254
idx = pd.bdate_range("2025-09-08", periods=n)
rng = np.random.default_rng(11)
close = 380 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
app.historical_data = forecaster.features.build_features(pd.DataFrame({
    "Open": close, "High": close * 1.01, "Low": close * 0.99,
    "Close": close, "Adj Close": close, "Volume": rng.integers(1e5, 5e6, n).astype(float),
}, index=idx))
app.historical_data["USD/TRY"] = np.linspace(35, 42, n)

app.tkr_var.set("ASELS.IS"); app.target_col_var.set("Close")
app.context_len_var.set(1056)          # the default, far longer than the data
app.horizon_var.set(7)
app.validation_ratio_var.set(0.2)

# Stub the model so this runs without weights: echo the last context value.
calls = []
def fake_predict(context, covariate_array, horizon, want_quantiles=False):
    context = np.asarray(context, dtype=float).reshape(-1)
    calls.append({"context_len": context.size,
                  "unique": int(np.unique(np.round(context, 6)).size),
                  "covariates": None if covariate_array is None else covariate_array.shape})
    point = np.full(horizon, float(context[-1]))
    return (point, None) if want_quantiles else point
app._predict_loaded_model = fake_predict
forecaster.TIMESFM_AVAILABLE = True
app._ensure_model_loaded = lambda *a, **k: False

captured = {}
def run():
    app._run_forecast_job()
    # log_message is posted via root.after(); let those callbacks run first.
    _root.after(150, lambda: (captured.__setitem__('log', app.log_text.get("1.0", "end")),
                              _root.quit()))
_root.after(100, run); _root.mainloop()

assert "IndexError" not in captured['log'], captured['log'][-800:]
assert "out of bounds" not in captured['log'], captured['log'][-800:]
print("no IndexError with 254 bars against a 1056 context")

assert app.forecast_data is not None, "forecast did not complete:\n" + captured['log'][-800:]
assert len(app.forecast_data) == 7
print(f"forecast produced: {len(app.forecast_data)} steps")

# Context clamped to a whole number of patches, not padded out to 1056.
assert 1056 % 32 == 0
expected = (n // 32) * 32
for call in calls:
    assert call["context_len"] == expected, (call["context_len"], expected)
print(f"context clamped to {expected} (= {n}//32*32), not padded to 1056")
assert "context reduced from 1056" in captured['log']

# The context must be real data, not a repeated edge value.
assert calls[0]["unique"] > expected * 0.9, calls[0]
print(f"context is real data: {calls[0]['unique']}/{expected} distinct values")

# Validation anchored inside the data, and scored.
origin = pd.Timestamp(app.validation_origin)
assert idx[0] <= origin <= idx[-1], (origin, idx[0], idx[-1])
print(f"validation origin {origin.date()} lies inside {idx[0].date()}..{idx[-1].date()}")
assert app.validation_metrics and dict(app.validation_metrics).get("MAE") is not None
print("validation metrics computed:", ", ".join(k for k, _ in app.validation_metrics))

assert pd.Timestamp(app.forecast_anchor) == idx[-1]
print(f"forecast anchor is the last real bar: {pd.Timestamp(app.forecast_anchor).date()}")

# Same again in return space: differencing drops a bar, so the date map shifts.
app.target_space_var.set(tf.SPACE_LABELS[tf.LOG_RETURN])
app.forecast_data = None
calls.clear()
def run2():
    app._run_forecast_job()
    _root.after(150, lambda: (captured.__setitem__('log2', app.log_text.get("1.0", "end")),
                              _root.quit()))
_root.after(100, run2); _root.mainloop()
assert app.forecast_data is not None, captured['log2'][-800:]
assert "out of bounds" not in captured['log2']
assert pd.Timestamp(app.forecast_anchor) == idx[-1], app.forecast_anchor
print("return space (253 values, 254 rows) stays aligned; anchor still the last bar")

_root.destroy()
print("\nALL SHORT-HISTORY TESTS PASSED")
