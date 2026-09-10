"""The app can forecast with SVR instead of TimesFM, with no weights loaded."""

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
db.DB_DIR = tempfile.mkdtemp(); db.DB_PATH = os.path.join(db.DB_DIR, 'svr.db')
import timesfm_forecaster as forecaster
import sk_models

if not sk_models.SKLEARN_AVAILABLE:
    print("SKIPPED: scikit-learn not installed"); raise SystemExit(0)

app = forecaster.TimesFMApp(_root)

rng = np.random.default_rng(23)
n = 900
idx = pd.bdate_range("2022-01-03", periods=n)
noise = rng.normal(0, 0.012, n)
ar = np.zeros(n)
for i in range(1, n):
    ar[i] = 0.6 * ar[i - 1] + noise[i]
close = 380 * np.exp(np.cumsum(ar))
app.historical_data = forecaster.features.build_features(pd.DataFrame({
    "Open": close, "High": close * 1.01, "Low": close * 0.99,
    "Close": close, "Adj Close": close, "Volume": rng.integers(1e5, 5e6, n).astype(float),
}, index=idx))
app.historical_data["USD/TRY"] = np.linspace(35, 42, n)
app._refresh_feature_controls()

app.tkr_var.set("ASELS.IS"); app.target_col_var.set("Close"); app.adjusted_var.set(False)
app.context_len_var.set(512); app.horizon_var.set(5)
app.validation_ratio_var.set(0.2); app.validation_origins_var.set(12)
app.model_family_var.set("SVR (RBF)")
assert not app._uses_timesfm()

# The whole point: no TimesFM weights anywhere.
app.loaded_model = None
forecaster.TIMESFM_AVAILABLE = False
app._ensure_model_loaded = lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("SVR path must not load TimesFM"))

app.feature_selected = {"USD/TRY"}
captured = {}
def run():
    app._run_forecast_job()
    _root.after(400, lambda: (captured.__setitem__('log', app.log_text.get("1.0", "end")),
                              _root.quit()))
_root.after(100, run); _root.mainloop()

log = captured['log']
assert "Traceback" not in log and "ERROR" not in log, log[-900:]
assert app.forecast_data is not None, log[-900:]
assert len(app.forecast_data) == 5
print(f"SVR forecast produced {len(app.forecast_data)} steps with no TimesFM loaded")

# Prices, not returns: the reconstruction must land near the last close.
last = float(close[-1])
assert all(last * 0.5 < v < last * 2.0 for v in app.forecast_data), app.forecast_data
print(f"  reconstructed to price space: last close {last:.1f}, forecast "
      f"{np.round(app.forecast_data, 1).tolist()}")

# It must NOT be the naive forecast wearing a kernel.
assert not np.allclose(app.forecast_data, last), "SVR just repeated the last price"
print("  and it is not simply the last price repeated")

assert "fits log returns internally" in log or "fitting on the context window" in log
print("  log records that SVR fits returns internally")

# Walk-forward validation ran on the SVR, and reports a verdict.
metrics_by_name = dict(app.validation_metrics)
print(f"  validation: {int(metrics_by_name['Origins'])} origins, "
      f"{int(metrics_by_name['Points'])} points, "
      f"skill {metrics_by_name['SkillVsNaive']*100:+.1f}%, "
      f"DM p(h1) {metrics_by_name['DM_p_h1']:.3f}")
assert metrics_by_name["Points"] > 7
assert "Verdict:" in log

# The saved run records which family produced it.
runs = [r for r in db.get_backtest_runs() if "validation" in r["model_name"]]
assert runs, "validation run not stored"
assert "SVR" in runs[0]["model_name"], runs[0]["model_name"]
print(f"  stored as '{runs[0]['model_name']}' with {runs[0]['n_points']} points")

# Switching back to TimesFM flips the routing.
app.model_family_var.set("TimesFM 3.0")
assert app._uses_timesfm()
print("switching family back to TimesFM restores the TimesFM path")

_root.destroy()
print("\nALL SVR-APP TESTS PASSED")
