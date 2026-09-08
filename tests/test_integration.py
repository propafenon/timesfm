"""End-to-end: build the UI, save a forecast, score it, run a backtest.

Needs a display for Tk and exits cleanly (skipped) without one. This suite is
worth its runtime: it is what caught show_step_detail crashing on a null
directional accuracy, which no unit test would have reached.

Note the harness drives a real mainloop. root.update() does NOT dispatch
after() callbacks scheduled from worker threads, so a polling loop would show
an empty grid and prove nothing.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    _root = tk.Tk()
    _root.withdraw()
except Exception as exc:                                  # no display available
    print(f"SKIPPED: Tk unavailable ({exc})")
    raise SystemExit(0)

import db_manager as db
db.DB_DIR = tempfile.mkdtemp()
db.DB_PATH = os.path.join(db.DB_DIR, 'integration.db')

import timesfm_forecaster_old as forecaster
from scipy import stats as st

app = forecaster.TimesFMApp(_root)
print("app constructed, all tabs built")

rng = np.random.default_rng(3)
index = pd.bdate_range("2022-01-03", periods=700)
prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 700)))
app.historical_data = pd.DataFrame(
    {"Close": prices, "Adj Close": prices, "Open": prices,
     "High": prices, "Low": prices, "Volume": 1e6},
    index=index,
)
app.tkr_var.set("TEST.IS")
app.interval_var.set("1d")
app.target_col_var.set("Close")
app.adjusted_var.set(False)          # avoids the look-ahead confirmation dialog

horizon = 5
anchor_price = float(prices[-horizon - 1])
levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
app.forecast_data = np.full(horizon, anchor_price)
app.forecast_anchor = index[-horizon - 1]
app.forecast_target_col = "Close"
app.forecast_quantiles = {
    "levels": levels,
    "values": [[anchor_price + st.norm.ppf(l) * anchor_price * 0.01 * np.sqrt(k + 1)
                for l in levels] for k in range(horizon)],
}

app.update_plot()
bands = [c.get_label() for c in app.ax.collections]
assert any("interval" in b for b in bands), f"fan chart missing: {bands}"
print(f"fan chart rendered: {bands}")

app.bt_context_var.set(128)
app.bt_horizon_var.set(5)
app.bt_step_var.set(20)
app.bt_baselines_var.set(True)
app.bt_timesfm_var.set(False)

captured = {}


def save_step():
    app.save_forecast_to_db()
    _root.after(500, score_step)


def score_step():
    rows = app.run_tree.get_children()
    assert rows, "forecast was not saved"
    app.run_tree.selection_set(rows[0])
    app.calculate_mae_for_selected()
    _root.after(500, backtest_step)


def backtest_step():
    captured['grid'] = [app.run_tree.item(r, "values") for r in app.run_tree.get_children()]
    app.thread_run_backtest()
    _root.after(9000, detail_step)


def detail_step():
    captured['backtest'] = [app.bt_tree.item(r, "values") for r in app.bt_tree.get_children()]
    children = app.bt_tree.get_children()
    if children:
        app.bt_tree.selection_set(children[0])
        app.show_step_detail()          # must not raise on null metrics
    captured['log'] = app.log_text.get("1.0", "end")
    _root.quit()


_root.after(200, save_step)
_root.mainloop()
_root.destroy()

row = captured['grid'][0]
assert row[11] != "-", "MASE was not written to the history grid"
print(f"forecast scored: MAE={row[10]} MASE={row[11]} Dir={row[12]}")

results = captured['backtest']
assert len(results) >= 5, f"expected every baseline to be saved, got {len(results)}"
by_model = {r[1].split(" ")[0]: r for r in results}
assert "naive" in by_model, by_model.keys()
# A flat forecast has no directional view; it must read as "-", not 0%.
assert by_model["naive"][6] == "-", by_model["naive"]
# Naive against itself has exactly zero skill.
assert by_model["naive"][5] == "0.0%", by_model["naive"]
print(f"{len(results)} backtest runs saved; naive reports no directional view")

points = db.get_backtest_points(int(results[0][0]))
assert points, "per-step points were not persisted"
assert {p['step'] for p in points} == {1, 2, 3, 4, 5}, "missing horizon steps"
print(f"backtest_points persisted: {len(points)} rows across all 5 horizon steps")

assert "by horizon step" in captured['log'], "per-step detail did not render"
print("per-step detail rendered without raising on null metrics")
print("\nALL INTEGRATION TESTS PASSED")
