"""The covariate table: scrolling, filtering, and selection that survives both.

After a fetch this list holds ~49 rows (OHLCV + 40 engineered columns + macro
series). It was a fixed 8-row Listbox with no scrollbar, so five sixths of the
table was unreachable. These tests drive real wheel events rather than calling
the handler directly, which is what the earlier scroll fix failed to check.
"""

import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    _root = tk.Tk(); _root.geometry("1400x850")
except Exception as exc:
    print(f"SKIPPED: Tk unavailable ({exc})"); raise SystemExit(0)

import db_manager as db
db.DB_DIR = tempfile.mkdtemp(); db.DB_PATH = os.path.join(db.DB_DIR, 'picker.db')
import timesfm_forecaster as forecaster

app = forecaster.TimesFMApp(_root)
_root.update_idletasks(); _root.update()

rng = np.random.default_rng(5)
n = 400
idx = pd.bdate_range("2024-01-02", periods=n)
close = 380 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
app.historical_data = forecaster.features.build_features(pd.DataFrame({
    "Open": close, "High": close * 1.01, "Low": close * 0.99,
    "Close": close, "Adj Close": close, "Volume": rng.integers(1e5, 5e6, n).astype(float),
}, index=idx))
app.historical_data["USD/TRY"] = np.linspace(35, 42, n)
app.historical_data["Brent crude"] = 80 + rng.normal(0, 5, n)

app._refresh_feature_controls()
_root.update_idletasks(); _root.update()

total = app.feature_listbox.size()
print(f"covariate table holds {total} rows, listbox shows {app.feature_listbox.cget('height')}")
assert total >= 40, total
assert total > app.feature_listbox.cget("height"), "nothing to scroll?"

# ---- a REAL wheel event must scroll the listbox, not the panel behind it --
panel = app.settings_frame.nametowidget(app.settings_frame.winfo_parent())
app.feature_listbox.yview_moveto(0.0); panel.yview_moveto(0.0)
_root.update_idletasks()
before_list = app.feature_listbox.yview()[0]
before_panel = panel.yview()[0]
app.feature_listbox.event_generate("<MouseWheel>", delta=-120, x=5, y=5)
_root.update_idletasks(); _root.update()
after_list = app.feature_listbox.yview()[0]
after_panel = panel.yview()[0]
print(f"  listbox yview {before_list:.3f} -> {after_list:.3f} | panel {before_panel:.3f} -> {after_panel:.3f}")
assert after_list > before_list, "wheel over the covariate table did not scroll it"
assert after_panel == before_panel, "the wheel scrolled the panel instead of the table"
print("  wheel scrolls the table, and does NOT leak to the settings panel")

# every row must be reachable
app.feature_listbox.yview_moveto(1.0); _root.update_idletasks()
assert app.feature_listbox.yview()[1] >= 0.999
print(f"  last row reachable: yview {app.feature_listbox.yview()[0]:.3f}-1.000")

# ---- selection survives filtering ----------------------------------------
app.feature_listbox.selection_clear(0, tk.END)
app.feature_selected.clear()
names = list(app.feature_catalog)
ema = next(c for c in names if c.startswith("EMA_"))
usdtry = "USD/TRY"
for column in (ema, usdtry):
    app.feature_selected.add(column)
app._render_feature_list(); _root.update_idletasks()
assert set(app._selected_covariates()) == {ema, usdtry}
print(f"selected {ema} and {usdtry}")

app.feature_filter_var.set("RSI")            # hides both selections
_root.update_idletasks(); _root.update()
shown = [app.feature_listbox.get(i) for i in range(app.feature_listbox.size())]
assert all("rsi" in s.lower() for s in shown), shown
assert ema not in shown and usdtry not in shown
# THE POINT: a covariate hidden by the filter must still be used.
assert set(app._selected_covariates()) == {ema, usdtry}, app._selected_covariates()
print(f"  filter 'RSI' shows {len(shown)} row(s); hidden selections still returned")

app.feature_filter_var.set("")
_root.update_idletasks(); _root.update()
restored = {app.feature_listbox.get(i) for i in app.feature_listbox.curselection()}
assert restored == {ema, usdtry}, restored
print("  clearing the filter restores both as visibly selected")

# ---- All/None act on the filtered view -----------------------------------
app.feature_filter_var.set("EMA"); _root.update_idletasks()
app._select_features("all"); _root.update_idletasks()
ema_columns = {c for c in names if "ema" in c.lower()}
assert ema_columns <= set(app._selected_covariates())
assert usdtry in app._selected_covariates(), "All wiped a selection outside the filter"
print(f"  'All' under filter 'EMA' added {len(ema_columns)} column(s) and kept USD/TRY")

app._select_features("none"); _root.update_idletasks()
assert not (ema_columns & set(app._selected_covariates()))
assert usdtry in app._selected_covariates()
print("  'None' removed only the filtered rows")

app.feature_filter_var.set("")
assert "selected" in app.feature_count_label.cget("text")
print("  count label:", app.feature_count_label.cget("text"))

_root.destroy()
print("\nALL FEATURE-PICKER TESTS PASSED")
