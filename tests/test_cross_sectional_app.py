"""The Cross-Sectional tab, driven end to end on a synthetic universe."""

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
db.DB_DIR = tempfile.mkdtemp(); db.DB_PATH = os.path.join(db.DB_DIR, 'xs.db')
import timesfm_forecaster as forecaster
import universe, cross_sectional as cs

app = forecaster.TimesFMApp(_root)
_root.update_idletasks(); _root.update()

assert hasattr(app, "xs_tree") and hasattr(app, "xs_factor_var")
print("Cross-Sectional tab built")
assert len(app.xs_tickers_var.get().split(",")) >= 30
print(f"  default universe: {len(app.xs_tickers_var.get().split(','))} tickers")

# Inject a panel rather than hitting the network.
rng = np.random.default_rng(91)
n_days, n_names, month = 1500, 40, 21
dates = pd.bdate_range("2019-01-02", periods=n_days)
n_months = n_days // month + 2
monthly = rng.normal(0, 1, (n_months, n_names))
month_of_day = np.arange(n_days) // month
driving = monthly[np.maximum(month_of_day - 1, 0)]
paths = 100 * np.exp(np.cumsum(0.0006 * driving + rng.normal(0, 0.012, (n_days, n_names)), axis=0))
names = [f"X{i:02d}.IS" for i in range(n_names)]
app.universe_prices = pd.DataFrame(paths, index=dates, columns=names)

# momentum on a panel with no momentum: the honest null case
app.xs_factor_var.set("momentum_12_1")
app.xs_freq_var.set("M")
app.xs_cost_var.set(20.0)
app.xs_vol_target_on.set(True)
app.xs_vol_target_var.set(0.15)

captured = {}
def run():
    app.thread_run_cross_sectional()
    _root.after(6000, collect)
def collect():
    captured['rows'] = [app.xs_tree.item(r, "values") for r in app.xs_tree.get_children()]
    captured['log'] = app.log_text.get("1.0", "end")
    _root.quit()
_root.after(300, run); _root.mainloop()

rows = captured['rows']
assert rows, captured['log'][-900:]
row = rows[0]
print(f"  result: factor={row[0]} years={row[2]} CAGR={row[3]} "
      f"SharpeNet={row[4]} SharpeGross={row[5]} DSR={row[6]} trials={row[7]}")
print(f"          maxDD={row[8]} turnover={row[10]} costdrag={row[11]} "
      f"lev={row[12]} IC={row[13]} ICt={row[14]}")
assert row[0].startswith("momentum_12_1")   # label now carries the currency
assert "[TRY]" in row[0], row[0]
assert row[4] not in ("-", ""), "no net Sharpe produced"
assert row[13] not in ("-", ""), "no IC produced"
assert row[6] not in ("-", ""), "no deflated Sharpe produced"
assert int(row[7]) >= 1, "trial count not recorded"

# Costs must be reported, and the log must state how much they took.
assert "Costs remove" in captured['log'], captured['log'][-600:]
print("  log states the share of gross Sharpe lost to costs")

# The IC guard must fire on a factor with no edge.
assert "IC t-stat" in captured['log'] or "has not shown it" in captured['log'] \
    or abs(float(row[14])) >= 2
print("  weak-IC warning present (or IC genuinely significant)")

# Equity curve drawn, net and gross.
labels = [l.get_label() for l in app.ax.get_lines()]
assert "Net of costs" in labels and "Gross" in labels, labels
print(f"  equity curve plotted: {labels}")

# Vol targeting actually levered.
assert float(row[12]) > 0
print(f"  volatility targeting applied, average leverage {row[12]}x")

# The deflated Sharpe must be reported, and the trial count must persist in the
# database so a search cannot be laundered by restarting the app.
assert "Deflated Sharpe" in captured['log'], captured['log'][-700:]
stored = db.count_backtest_runs("xs:")
assert stored >= 1, stored
print(f"  deflated Sharpe reported; {stored} cross-sectional trial(s) recorded in the DB")

# Survivorship reporting on a panel with a delisting.
panel = app.universe_prices.copy()
panel.iloc[800:, 3] = np.nan
report = universe.coverage_report(panel)
assert panel.columns[3] in report["stale_names"]
assert "optimistic" in universe.survivorship_note(report)
print(f"  survivorship: {len(report['stale_names'])} stale name(s) flagged")

_root.destroy()
print("\nALL CROSS-SECTIONAL APP TESTS PASSED")
