"""Survivorship: bound the bias, and start recording point-in-time membership."""

import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import universe
import cross_sectional as cs
import db_manager as db

db.DB_DIR = tempfile.mkdtemp(); db.DB_PATH = os.path.join(db.DB_DIR, 'surv.db')
db.init_db()

rng = np.random.default_rng(5)
n_days, n_names = 1600, 40
dates = pd.bdate_range("2019-01-02", periods=n_days)
prices = pd.DataFrame(
    100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, (n_days, n_names)), axis=0)),
    index=dates, columns=[f"N{i:02d}.IS" for i in range(n_names)])

# ---- late joiners: names that only appear part-way through ---------------
late = prices.copy()
for column in late.columns[:12]:
    late.loc[late.index[:700], column] = np.nan     # listed later
subset = universe.full_history_subset(late)
assert subset.shape[1] == n_names - 12, subset.shape
assert all(c not in subset.columns for c in late.columns[:12])
print(f"full-history subset: {late.shape[1]} names -> {subset.shape[1]} present from the start")

# a panel with no late joiners must be returned unchanged
assert universe.full_history_subset(prices).shape[1] == n_names
print("  a panel with no late joiners is untouched")

# ---- injected delistings behave like real ones --------------------------
stressed, n_ghosts = universe.inject_delistings(prices, annual_rate=0.03, seed=1)
assert n_ghosts >= 1
ghosts = [c for c in stressed.columns if c.startswith("GHOST")]
assert len(ghosts) == n_ghosts
assert stressed.shape[1] == prices.shape[1] + n_ghosts
print(f"injected {n_ghosts} synthetic delistings over {n_days/252:.1f} years "
      f"at a 3%/yr rate")

for ghost in ghosts:
    series = stressed[ghost]
    valid = series.dropna()
    # must actually stop trading before the panel ends
    assert valid.index[-1] < prices.index[-1], ghost
    # must decline into its delisting rather than vanish at full price
    assert valid.iloc[-1] < valid.iloc[len(valid) // 2], ghost
    # and must not contaminate the real names
    assert stressed[prices.columns].equals(prices)
print("  each ghost declines, then stops quoting; real names are untouched")

# the delisting rate must control how many appear
few, n_few = universe.inject_delistings(prices, annual_rate=0.01, seed=1)
many, n_many = universe.inject_delistings(prices, annual_rate=0.10, seed=1)
assert n_few < n_ghosts < n_many, (n_few, n_ghosts, n_many)
print(f"  rate controls the count: 1%->{n_few}, 3%->{n_ghosts}, 10%->{n_many}")

# ---- the stress test must actually move the answer ----------------------
def sharpe_of(panel):
    factor = cs.momentum_12_1(panel)
    return cs.run(panel, factor=factor, frequency="M", cost_bps=0.0)["summary"]["sharpe_net"]

base = sharpe_of(prices)
with_ghosts = sharpe_of(many)
print(f"Sharpe: {base:+.3f} as tested -> {with_ghosts:+.3f} with 10%/yr delistings")
assert base != with_ghosts, "injecting casualties changed nothing, which cannot be right"
print("  including the casualties changes the result, which is the whole point")

# ---- point-in-time snapshots --------------------------------------------
db.record_universe(["A.IS", "B.IS", "C.IS"], label="bist", snapshot_date="2024-01-31")
db.record_universe(["A.IS", "B.IS", "D.IS"], label="bist", snapshot_date="2025-01-31")
db.record_universe(["A.IS", "D.IS", "E.IS"], label="bist", snapshot_date="2026-01-31")
snapshots = db.get_universe_snapshots()
assert len(snapshots) == 3, len(snapshots)
print(f"recorded {len(snapshots)} dated snapshots")

# the query a point-in-time backtest needs: membership AS OF a past date
assert db.universe_as_of("2024-06-01", "bist") == ["A.IS", "B.IS", "C.IS"]
assert db.universe_as_of("2025-06-01", "bist") == ["A.IS", "B.IS", "D.IS"]
assert db.universe_as_of("2023-01-01", "bist") == []
print("  universe_as_of returns THEN's membership, not today's")
print("  C.IS was in the 2024 universe and is gone by 2026 - exactly the name a")
print("  current-membership list would silently drop")

# same day twice must update, not duplicate
db.record_universe(["A.IS", "Z.IS"], label="bist", snapshot_date="2026-01-31")
assert len(db.get_universe_snapshots()) == 3
assert db.universe_as_of("2026-06-01", "bist") == ["A.IS", "Z.IS"]
print("  re-fetching on the same day updates rather than duplicating")

print("\nALL SURVIVORSHIP TESTS PASSED")
