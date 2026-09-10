"""Cross-sectional ranking: no look-ahead, costs bite, vol targeting works."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import cross_sectional as cs
import universe

rng = np.random.default_rng(77)


def make_panel(n_days=1600, n_names=40, seed_drift=None, start="2019-01-02"):
    dates = pd.bdate_range(start, periods=n_days)
    drift = seed_drift if seed_drift is not None else rng.normal(0, 0.0004, n_names)
    frame = {}
    for i in range(n_names):
        returns = rng.normal(drift[i], 0.018, n_days)
        frame[f"T{i:02d}.IS"] = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(frame, index=dates)


prices = make_panel()
print(f"panel: {prices.shape[0]} days x {prices.shape[1]} names")

# ---- FACTORS ARE CAUSAL --------------------------------------------------
for name, fn in cs.FACTORS.items():
    full = fn(prices)
    truncated = fn(prices.iloc[:900])
    a = full.iloc[:900].to_numpy(float)
    b = truncated.to_numpy(float)
    assert np.allclose(a, b, equal_nan=True), f"NON-CAUSAL factor: {name}"
print(f"CAUSAL: all {len(cs.FACTORS)} factors unchanged when the future is removed")

# ---- NO LOOK-AHEAD: poisoning the future must not change the past --------
clean = cs.run(prices, factor_name="momentum_12_1", frequency="M", cost_bps=0.0)
poisoned_prices = prices.copy()
cut = prices.index[1200]
poisoned_prices.loc[poisoned_prices.index > cut] *= 5.0     # violent future move
poisoned = cs.run(poisoned_prices, factor_name="momentum_12_1", frequency="M", cost_bps=0.0)

overlap = clean["net_returns"].index[clean["net_returns"].index <= cut]
a = clean["net_returns"].reindex(overlap).to_numpy(float)
b = poisoned["net_returns"].reindex(overlap).to_numpy(float)
assert np.allclose(a, b, equal_nan=True, atol=1e-12), "returns before the cut changed"
print(f"NO LOOK-AHEAD: {len(overlap)} days before the cut are bit-identical "
      "after multiplying every later price by 5")

# ---- weights are formed BEFORE the returns they earn ---------------------
weights = clean["weights"]
first_weight_date = weights.index[0]
assert clean["net_returns"].index[0] > first_weight_date
print("weights dated t earn returns strictly after t")

# ---- gross exposure and long/short balance -------------------------------
row = weights.iloc[0]
assert abs(row.abs().sum() - 1.0) < 1e-9, row.abs().sum()
assert abs(row[row > 0].sum() - 0.5) < 1e-9
assert abs(row[row < 0].sum() + 0.5) < 1e-9
print(f"gross exposure 1.0, balanced {int((row>0).sum())} long / {int((row<0).sum())} short")

long_only = cs.run(prices, frequency="M", long_only=True, cost_bps=0.0)
lo_row = long_only["weights"].iloc[0]
assert (lo_row >= 0).all() and abs(lo_row.sum() - 1.0) < 1e-9
print("long-only mode holds no shorts and is fully invested")

# ---- COSTS BITE ----------------------------------------------------------
free = cs.run(prices, frequency="M", cost_bps=0.0)["summary"]
cheap = cs.run(prices, frequency="M", cost_bps=20.0)["summary"]
dear = cs.run(prices, frequency="M", cost_bps=200.0)["summary"]
print(f"costs: Sharpe {free['sharpe_net']:+.2f} (0bps) -> {cheap['sharpe_net']:+.2f} "
      f"(20bps) -> {dear['sharpe_net']:+.2f} (200bps)")
assert free["sharpe_net"] > cheap["sharpe_net"] > dear["sharpe_net"]
assert dear["cost_drag_annual"] > cheap["cost_drag_annual"] > 0
print(f"  annual cost drag at 20bps: {cheap['cost_drag_annual']*100:.2f}% "
      f"on {cheap['turnover_annual']:.1f}x turnover")

# more frequent rebalancing must cost more
monthly = cs.run(prices, frequency="M", cost_bps=20.0)["summary"]
weekly = cs.run(prices, frequency="W", cost_bps=20.0)["summary"]
assert weekly["turnover_annual"] > monthly["turnover_annual"]
print(f"  weekly turnover {weekly['turnover_annual']:.1f}x vs monthly "
      f"{monthly['turnover_annual']:.1f}x")

# ---- POSITIVE CONTROL: a factor that genuinely predicts ------------------
# A panel where THIS month's signal drives NEXT month's returns. The signal
# has to persist across the holding period, otherwise the monthly forward
# return depends on ~21 different signal values and its correlation with the
# one at the rebalance date is diluted by sqrt(21).
n_days, n_names, month = 1600, 40, 21
dates = pd.bdate_range("2019-01-02", periods=n_days)
n_months = n_days // month + 2
monthly_signal = rng.normal(0, 1, (n_months, n_names))
month_of_day = np.arange(n_days) // month

daily_signal = monthly_signal[month_of_day]              # what we know at t
driving = monthly_signal[np.maximum(month_of_day - 1, 0)]  # what moved prices
alpha = 0.0006
noise = rng.normal(0, 0.012, (n_days, n_names))
paths = 100 * np.exp(np.cumsum(alpha * driving + noise, axis=0))
predictive_prices = pd.DataFrame(paths, index=dates,
                                 columns=[f"S{i:02d}.IS" for i in range(n_names)])
known_factor = pd.DataFrame(daily_signal, index=dates, columns=predictive_prices.columns)

controlled = cs.run(predictive_prices, factor=known_factor, frequency="M", cost_bps=0.0)
ic = cs.information_coefficient(predictive_prices, known_factor, "M")
print(f"positive control: Sharpe {controlled['summary']['sharpe_net']:+.2f}, "
      f"IC {ic['ic_mean']:+.3f} (t={ic['ic_t_stat']:.1f}, {ic['n_periods']} periods)")
assert controlled["summary"]["sharpe_net"] > 0.8, controlled["summary"]["sharpe_net"]
assert ic["ic_t_stat"] > 3, ic
print("  a real cross-sectional signal is detected by both Sharpe and IC")

# and a random factor is not
random_factor = pd.DataFrame(rng.normal(0, 1, (n_days, n_names)),
                             index=dates, columns=predictive_prices.columns)
noise_ic = cs.information_coefficient(predictive_prices, random_factor, "M")
assert abs(noise_ic["ic_t_stat"]) < 3, noise_ic
print(f"  a random factor is not: IC {noise_ic['ic_mean']:+.3f} "
      f"(t={noise_ic['ic_t_stat']:.1f})")

# ---- VOL TARGETING -------------------------------------------------------
plain = cs.run(prices, frequency="M", cost_bps=0.0)
targeted = cs.run(prices, frequency="M", cost_bps=0.0, vol_target=0.10)
plain_vol = float(np.std(plain["net_returns"], ddof=1) * np.sqrt(252))
target_vol = float(np.std(targeted["net_returns"], ddof=1) * np.sqrt(252))
print(f"vol targeting: realised {plain_vol*100:.1f}% -> {target_vol*100:.1f}% "
      f"(target 10%), avg leverage {targeted['summary']['avg_leverage']:.2f}x")
assert abs(target_vol - 0.10) < abs(plain_vol - 0.10), (plain_vol, target_vol)
print("  moves realised volatility toward the target")

# ---- survivorship reporting ----------------------------------------------
delisted = prices.copy()
delisted.iloc[900:, 0] = np.nan          # a name stops trading
report = universe.coverage_report(delisted)
assert delisted.columns[0] in report["stale_names"], report["stale_names"]
print(f"survivorship: flags {len(report['stale_names'])} stale name(s); "
      f"{report['names_at_start']} at start vs {report['names_at_end']} at end")
assert "optimistic" in universe.survivorship_note(report)
print("  and states the warning in plain words")

print("\nALL CROSS-SECTIONAL TESTS PASSED")
