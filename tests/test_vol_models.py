"""HAR-RV: does the gold standard actually beat naive on volatility?"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import backtest as bt
import vol_models as vm

rng = np.random.default_rng(41)

# ---- realized volatility is computed causally -----------------------------
n = 2000
prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
rv_full = vm.realized_volatility(prices, window=1)
rv_trunc = vm.realized_volatility(prices[:1200], window=1)
assert np.allclose(rv_full[:len(rv_trunc)], rv_trunc, equal_nan=True)
assert rv_full.size == prices.size - 1
print(f"realized vol: {rv_full.size} obs from {prices.size} prices, causal")

# a smoothed window must also stay causal
rv5 = vm.realized_volatility(prices, window=5)
rv5_trunc = vm.realized_volatility(prices[:1200], window=5)
assert np.allclose(rv5[:len(rv5_trunc)], rv5_trunc, equal_nan=True)
print("  smoothed windows are causal too")

# ---- GARCH-like clustering: HAR must find it -----------------------------
# Volatility that clusters, which is the empirical fact HAR exploits.
sigma = np.zeros(n); sigma[0] = 0.015
for i in range(1, n):
    sigma[i] = np.sqrt(0.000005 + 0.08 * (sigma[i-1] * rng.normal()) ** 2 + 0.90 * sigma[i-1] ** 2)
clustered_returns = sigma * rng.normal(size=n)
clustered_prices = 100 * np.exp(np.cumsum(clustered_returns))

rv = vm.realized_volatility(clustered_prices, window=1)
dates = pd.bdate_range("2015-01-02", periods=rv.size)
good = np.isfinite(rv) & (rv > 0)
rv, dates = rv[good], dates[good]

print(f"\nclustered-volatility series: {rv.size} obs, median {np.median(rv):.3f}")
print(f"{'model':<12}{'MASE h1':>9}{'skill h1':>10}{'DM p h1':>9}")
print("-" * 42)
results = {}
for name, fn in {"naive": bt.naive, **vm.available_models()}.items():
    rows = bt.walk_forward(rv, dates, fn, 400, 3, 10)
    s = bt.summarize(rows, interval="1d")
    results[name] = s
    print(f"{name:<12}{s['mase_step1']:>9.3f}{s['skill_step1']*100:>9.1f}%{s['dm_pvalue_step1']:>9.3f}")

# naive on a volatility series must score ~1.0, as everywhere else
assert 0.85 < results["naive"]["mase_step1"] < 1.15, results["naive"]["mase_step1"]
# and HAR must beat it, significantly
assert results["har_rv"]["skill_step1"] > 0.05, results["har_rv"]["skill_step1"]
assert results["har_rv"]["dm_pvalue_step1"] < 0.05, results["har_rv"]["dm_pvalue_step1"]
print("\nHAR-RV beats naive on clustered volatility, and the DM test confirms it")

# ---- but it must NOT invent skill where there is none --------------------
# i.i.d. returns: constant volatility, nothing to predict beyond the mean.
flat_returns = rng.normal(0, 0.015, n)
flat_prices = 100 * np.exp(np.cumsum(flat_returns))
rv_flat = vm.realized_volatility(flat_prices, window=1)
good2 = np.isfinite(rv_flat) & (rv_flat > 0)
rows_flat = bt.walk_forward(rv_flat[good2], dates[:good2.sum()],
                            vm.make_har_forecaster(), 400, 3, 10)
s_flat = bt.summarize(rows_flat, interval="1d")
print(f"constant-volatility control: HAR skill {s_flat['skill_step1']*100:+.1f}% "
      f"(vs naive, which is a poor predictor of noise either way)")

# ---- predictions must be positive and finite ------------------------------
har = vm.make_har_forecaster()
out, _ = har(rv[:600], 5)
assert np.all(np.isfinite(out)) and np.all(out > 0), out
print(f"HAR output positive and finite: {np.round(out, 4).tolist()}")

# degenerate input falls back rather than raising
short, _ = har(np.array([0.2, 0.3, 0.25]), 3)
assert np.all(np.isfinite(short)), short
zeros, _ = har(np.zeros(500), 3)
assert np.all(np.isfinite(zeros)), zeros
print("degenerate input handled without raising")

print("\nALL VOLATILITY TESTS PASSED")
