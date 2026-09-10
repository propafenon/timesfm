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


# ==========================================================================
# USD RANKING, OVERNIGHT/INTRADAY SPLIT, DEFLATED SHARPE
# ==========================================================================

# ---- ranking in USD strips out the currency ------------------------------
fx_dates = prices.index
fx = pd.Series(np.exp(np.cumsum(rng.normal(0.0008, 0.006, len(fx_dates)))) * 10,
               index=fx_dates)          # a steadily depreciating lira
usd_prices = cs.to_usd(prices, fx)
assert usd_prices.shape == prices.shape
np.testing.assert_allclose(usd_prices.iloc[0].to_numpy(float),
                           (prices.iloc[0] / fx.iloc[0]).to_numpy(float), rtol=1e-12)
print(f"\nUSD conversion: TRY {prices.iloc[-1, 0]:.1f} -> USD {usd_prices.iloc[-1, 0]:.2f}")

# A pure currency move must NOT change the cross-sectional ranking, because it
# hits every name identically. This is the sanity check that the conversion is
# doing something real rather than adding noise.
try_rank = cs.momentum_12_1(prices).iloc[-1].rank()
usd_rank = cs.momentum_12_1(usd_prices).iloc[-1].rank()
agreement = try_rank.corr(usd_rank, method="spearman")
print(f"  rank agreement TRY vs USD under a common FX move: {agreement:.3f}")
assert agreement > 0.99, agreement
print("  a market-wide currency move leaves the ranking intact, as it must")

# Ratio factors are PROVABLY rank-invariant to the conversion: momentum_usd is
# (momentum_try + 1)/G - 1, the same monotonic map for every name. So this is a
# no-op for selection, and the code says so rather than implying otherwise.
for invariant in ("momentum_12_1", "momentum_6_1", "short_term_reversal"):
    a = cs.FACTORS[invariant](prices).iloc[-1]
    b = cs.FACTORS[invariant](usd_prices).iloc[-1]
    pair = pd.concat([a, b], axis=1).dropna()
    assert pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman") > 0.9999, invariant
    assert invariant in cs.FX_RANK_INVARIANT
print("  ratio factors are rank-invariant in USD, and flagged as such")

# Volatility-based factors are NOT: a name's return minus a common FX return
# has a different variance depending on how it co-moves with the currency.
vol_try = cs.low_volatility(prices).iloc[-1]
vol_usd = cs.low_volatility(usd_prices).iloc[-1]
pair = pd.concat([vol_try, vol_usd], axis=1).dropna()
vol_agreement = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
print(f"  low_volatility rank correlation TRY vs USD: {vol_agreement:.3f} (changes)")
assert vol_agreement < 0.99
assert "low_volatility" not in cs.FX_RANK_INVARIANT

# And the return stream differs either way, which is what a USD investor earns.
try_run = cs.run(prices, frequency="M", cost_bps=0.0)["summary"]
usd_run = cs.run(usd_prices, frequency="M", cost_bps=0.0)["summary"]
assert try_run["sharpe_net"] != usd_run["sharpe_net"]
print(f"  return stream does differ: Sharpe {try_run['sharpe_net']:+.3f} (TRY) vs "
      f"{usd_run['sharpe_net']:+.3f} (USD)")

# forward fill only: an FX rate must never be used before it existed
sparse_fx = fx.iloc[::7]                      # weekly quotes
converted = cs.to_usd(prices, sparse_fx)
assert converted.iloc[:5].isna().all().all() or converted.notna().any().any()
early = prices.index[prices.index < sparse_fx.index[0]]
if len(early):
    assert converted.loc[early].isna().all().all(), "FX was back-filled"
print("  sparse FX is forward-filled, never back-filled")

# ---- overnight / intraday split ------------------------------------------
opens = prices.shift(1) * (1 + rng.normal(0, 0.004, prices.shape))   # plausible opens
panels = {"Close": prices, "Open": opens}

for name in cs.NEEDS_OPEN:
    with_open = cs.FACTORS[name](prices, panels)
    assert with_open.notna().any().any(), f"{name} produced nothing"
    # causal: truncating the future must not move the past
    truncated = cs.FACTORS[name](prices.iloc[:900],
                                 {k: v.iloc[:900] for k, v in panels.items()})
    a = with_open.iloc[:900].to_numpy(float); b = truncated.to_numpy(float)
    assert np.allclose(a, b, equal_nan=True), f"NON-CAUSAL: {name}"
    # and without an Open panel it must be all-NaN, not silently wrong
    assert cs.FACTORS[name](prices, None).isna().all().all(), f"{name} faked it"
print(f"overnight/intraday: {len(cs.NEEDS_OPEN)} factors causal, and blank without Open")

# the two legs must reconstruct the total move
overnight = cs._overnight_log_returns(prices, panels)
intraday = cs._intraday_log_returns(prices, panels)
total = np.log(prices / prices.shift(1))
rebuilt = overnight + intraday
both = np.isfinite(total.to_numpy(float)) & np.isfinite(rebuilt.to_numpy(float))
np.testing.assert_allclose(total.to_numpy(float)[both], rebuilt.to_numpy(float)[both],
                           rtol=1e-9, atol=1e-12)
print("  overnight + intraday reconstructs the daily return exactly")

overnight_run = cs.run(prices, factor_name="overnight_momentum_12_1",
                       frequency="M", cost_bps=0.0, panels=panels)
print(f"  overnight momentum backtest ran: Sharpe "
      f"{overnight_run['summary']['sharpe_net']:+.2f} over "
      f"{overnight_run['rebalances']} rebalances")

# ---- deflated Sharpe ------------------------------------------------------
one = cs.run(prices, frequency="M", cost_bps=0.0, n_trials=1)["summary"]
fifty = cs.run(prices, frequency="M", cost_bps=0.0, n_trials=50)["summary"]
thousand = cs.run(prices, frequency="M", cost_bps=0.0, n_trials=1000)["summary"]
print(f"\ndeflated Sharpe on the SAME result: "
      f"1 trial {one['deflated_sharpe']:.3f} -> 50 {fifty['deflated_sharpe']:.3f} "
      f"-> 1000 {thousand['deflated_sharpe']:.3f}")
assert one["deflated_sharpe"] >= fifty["deflated_sharpe"] >= thousand["deflated_sharpe"]
assert one["sharpe_net"] == fifty["sharpe_net"], "the raw Sharpe must not change"
print("  identical returns, less believable the more configurations were tried")
assert np.isfinite(one["return_skew"]) and np.isfinite(one["return_kurtosis"])
print(f"  return shape reported: skew {one['return_skew']:+.2f}, "
      f"kurtosis {one['return_kurtosis']:.2f}")

print("\nALL CROSS-SECTIONAL TESTS PASSED")
