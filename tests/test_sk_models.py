"""SVR forecaster: leak-free, and fitted on returns rather than price levels."""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import backtest as bt
import sk_models

if not sk_models.SKLEARN_AVAILABLE:
    print("SKIPPED: scikit-learn not installed"); raise SystemExit(0)

rng = np.random.default_rng(17)

# ---- POSITIVE CONTROL: returns with real AR(1) structure -----------------
n = 1200
dates = pd.bdate_range("2021-01-04", periods=n)
noise = rng.normal(0, 0.01, n)
ar = np.zeros(n)
for i in range(1, n):
    ar[i] = 0.7 * ar[i - 1] + noise[i]           # strongly predictable
predictable = 100 * np.exp(np.cumsum(ar))

svr = sk_models.make_svr_forecaster(kernel="rbf", C=2.0, lags=8, max_train=600)
start = time.time()
rows = bt.walk_forward(predictable, dates, svr, 400, 3, 10)
elapsed = time.time() - start
summary = bt.summarize(rows, interval="1d")
origins = summary["n_origins"]
print(f"AR(1) returns, {origins} origins in {elapsed:.1f}s ({elapsed/origins:.2f}s each)")
print(f"  h=1 MASE {summary['mase_step1']:.3f}  skill {summary['skill_step1']*100:+.1f}%  "
      f"DM p {summary['dm_pvalue_step1']:.4f}  dir {summary['directional_accuracy']:.3f}")
assert summary["skill_step1"] > 0.05, summary["skill_step1"]
assert summary["dm_pvalue_step1"] < 0.05, summary["dm_pvalue_step1"]
print("  SVR finds real structure and the DM test confirms it")

# ---- NEGATIVE CONTROL: a random walk has nothing to find -----------------
walk = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
rows_rw = bt.walk_forward(walk, dates, svr, 400, 3, 10)
summary_rw = bt.summarize(rows_rw, interval="1d")
print(f"random walk: h=1 MASE {summary_rw['mase_step1']:.3f}  "
      f"skill {summary_rw['skill_step1']*100:+.1f}%  DM p {summary_rw['dm_pvalue_step1']:.3f}")
assert summary_rw["dm_pvalue_step1"] > 0.05, summary_rw["dm_pvalue_step1"]
print("  and finds nothing on a random walk, as it must")

# ---- THE PRICE-LEVEL TRAP -------------------------------------------------
# An SVR fitted on lagged PRICE LEVELS looks superb and is just persistence.
from sklearn.svm import SVR as RawSVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def level_fitted(context, horizon, covariate_window=None):
    prices = np.asarray(context, dtype=float)
    lags = 8
    X = np.array([prices[t - lags:t] for t in range(lags, prices.size - 1)])
    y = prices[lags + 1:]
    model = make_pipeline(StandardScaler(), RawSVR(kernel="rbf", C=100.0))
    model.fit(X[-600:], y[-600:])
    step = float(model.predict(prices[-lags:].reshape(1, -1))[0])
    return np.full(horizon, step), None
level_fitted.wants_covariates = True

rows_level = bt.walk_forward(walk, dates, level_fitted, 400, 3, 40)
summary_level = bt.summarize(rows_level, interval="1d")
truth = np.array([r["y_true"] for r in rows_level])
pred = np.array([r["y_pred"] for r in rows_level])
r_squared = 1 - np.sum((truth - pred) ** 2) / np.sum((truth - truth.mean()) ** 2)
print(f"level-fitted SVR on the SAME random walk: R^2 {r_squared:.4f} (looks superb) "
      f"but skill vs naive {summary_level['skill_step1']*100:+.1f}%")
assert r_squared > 0.9, r_squared
assert summary_level["skill_step1"] < 0.05, summary_level["skill_step1"]
print("  high R^2, no skill: it learned persistence. This is why we fit returns.")

# ---- leak-free: the model never sees past its origin ----------------------
poisoned = walk.copy()
poisoned[700:] = 9e9
seen = []
def spy(context, horizon, covariate_window=None):
    seen.append(float(np.max(context)))
    return svr(context, horizon, covariate_window)
spy.wants_covariates = True
bt.walk_forward(poisoned[:760], dates[:760], spy, 400, 3, 40)
for origin, peak in zip(bt.plan_origins(760, 400, 3, 40), seen):
    assert peak <= poisoned[:origin + 1].max() + 1e-6, (origin, peak)
print(f"leak-free across {len(seen)} origins")

# ---- covariates are consumed, and the length mismatch is handled ---------
covariates = np.vstack([np.linspace(30, 45, n), np.arange(n) * 0.01])
rows_cov = bt.walk_forward(predictable, dates, svr, 400, 3, 10, covariates=covariates)
assert len(rows_cov) == len(rows)
print(f"accepts a ({covariates.shape[0]}, n) covariate matrix and aligns it to returns")

# ---- degenerate input falls back instead of raising ----------------------
flat = np.full(500, 100.0)
out, _ = svr(flat, 3)
assert np.allclose(out, 100.0), out
short, _ = svr(np.array([10.0, 11.0, 12.0]), 3)
assert np.allclose(short, 12.0), short
print("degenerate and too-short input fall back to no change")

print("\nALL SVR TESTS PASSED")
