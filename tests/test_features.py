import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import features as ft

rng = np.random.default_rng(4)
n = 600
idx = pd.bdate_range("2023-01-02", periods=n)
close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
frame = pd.DataFrame({
    "Open": close * (1 + rng.normal(0, 0.002, n)),
    "High": close * (1 + abs(rng.normal(0, 0.006, n))),
    "Low":  close * (1 - abs(rng.normal(0, 0.006, n))),
    "Close": close,
    "Adj Close": close,
    "Volume": rng.integers(1e5, 5e6, n).astype(float),
}, index=idx)

out = ft.build_features(frame)
added = [c for c in out.columns if c not in frame.columns]
print(f"{len(added)} engineered columns")
assert len(added) >= 30, len(added)

# ---- CAUSALITY: truncating the future must not change past values --------
truncated = ft.build_features(frame.iloc[:400])
mismatched = []
for column in added:
    a = out[column].iloc[:400]
    b = truncated[column]
    if not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=1e-9):
        mismatched.append(column)
assert not mismatched, f"NON-CAUSAL (look-ahead) columns: {mismatched}"
print("CAUSALITY: every engineered column is unchanged when the future is removed")

# ---- external alignment must never back-fill -----------------------------
market = idx
# a macro series that starts late and is published weekly
macro_idx = pd.bdate_range("2023-06-01", periods=100, freq="7D")
macro = pd.Series(np.arange(100.0), index=macro_idx)
aligned = ft.align_external(macro, market, "macro")
assert aligned.loc[:"2023-05-31"].isna().all(), "leading gap was back-filled with a future value"
print("NO BACKFILL: dates before the series starts stay NaN")

# a value must hold until the next release, never anticipate it
first = macro_idx[0]
nxt = macro_idx[1]
between = aligned.loc[first:nxt].iloc[:-1]
assert (between == macro.iloc[0]).all(), "value changed before the next release"
print("FORWARD FILL: each value holds until the next actual release")

# ---- standardisation ------------------------------------------------------
raw = np.array([[50.0]*100, [400.0 + i for i in range(100)]], dtype=float)
scaled = ft.standardize_context(raw)
assert scaled.shape == raw.shape
assert abs(float(np.mean(scaled[1]))) < 1e-5, np.mean(scaled[1])
assert abs(float(np.std(scaled[1])) - 1.0) < 1e-4, np.std(scaled[1])
# a constant row must not explode
assert np.all(np.isfinite(scaled[0])) and np.allclose(scaled[0], 0.0)
print("STANDARDISE: zero mean / unit variance, constant rows collapse to 0 not inf")

# NaNs must not propagate
noisy = raw.copy(); noisy[0, :10] = np.nan
assert np.all(np.isfinite(ft.standardize_context(noisy)))
print("STANDARDISE: NaNs do not propagate")

# ---- catalogue sanity -----------------------------------------------------
assert len(ft.MACRO_CATALOG) >= 20, len(ft.MACRO_CATALOG)
for label in ft.DEFAULT_MACRO_SELECTION:
    assert label in ft.MACRO_CATALOG, label
sources = {v[0] for v in ft.MACRO_CATALOG.values()}
assert sources == {"fred", "yahoo"}, sources
print(f"CATALOGUE: {len(ft.MACRO_CATALOG)} series, {len(ft.DEFAULT_MACRO_SELECTION)} default, sources {sources}")
print("\nALL FEATURE TESTS PASSED")
