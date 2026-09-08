import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, backtest as bt, metrics as m

# ---- LEAKAGE: a model that peeks would see these sentinels -------------
n=200
values=np.arange(100.,100.+n)
values[150:]=9e9                      # anything at/after 150 is poison
dates=pd.bdate_range("2024-01-02",periods=n)

seen_max=[]
def spy(context,horizon):
    seen_max.append(float(np.max(context)))
    return np.full(horizon,float(context[-1])),None

rows=bt.walk_forward(values,dates,spy,context_len=64,horizon=5,step=7)
by_origin={r["origin_index"] for r in rows}
for r in rows:
    o=r["origin_index"]
    # the context handed to the model must never exceed values[:o+1]
    pass
for o,mx in zip(sorted(by_origin),seen_max):
    assert mx<=values[:o+1].max()+1e-9, f"origin {o} saw {mx}"
print(f"NO LEAKAGE: {len(seen_max)} origins, context never reached past the origin")

# ---- ORIGIN PLANNING: every origin must have a full horizon of truth ---
origins=bt.plan_origins(n,context_len=64,horizon=5,step=7)
assert max(origins)+5<=n-1, max(origins)
print(f"origins planned: {len(origins)}, last={max(origins)}, safe against n={n}")

# ---- CALENDAR: target dates come from the real index, holidays excluded
idx=pd.DatetimeIndex(list(pd.bdate_range("2024-01-02",periods=40)))
idx=idx.delete(20)                    # simulate an exchange holiday
vals=np.arange(100.,100.+len(idx))
rows2=bt.walk_forward(vals,idx,bt.naive,context_len=10,horizon=3,step=5)
targets={pd.Timestamp(r["target_date"]) for r in rows2}
assert targets.issubset(set(idx)), "invented a date not in the trading calendar"
assert idx[19]+pd.Timedelta(days=0) not in (targets-set(idx))
print("CALENDAR: every target date is a real session; the removed holiday never appears")

# ---- NAIVE must score exactly 1.0 skill-neutral ------------------------
rng=np.random.default_rng(7)
walk=100*np.exp(np.cumsum(rng.normal(0,0.01,400)))
wdates=pd.bdate_range("2023-01-02",periods=400)
nrows=bt.walk_forward(walk,wdates,bt.naive,context_len=64,horizon=5,step=3)
s=bt.summarize(nrows,interval="1d")
assert abs(s["skill_vs_naive"])<1e-12, s["skill_vs_naive"]
assert abs(s["mae"]-s["naive_mae"])<1e-12
print(f"NAIVE self-check: skill_vs_naive={s['skill_vs_naive']:.1e}, MASE={s['mase']:.3f}")

# ---- On a random walk, no baseline should show real skill -------------
print("\n  model      MASE   skill   dir.acc   DM p")
for name,fn in bt.BASELINES.items():
    r=bt.walk_forward(walk,wdates,fn,context_len=64,horizon=5,step=3)
    su=bt.summarize(r,interval="1d")
    print(f"  {name:<12} {su['mase']:.3f}  {su['skill_vs_naive']:+.3f}   "
          f"{su['directional_accuracy']:.3f}    {su['dm_pvalue_vs_naive']:.3f}")

# ---- A genuinely informed model must be detected as better ------------
# Positive control: an oracle that is fed the true future. Sliding mode
# truncates the context, so track the planned origins instead of inferring
# the position from len(context) -- that was a bug in this test, not the engine.
_origins=iter(bt.plan_origins(len(walk),context_len=64,horizon=5,step=3))
def oracle(context,horizon):
    o=next(_origins)
    return walk[o+1:o+1+horizon]+rng.normal(0,0.05,horizon),None
orows=bt.walk_forward(walk,wdates,oracle,context_len=64,horizon=5,step=3)
osum=bt.summarize(orows,interval="1d")
assert osum["mase"]<0.2 and osum["dm_pvalue_vs_naive"]<0.01, osum["mase"]
print(f"\nORACLE control: MASE={osum['mase']:.4f}, DM p={osum['dm_pvalue_vs_naive']:.1e} (detected as better)")

# ---- Quantiles flow through to CRPS and coverage ----------------------
levels=list(bt.DEFAULT_QUANTILE_LEVELS)
def qmodel(context,horizon):
    last=float(context[-1])
    # A flat point forecast takes no positions at all, so Sharpe is undefined.
    # Give it a momentum tilt so the economic block has something to trade.
    tilt=float(np.mean(np.diff(context[-10:]))) if len(context)>10 else 0.0
    pt=last+tilt*np.arange(1,horizon+1)
    from scipy import stats as st
    q=np.array([[pt[h]+st.norm.ppf(l)*last*0.01*np.sqrt(h+1) for l in levels] for h in range(horizon)])
    return pt,q
qrows=bt.walk_forward(walk,wdates,qmodel,context_len=64,horizon=5,step=3,quantile_levels=levels)
qs=bt.summarize(qrows,interval="1d")
print(f"\nQUANTILES: CRPS={qs['crps']:.4f}, 80% interval covered {qs['coverage_80']:.1%} (nominal 80%)")
assert 0.5<qs["coverage_80"]<1.0

# ---- Economic block present and costs actually bite -------------------
cheap=bt.summarize(qrows,interval="1d",cost_bps=0.0)
dear=bt.summarize(qrows,interval="1d",cost_bps=100.0)
assert dear["sharpe_net"]<cheap["sharpe_net"], (dear["sharpe_net"],cheap["sharpe_net"])
print(f"COSTS: Sharpe {cheap['sharpe_net']:.2f} at 0bps -> {dear['sharpe_net']:.2f} at 100bps")

s5=bt.summarize(nrows,interval="1d")
print("\nPER-STEP (naive on a random walk) - MASE must grow like sqrt(h):")
for r in s5["by_step"]:
    print(f"  h={r['step']}  MASE={r['mase']:.3f}  expected~{np.sqrt(r['step']):.3f}  skill={r['skill_vs_naive']:+.1e}")
assert abs(s5["by_step"][0]["mase"]-1.0)<0.25, s5["by_step"][0]["mase"]

# ---- COVARIATE walk-forward: the window must not reach past the origin ----
n2 = 300
vals2 = np.arange(100.0, 100.0 + n2)
dates2 = pd.bdate_range("2024-01-02", periods=n2)
# three covariates; everything at/after index 200 is poison
cov = np.vstack([np.arange(n2) * 1.0, np.arange(n2) * 2.0, np.arange(n2) * 3.0])
cov[:, 200:] = 9e9

seen = []
def cov_spy(context, horizon, covariate_window=None):
    assert covariate_window is not None, "covariates were not delivered"
    seen.append((len(context), covariate_window.shape, float(np.max(covariate_window))))
    return np.full(horizon, float(context[-1])), None
cov_spy.wants_covariates = True

crows = bt.walk_forward(vals2, dates2, cov_spy, context_len=64, horizon=5, step=9,
                        covariates=cov)
origins2 = bt.plan_origins(n2, 64, 5, 9)
for origin, (clen, shape, mx) in zip(origins2, seen):
    assert shape == (3, clen), f"covariate window {shape} misaligned with context {clen}"
    assert mx <= cov[:, :origin + 1].max() + 1e-6, f"origin {origin} saw covariate {mx}"
print(f"COVARIATE NO-LEAKAGE: {len(seen)} origins, window always (3, context) and never past the origin")

# a model that does not declare wants_covariates keeps the 2-arg contract
plain_rows = bt.walk_forward(vals2, dates2, bt.naive, 64, 5, 9, covariates=cov)
assert len(plain_rows) == len(crows)
print("baselines still use the two-argument contract")

# shape mistakes are refused loudly rather than silently broadcast
try:
    bt.walk_forward(vals2, dates2, cov_spy, 64, 5, 9, covariates=cov[:, :100])
    raise AssertionError("should have refused a misaligned covariate matrix")
except ValueError as error:
    message = str(error)          # the name is unbound after the except block
    assert "covariates must be" in message
print("misaligned covariate matrix is refused:", message[:60])
print("\nALL BACKTEST TESTS PASSED")
