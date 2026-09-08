import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, metrics as m

def close(a,b,t=1e-9): assert abs(a-b)<t, f"{a} != {b}"

# MASE: naive scale of [1..5] is 1.0, MAE is 0.5 -> MASE 0.5
close(m.naive_scale([1,2,3,4,5]), 1.0)
close(m.mase([6,7],[6.5,7.5],[1,2,3,4,5]), 0.5)
# A flat series gives no scale -> nan, not a divide-by-zero
assert np.isnan(m.mase([1,2],[1,2],[5,5,5,5]))
print("MASE OK")

# Pinball at the median is half the MAE, by definition
close(m.pinball_loss([1,2,3],[1.5,2.5,3.5],0.5), 0.25)
# Asymmetry: under-predicting is penalised more at a high level
assert m.pinball_loss([10],[8],0.9) > m.pinball_loss([10],[12],0.9)
print("pinball OK")

# A tight correct distribution must beat a wide one on CRPS
levels=[0.1,0.5,0.9]
tight=np.array([[9.5,10.,10.5]]*20); wide=np.array([[5.,10.,15.]]*20)
y=np.full(20,10.)
assert m.crps_from_quantiles(y,tight,levels) < m.crps_from_quantiles(y,wide,levels)
print("CRPS OK")

# Coverage
close(m.interval_coverage([1,2,3,10],[0,0,0,0],[5,5,5,5]), 0.75)
print("coverage OK")

# Direction: anchor 100, truths up/up/down, preds up/down/down -> 2 of 3
close(m.directional_accuracy([101,102,99],[105,95,98],100), 2/3)
# A flat forecast has no directional view: nan, not 0%
assert np.isnan(m.directional_accuracy([101,102,99],[100,100,100],100))
# Steps where the model is flat are excluded, not counted as misses
close(m.directional_accuracy([101,102,99],[105,100,98],100), 1.0)
print("directional OK (flat forecasts excluded, not scored as wrong)")

# Diebold-Mariano: B is plainly better, statistic must be positive and significant
rng=np.random.default_rng(0)
ea=rng.normal(0,3,300); eb=rng.normal(0,1,300)
stat,p=m.diebold_mariano(ea,eb,horizon=1)
assert stat>0 and p<0.01, (stat,p)
# Same errors -> degenerate, must return nan rather than divide by zero
assert np.isnan(m.diebold_mariano(ea,ea)[0])
# Indistinguishable models -> not significant
stat2,p2=m.diebold_mariano(rng.normal(0,1,300),rng.normal(0,1,300))
assert p2>0.05, (stat2,p2)
print(f"DM OK (better model p={p:.2e}, tie p={p2:.2f})")

# Deflated Sharpe: more trials must deflate the same observed Sharpe
d1=m.deflated_sharpe_ratio(0.15,n_trials=1,n_obs=100)
d50=m.deflated_sharpe_ratio(0.15,n_trials=50,n_obs=100)
d5000=m.deflated_sharpe_ratio(0.15,n_trials=5000,n_obs=100)
assert d1>0.9 and d5000<0.05, (d1,d5000)
assert d1>d50>d5000, (d1,d50,d5000)
print(f"deflated Sharpe OK (1 trial {d1:.3f} > 50 {d50:.3f} > 5000 {d5000:.3f})")

# Economic
close(m.max_drawdown([0.1,-0.5,0.2]), -0.5)
net=m.signal_pnl([1,1,0],[0.01,0.02,0.03],cost_bps=10)
np.testing.assert_allclose(net,[0.009,0.02,-0.001],atol=1e-12)
close(m.turnover([1,1,0]), 2/3)
close(m.hit_rate([0.1,-0.2,0.3,0.0]), 2/3)
close(m.profit_factor([1.0,-0.5]), 2.0)
sr=m.sharpe_ratio(np.full(100,0.001))  # zero variance -> nan, not inf
assert np.isnan(sr)
print("economic OK")
print("\nALL METRICS TESTS PASSED")
