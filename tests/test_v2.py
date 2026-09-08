import os,sys,sqlite3,pickle,json,tempfile,shutil
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_manager as db

ORIGINAL='''CREATE TABLE forecast_runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
 ticker TEXT NOT NULL, interval TEXT NOT NULL, context_length INTEGER NOT NULL,
 horizon_length INTEGER NOT NULL, model_repo TEXT NOT NULL, period INTEGER NOT NULL,
 forecast_data BLOB NOT NULL, mae_score REAL)'''

def tmpdb():
    t=tempfile.mkdtemp(); db.DB_DIR=t; db.DB_PATH=os.path.join(t,'t.db'); return t
def uv(): return sqlite3.connect(db.DB_PATH).execute('PRAGMA user_version').fetchone()[0]

# v0 (original, pickle) -> v2 in one shot
t=tmpdb()
c=sqlite3.connect(db.DB_PATH); c.execute(ORIGINAL)
c.execute('INSERT INTO forecast_runs (ticker,interval,context_length,horizon_length,model_repo,period,forecast_data,mae_score) VALUES (?,?,?,?,?,?,?,?)',
          ('OLD.IS','1d',1056,3,'repo','max',pickle.dumps([1.,2.,3.]),0.5))
c.commit(); c.close()
db.init_db(); db.init_db()
assert uv()==2, uv()
r=db.get_forecast_by_id(1)
assert r['forecast_data']==[1.,2.,3.] and r['mae_score']==0.5 and r['quantiles'] is None
assert r['mase_score'] is None
print("v0 -> v2 direct OK; legacy pickle row intact")

# quantiles round-trip
q={"levels":[0.1,0.5,0.9],"values":[[9.,10.,11.],[8.,10.,12.]]}
i=db.insert_forecast('A.IS','1d',64,2,'repo','1y',[10.,10.],target_column='Close',
                     anchor_date='2026-09-04T00:00:00',quantiles=q)
assert db.get_forecast_by_id(i)['quantiles']==q
print("quantiles round-trip OK")

# scores
assert db.update_scores(i,mae_score=0.4,mase_score=1.02,directional_accuracy=0.55)
h=[x for x in db.get_forecast_history() if x['id']==i][0]
assert h['mase_score']==1.02 and h['directional_accuracy']==0.55
assert db.update_mae_score(i,0.9)   # legacy shim still works
print("scores + legacy shim OK")
assert db.count_forecast_runs()==2
print("trial count for deflated Sharpe:", db.count_forecast_runs())

# backtest run + bulk points
rid=db.insert_backtest_run('A.IS','1d','5y','timesfm',64,5,3,'sliding',False,10.0,
                           120,600,{'mase':1.01,'sharpe_net':float('nan'),'x':None})
rows=[{'origin_date':f'2026-01-{d:02d}','origin_index':d,'step':s,
       'target_date':f'2026-02-{d:02d}','anchor_value':100.+d,'y_true':101.+d,
       'y_pred':100.5+d,'naive_pred':100.+d,'naive_scale':1.0,
       'quantiles':{'0.5':100.5+d} if s==1 else None}
      for d in range(1,21) for s in range(1,6)]
assert db.insert_backtest_points(rid,rows)==100
got=db.get_backtest_points(rid)
assert len(got)==100 and got[0]['step']==1 and got[0]['quantiles']=={'0.5':101.5}
runs=db.get_backtest_runs()
assert runs[0]['id']==rid and runs[0]['summary']['mase']==1.01
assert runs[0]['summary']['sharpe_net'] is None   # nan sanitised, not "NaN"
print(f"backtest run + {len(got)} points OK; nan sanitised to null")

# cascade delete
assert db.delete_backtest_run(rid) and db.get_backtest_points(rid)==[]
assert db.delete_backtest_run(rid) is False
print("cascade delete OK")
shutil.rmtree(t)
print("\nALL V2 TESTS PASSED")
