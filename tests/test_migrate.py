import os, sys, sqlite3, pickle, json, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_manager as db

ORIGINAL = '''CREATE TABLE forecast_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL, interval TEXT NOT NULL,
    context_length INTEGER NOT NULL, horizon_length INTEGER NOT NULL,
    model_repo TEXT NOT NULL, period INTEGER NOT NULL,
    forecast_data BLOB NOT NULL, mae_score REAL)'''

def fresh_tmp():
    t = tempfile.mkdtemp()
    db.DB_DIR = t
    db.DB_PATH = os.path.join(t, 'forecast_history.db')
    return t

def types_of():
    return {r[1]: r[2] for r in sqlite3.connect(db.DB_PATH).execute('PRAGMA table_info(forecast_runs)')}

def uv():
    return sqlite3.connect(db.DB_PATH).execute('PRAGMA user_version').fetchone()[0]

# ---- PATH A: original v0 schema, pickled blobs -------------------------
t = fresh_tmp()
c = sqlite3.connect(db.DB_PATH); c.execute(ORIGINAL)
for i, vals in enumerate([[1.0,2.0,3.0], [9.5,8.25]], start=1):
    c.execute('INSERT INTO forecast_runs (ticker,interval,context_length,horizon_length,model_repo,period,forecast_data,mae_score) VALUES (?,?,?,?,?,?,?,?)',
              (f'OLD{i}.IS','1d',1056,len(vals),'repo','max', pickle.dumps(vals), None))
c.commit(); c.close()

db.init_db(); db.init_db()          # migrate, then prove idempotent
assert uv() == 3, uv()
ty = types_of()
assert ty['period'] == 'TEXT', ty['period']
assert ty['forecast_data'] == 'TEXT', ty['forecast_data']
assert ty['anchor_date'] == 'TEXT'
assert db.get_forecast_by_id(1)['forecast_data'] == [1.0,2.0,3.0]
assert db.get_forecast_by_id(2)['forecast_data'] == [9.5,8.25]
assert db.get_forecast_by_id(1)['ticker'] == 'OLD1.IS'
assert [r['id'] for r in db.get_forecast_history()] == [2,1] or len(db.get_forecast_history())==2
raw = sqlite3.connect(db.DB_PATH).execute('SELECT forecast_data FROM forecast_runs WHERE id=1').fetchone()[0]
assert json.loads(raw) == [1.0,2.0,3.0], raw
print("PATH A (v0 pickle -> v1 json) OK; ids and values preserved, stored as", repr(raw))
shutil.rmtree(t)

# ---- PATH B: the intermediate state my previous commit produced --------
t = fresh_tmp()
c = sqlite3.connect(db.DB_PATH); c.execute(ORIGINAL)
c.execute('ALTER TABLE forecast_runs ADD COLUMN target_column TEXT')
c.execute('ALTER TABLE forecast_runs ADD COLUMN anchor_date TEXT')
c.execute('INSERT INTO forecast_runs (ticker,interval,context_length,horizon_length,model_repo,period,forecast_data,mae_score,target_column,anchor_date) VALUES (?,?,?,?,?,?,?,?,?,?)',
          ('MID.IS','1d',1056,3,'repo','max', pickle.dumps([5.0,6.0,7.0]), 0.42,'Close','2026-09-04T00:00:00'))
c.commit(); c.close()
db.init_db()
r = db.get_forecast_by_id(1)
assert r['forecast_data'] == [5.0,6.0,7.0] and r['mae_score'] == 0.42
assert r['target_column'] == 'Close' and r['anchor_date'] == '2026-09-04T00:00:00'
assert types_of()["period"] == "TEXT" and uv() == 3
print("PATH B (columns present, still pickle) OK; mae/anchor/target preserved")
shutil.rmtree(t)

# ---- PATH C: brand new database ---------------------------------------
t = fresh_tmp()
db.init_db()
assert uv() == 3 and types_of()['forecast_data'] == 'TEXT'
new_id = db.insert_forecast('ASELS.IS','1d',1056,3,'repo','max',[10.5,11.0,10.75],
                            target_column='Close', anchor_date='2026-09-04T00:00:00')
assert new_id == 1
assert db.get_forecast_by_id(new_id)['forecast_data'] == [10.5,11.0,10.75]
import numpy as np
assert db.get_forecast_by_id(db.insert_forecast("N.IS","1d",1,2,"r","max", np.array([1.5,2.5],dtype=np.float32)))["forecast_data"] == [1.5,2.5]
print("PATH C (fresh db) OK; insert returned id", new_id)
shutil.rmtree(t)
print("\nALL MIGRATION TESTS PASSED")
