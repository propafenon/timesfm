import os, sys, sqlite3, pickle, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_manager as db

tmp = tempfile.mkdtemp()
db.DB_DIR = tmp
db.DB_PATH = os.path.join(tmp, 'forecast_history.db')

# 1. Simulate a PRE-migration database written by the old code.
conn = sqlite3.connect(db.DB_PATH)
conn.execute('''CREATE TABLE forecast_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL, interval TEXT NOT NULL,
    context_length INTEGER NOT NULL, horizon_length INTEGER NOT NULL,
    model_repo TEXT NOT NULL, period INTEGER NOT NULL,
    forecast_data BLOB NOT NULL, mae_score REAL)''')
conn.execute('INSERT INTO forecast_runs (ticker,interval,context_length,horizon_length,model_repo,period,forecast_data,mae_score) VALUES (?,?,?,?,?,?,?,?)',
             ('LEGACY.IS','1d',1056,3,'google/timesfm-3.0-pytorch','max', pickle.dumps([1.0,2.0,3.0]), None))
conn.commit(); conn.close()

db.init_db()          # must migrate in place
db.init_db()          # must be idempotent

cols = [r[1] for r in sqlite3.connect(db.DB_PATH).execute('PRAGMA table_info(forecast_runs)')]
assert 'anchor_date' in cols and 'target_column' in cols, cols
print("migration OK ->", cols)

legacy = db.get_forecast_by_id(1)
assert legacy['ticker'] == 'LEGACY.IS' and legacy['forecast_data'] == [1.0,2.0,3.0]
assert legacy['anchor_date'] is None and legacy['target_column'] is None
print("legacy row survived, anchor NULL as expected")

# 2. New-style insert
db.insert_forecast('ASELS.IS','1d',1056,3,'google/timesfm-3.0-pytorch','max',
                   [10.5,11.0,10.75], target_column='Close',
                   anchor_date='2026-09-04T00:00:00')
rec = db.get_forecast_by_id(2)
assert rec['target_column']=='Close' and rec['anchor_date']=='2026-09-04T00:00:00'
assert rec['forecast_data']==[10.5,11.0,10.75]
print("insert/get round-trip OK")

# 3. History listing must NOT carry blobs
hist = db.get_forecast_history()
assert len(hist)==2 and 'forecast_data' not in hist[0], hist[0].keys()
print("list view excludes blob OK ->", sorted(hist[0].keys()))

# 4. MAE write-back + delete
assert db.update_mae_score(2, 0.4213) is True
assert db.update_mae_score(999, 1.0) is False
assert db.get_forecast_by_id(2)['mae_score'] == 0.4213
assert db.delete_forecast(1) is True
assert db.delete_forecast(1) is False
assert len(db.get_forecast_history()) == 1
print("update_mae_score + delete_forecast OK")

# 5. Connections must not leak on failure
try:
    db.insert_forecast(None,'1d',1,1,'r','max',[1.0])   # ticker NOT NULL -> IntegrityError
except sqlite3.IntegrityError:
    print("failed insert raised and rolled back OK")
assert len(db.get_forecast_history()) == 1, "rollback did not happen"

# 6. String id from the treeview must still resolve
assert db.get_forecast_by_id('2')['id'] == 2
print("string id from treeview OK")

shutil.rmtree(tmp)
print("\nALL DB TESTS PASSED")
