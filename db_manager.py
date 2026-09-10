import sqlite3 as sq3
import os
import json
import pickle
from contextlib import contextmanager

# Anchor the database to this file, not to the process working directory, so the
# app finds the same history no matter where it is launched from.
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
DB_PATH = os.path.join(DB_DIR, 'forecast_history.db')

# Bumped whenever the schema or the blob encoding changes. Tracked in SQLite's
# own PRAGMA user_version so migrations run once, not on every startup.
SCHEMA_VERSION = 3

# Everything the history grid displays. The forecast blob is deliberately absent
# so listing runs does not decode every forecast ever saved.
METADATA_COLUMNS = (
    'id', 'timestamp', 'ticker', 'interval', 'context_length', 'horizon_length',
    'model_repo', 'period', 'target_column', 'anchor_date', 'mae_score',
    'mase_score', 'directional_accuracy', 'feature_columns', 'target_space',
)

BACKTEST_RUN_COLUMNS = (
    'id', 'timestamp', 'ticker', 'interval', 'period', 'model_name',
    'context_length', 'horizon_length', 'step', 'mode', 'adjusted', 'cost_bps',
    'n_origins', 'n_points', 'summary',
)

_CREATE_BACKTEST_RUNS = '''
    CREATE TABLE IF NOT EXISTS backtest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        ticker TEXT NOT NULL,
        interval TEXT NOT NULL,
        period TEXT NOT NULL,
        model_name TEXT NOT NULL,
        context_length INTEGER NOT NULL,
        horizon_length INTEGER NOT NULL,
        step INTEGER NOT NULL,
        mode TEXT NOT NULL,
        adjusted INTEGER NOT NULL,
        cost_bps REAL,
        n_origins INTEGER,
        n_points INTEGER,
        summary TEXT
    )
'''

# One row per (origin, horizon step). Every metric is a GROUP BY over this.
_CREATE_BACKTEST_POINTS = '''
    CREATE TABLE IF NOT EXISTS backtest_points (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
        origin_date TEXT NOT NULL,
        origin_index INTEGER,
        step INTEGER NOT NULL,
        target_date TEXT NOT NULL,
        anchor_value REAL,
        y_true REAL,
        y_pred REAL,
        naive_pred REAL,
        naive_scale REAL,
        quantiles TEXT
    )
'''

_CREATE_TABLE = '''
    CREATE TABLE IF NOT EXISTS forecast_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        ticker TEXT NOT NULL,
        interval TEXT NOT NULL,
        context_length INTEGER NOT NULL,
        horizon_length INTEGER NOT NULL,
        model_repo TEXT NOT NULL,
        period TEXT NOT NULL,
        forecast_data TEXT NOT NULL,
        mae_score REAL,
        target_column TEXT,
        anchor_date TEXT,
        quantiles TEXT,
        mase_score REAL,
        directional_accuracy REAL,
        feature_columns TEXT,
        target_space TEXT
    )
'''


@contextmanager
def _connect():
    """Commit on success, roll back on failure, close either way."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sq3.connect(DB_PATH)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _encode(forecast_data):
    """Forecasts are plain lists of floats, so JSON is enough.

    It is also inspectable in any SQLite browser and, unlike pickle, decoding it
    cannot execute code if the file is ever tampered with or shared.
    """
    return json.dumps([float(value) for value in forecast_data])


def _decode(blob):
    """Read JSON, falling back to pickle for rows written before the switch."""
    if isinstance(blob, (bytes, bytearray)):
        try:
            blob = blob.decode('utf-8')
        except UnicodeDecodeError:
            return pickle.loads(blob)
    try:
        return json.loads(blob)
    except (ValueError, TypeError):
        return pickle.loads(blob)


def _table_exists(conn):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='forecast_runs'"
    ).fetchone() is not None


def _migrate_to_v1(conn):
    """Rebuild the table with correct column types and re-encode blobs as JSON.

    Fixes three things at once, because they all require touching every row:
      - period was declared INTEGER while holding values like 'max' and '1y'
      - forecast_data was declared BLOB and held pickle bytes
      - target_column / anchor_date did not exist
    """
    existing = {row[1] for row in conn.execute('PRAGMA table_info(forecast_runs)')}
    for column in ('target_column', 'anchor_date'):
        if column not in existing:
            conn.execute(f'ALTER TABLE forecast_runs ADD COLUMN {column} TEXT')

    conn.execute('ALTER TABLE forecast_runs RENAME TO forecast_runs_legacy')
    conn.execute(_CREATE_TABLE)

    legacy = {row[1] for row in conn.execute('PRAGMA table_info(forecast_runs_legacy)')}

    # A database written by a different branch may name the anchor
    # 'forecast_origin'. It is the same field, so carry it across rather than
    # leaving every historical row with a NULL anchor and therefore unscorable.
    if 'forecast_origin' in legacy:
        anchor_expression = 'COALESCE(anchor_date, forecast_origin)'
    else:
        anchor_expression = 'anchor_date'

    rows = conn.execute(f'''
        SELECT id, timestamp, ticker, interval, context_length, horizon_length,
               model_repo, period, forecast_data, mae_score, target_column,
               {anchor_expression}
        FROM forecast_runs_legacy
    ''').fetchall()

    for row in rows:
        values = list(row)
        values[8] = _encode(_decode(values[8]))
        conn.execute('''
            INSERT INTO forecast_runs (
                id, timestamp, ticker, interval, context_length, horizon_length,
                model_repo, period, forecast_data, mae_score, target_column, anchor_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)

    conn.execute('DROP TABLE forecast_runs_legacy')


def _migrate_to_v2(conn):
    """Add distributional columns and the backtest tables.

    Plain ALTER TABLE this time: unlike v1 there are no wrong column types to
    correct, so there is no reason to rebuild and risk the existing rows.
    """
    existing = {row[1] for row in conn.execute('PRAGMA table_info(forecast_runs)')}
    for column, decl in (('quantiles', 'TEXT'),
                         ('mase_score', 'REAL'),
                         ('directional_accuracy', 'REAL')):
        if column not in existing:
            conn.execute(f'ALTER TABLE forecast_runs ADD COLUMN {column} {decl}')


def _migrate_to_v3(conn):
    """Record the covariate recipe and modelling space alongside each run.

    Without feature_columns a saved forecast cannot say which of the ~27
    available macro series it was conditioned on, which makes runs impossible
    to reproduce or compare once more than a couple of covariates are in play.
    """
    existing = {row[1] for row in conn.execute('PRAGMA table_info(forecast_runs)')}
    for column in ('feature_columns', 'target_space'):
        if column not in existing:
            conn.execute(f'ALTER TABLE forecast_runs ADD COLUMN {column} TEXT')


def init_db():
    with _connect() as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        version = conn.execute('PRAGMA user_version').fetchone()[0]

        if not _table_exists(conn):
            conn.execute(_CREATE_TABLE)
        else:
            if version < 1:
                _migrate_to_v1(conn)
            if version < 2:
                _migrate_to_v2(conn)
            if version < 3:
                _migrate_to_v3(conn)

        conn.execute(_CREATE_BACKTEST_RUNS)
        conn.execute(_CREATE_BACKTEST_POINTS)
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_forecast_runs_ticker_ts
            ON forecast_runs (ticker, timestamp)
        ''')
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_backtest_points_run
            ON backtest_points (run_id, step)
        ''')
        conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')


def insert_forecast(ticker, interval, context_length, horizon_length, model_repo,
                    period, forecast_data, mae_score=None, target_column=None,
                    anchor_date=None, quantiles=None, feature_columns=None,
                    target_space=None):
    """Persist one run.

    anchor_date is the timestamp of the last historical observation the forecast
    was conditioned on; without it a saved forecast cannot be aligned to actuals
    and therefore cannot be scored.
    """
    with _connect() as conn:
        cursor = conn.execute('''
            INSERT INTO forecast_runs (
                ticker, interval, context_length, horizon_length, model_repo,
                period, forecast_data, mae_score, target_column, anchor_date,
                quantiles, feature_columns, target_space
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, interval, context_length, horizon_length, model_repo, period,
              _encode(forecast_data), mae_score, target_column, anchor_date,
              json.dumps(quantiles) if quantiles is not None else None,
              json.dumps(list(feature_columns)) if feature_columns else None,
              target_space))
        return cursor.lastrowid


def get_forecast_history():
    """Metadata for every run, newest first. Does not read the forecast blobs."""
    columns = ', '.join(METADATA_COLUMNS)
    with _connect() as conn:
        rows = conn.execute(
            f'SELECT {columns} FROM forecast_runs ORDER BY timestamp DESC'
        ).fetchall()

    records = [dict(zip(METADATA_COLUMNS, row)) for row in rows]
    for record in records:
        record['feature_columns'] = _load_list(record.get('feature_columns'))
    return records


def get_forecast_by_id(forecast_id):
    """One full run, including the decoded forecast values."""
    columns = ', '.join(METADATA_COLUMNS)
    with _connect() as conn:
        row = conn.execute(
            f'SELECT {columns}, forecast_data, quantiles FROM forecast_runs WHERE id = ?',
            (int(forecast_id),)
        ).fetchone()

    if row is None:
        return None

    record = dict(zip(METADATA_COLUMNS, row))
    record['forecast_data'] = _decode(row[-2])
    try:
        record['quantiles'] = json.loads(row[-1]) if row[-1] else None
    except (ValueError, TypeError):
        record['quantiles'] = None
    record['feature_columns'] = _load_list(record.get('feature_columns'))
    return record


def _load_list(value):
    if not value:
        return []
    try:
        loaded = json.loads(value)
    except (ValueError, TypeError):
        return []
    return loaded if isinstance(loaded, list) else []


def update_scores(forecast_id, mae_score=None, mase_score=None,
                  directional_accuracy=None):
    """Write computed scores back to a run. Returns True if the row existed.

    MASE is the one that matters: MAE alone cannot say whether the model beat
    a naive forecast, and on price series it almost never does.
    """
    with _connect() as conn:
        cursor = conn.execute('''
            UPDATE forecast_runs
            SET mae_score = ?, mase_score = ?, directional_accuracy = ?
            WHERE id = ?
        ''', (mae_score, mase_score, directional_accuracy, int(forecast_id)))
        return cursor.rowcount > 0


def update_mae_score(forecast_id, mae_score):
    """Backwards-compatible shim for the MAE-only path."""
    return update_scores(forecast_id, mae_score=mae_score)


def count_forecast_runs():
    """How many configurations have been tried.

    This is the trial count the deflated Sharpe ratio needs: the more configs
    you have run, the more the best one is expected to be luck.
    """
    with _connect() as conn:
        return int(conn.execute('SELECT COUNT(*) FROM forecast_runs').fetchone()[0])


# --------------------------------------------------------------------------
# Backtests
# --------------------------------------------------------------------------

def insert_backtest_run(ticker, interval, period, model_name, context_length,
                        horizon_length, step, mode, adjusted, cost_bps,
                        n_origins, n_points, summary):
    with _connect() as conn:
        cursor = conn.execute('''
            INSERT INTO backtest_runs (
                ticker, interval, period, model_name, context_length,
                horizon_length, step, mode, adjusted, cost_bps,
                n_origins, n_points, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, interval, period, model_name, context_length,
              horizon_length, step, mode, int(bool(adjusted)), cost_bps,
              n_origins, n_points,
              # allow_nan=False makes any survivor fail loudly instead of
              # writing a token that only Python can read back.
              json.dumps(_json_sanitize(summary), allow_nan=False) if summary else None))
        return cursor.lastrowid


def _json_sanitize(value):
    """Recursively replace nan/inf with None and unwrap numpy scalars.

    json.dumps happily writes bare NaN / Infinity tokens, which are not valid
    JSON: Python reads them back, but any other reader chokes. A default= hook
    does not help, because json can already serialise floats and so never calls
    it. The values have to be replaced before they reach the encoder.
    """
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if hasattr(value, 'item') and not isinstance(value, (int, float)):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float('inf'), float('-inf')) else None
    return value


def insert_backtest_points(run_id, rows):
    """Bulk-insert per-step rows. One executemany, not one INSERT per point."""
    payload = [(
        int(run_id), row['origin_date'], row.get('origin_index'), int(row['step']),
        row['target_date'], row.get('anchor_value'), row.get('y_true'),
        row.get('y_pred'), row.get('naive_pred'), row.get('naive_scale'),
        json.dumps(row['quantiles']) if row.get('quantiles') else None,
    ) for row in rows]

    with _connect() as conn:
        conn.executemany('''
            INSERT INTO backtest_points (
                run_id, origin_date, origin_index, step, target_date,
                anchor_value, y_true, y_pred, naive_pred, naive_scale, quantiles
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', payload)
    return len(payload)


def count_backtest_runs(prefix=None):
    """How many backtests have been stored, optionally by model-name prefix.

    This is the trial count the deflated Sharpe needs. It must persist across
    sessions: the whole point is that every configuration you have ever tried
    makes the best-looking one more likely to be luck, and a counter that resets
    on restart would quietly flatter every result.
    """
    with _connect() as conn:
        if prefix:
            row = conn.execute(
                'SELECT COUNT(*) FROM backtest_runs WHERE model_name LIKE ?',
                (f'{prefix}%',)
            ).fetchone()
        else:
            row = conn.execute('SELECT COUNT(*) FROM backtest_runs').fetchone()
        return int(row[0])


def get_backtest_runs():
    columns = ', '.join(BACKTEST_RUN_COLUMNS)
    with _connect() as conn:
        rows = conn.execute(
            f'SELECT {columns} FROM backtest_runs ORDER BY timestamp DESC, id DESC'
        ).fetchall()

    out = []
    for row in rows:
        record = dict(zip(BACKTEST_RUN_COLUMNS, row))
        try:
            record['summary'] = json.loads(record['summary']) if record['summary'] else {}
        except (ValueError, TypeError):
            record['summary'] = {}
        out.append(record)
    return out


def get_backtest_points(run_id):
    with _connect() as conn:
        rows = conn.execute('''
            SELECT origin_date, origin_index, step, target_date, anchor_value,
                   y_true, y_pred, naive_pred, naive_scale, quantiles
            FROM backtest_points WHERE run_id = ? ORDER BY origin_index, step
        ''', (int(run_id),)).fetchall()

    keys = ('origin_date', 'origin_index', 'step', 'target_date', 'anchor_value',
            'y_true', 'y_pred', 'naive_pred', 'naive_scale', 'quantiles')
    out = []
    for row in rows:
        record = dict(zip(keys, row))
        try:
            record['quantiles'] = json.loads(record['quantiles']) if record['quantiles'] else None
        except (ValueError, TypeError):
            record['quantiles'] = None
        out.append(record)
    return out


def delete_backtest_run(run_id):
    """Remove a backtest and its points. Returns True if the run existed."""
    with _connect() as conn:
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('DELETE FROM backtest_points WHERE run_id = ?', (int(run_id),))
        cursor = conn.execute('DELETE FROM backtest_runs WHERE id = ?', (int(run_id),))
        return cursor.rowcount > 0


def delete_forecast(forecast_id):
    """Remove a run permanently. Returns True if a row was deleted."""
    with _connect() as conn:
        cursor = conn.execute(
            'DELETE FROM forecast_runs WHERE id = ?', (int(forecast_id),)
        )
        return cursor.rowcount > 0
