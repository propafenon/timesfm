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
SCHEMA_VERSION = 1

# Everything the history grid displays. The forecast blob is deliberately absent
# so listing runs does not decode every forecast ever saved.
METADATA_COLUMNS = (
    'id', 'timestamp', 'ticker', 'interval', 'context_length', 'horizon_length',
    'model_repo', 'period', 'target_column', 'anchor_date', 'mae_score',
)

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
        anchor_date TEXT
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

    rows = conn.execute('''
        SELECT id, timestamp, ticker, interval, context_length, horizon_length,
               model_repo, period, forecast_data, mae_score, target_column, anchor_date
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


def init_db():
    with _connect() as conn:
        version = conn.execute('PRAGMA user_version').fetchone()[0]

        if not _table_exists(conn):
            conn.execute(_CREATE_TABLE)
        elif version < 1:
            _migrate_to_v1(conn)

        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_forecast_runs_ticker_ts
            ON forecast_runs (ticker, timestamp)
        ''')
        conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')


def insert_forecast(ticker, interval, context_length, horizon_length, model_repo,
                    period, forecast_data, mae_score=None, target_column=None,
                    anchor_date=None):
    """Persist one run.

    anchor_date is the timestamp of the last historical observation the forecast
    was conditioned on; without it a saved forecast cannot be aligned to actuals
    and therefore cannot be scored.
    """
    with _connect() as conn:
        cursor = conn.execute('''
            INSERT INTO forecast_runs (
                ticker, interval, context_length, horizon_length, model_repo,
                period, forecast_data, mae_score, target_column, anchor_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, interval, context_length, horizon_length, model_repo, period,
              _encode(forecast_data), mae_score, target_column, anchor_date))
        return cursor.lastrowid


def get_forecast_history():
    """Metadata for every run, newest first. Does not read the forecast blobs."""
    columns = ', '.join(METADATA_COLUMNS)
    with _connect() as conn:
        rows = conn.execute(
            f'SELECT {columns} FROM forecast_runs ORDER BY timestamp DESC'
        ).fetchall()

    return [dict(zip(METADATA_COLUMNS, row)) for row in rows]


def get_forecast_by_id(forecast_id):
    """One full run, including the decoded forecast values."""
    columns = ', '.join(METADATA_COLUMNS)
    with _connect() as conn:
        row = conn.execute(
            f'SELECT {columns}, forecast_data FROM forecast_runs WHERE id = ?',
            (int(forecast_id),)
        ).fetchone()

    if row is None:
        return None

    record = dict(zip(METADATA_COLUMNS, row))
    record['forecast_data'] = _decode(row[-1])
    return record


def update_mae_score(forecast_id, mae_score):
    """Write a computed score back to a run. Returns True if the row existed."""
    with _connect() as conn:
        cursor = conn.execute(
            'UPDATE forecast_runs SET mae_score = ? WHERE id = ?',
            (mae_score, int(forecast_id))
        )
        return cursor.rowcount > 0


def delete_forecast(forecast_id):
    """Remove a run permanently. Returns True if a row was deleted."""
    with _connect() as conn:
        cursor = conn.execute(
            'DELETE FROM forecast_runs WHERE id = ?', (int(forecast_id),)
        )
        return cursor.rowcount > 0
