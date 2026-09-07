import sqlite3 as sq3
import os
import pickle

def init_db():

    os.makedirs('./runs', exist_ok=True)
    conn = sq3.connect('./runs/forecast_history.db')

    # Create the table if it doesn't exist
    conn.execute('''
        CREATE TABLE IF NOT EXISTS forecast_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT NOT NULL,
            interval TEXT NOT NULL,
            context_length INTEGER NOT NULL,
            horizon_length INTEGER NOT NULL,
            model_repo TEXT NOT NULL,
            period INTEGER NOT NULL,
            forecast_data BLOB NOT NULL,
            mae_score REAL
        )
    ''')

    conn.commit()
    conn.close()    

def insert_forecast(ticker, interval, context_length, horizon_length, model_repo, period, forecast_data, mae_score):
    conn = sq3.connect('./runs/forecast_history.db')
    cursor = conn.cursor()

    # Serialize the forecast_data using pickle
    forecast_data = pickle.dumps(forecast_data)

    cursor.execute('''
        INSERT INTO forecast_runs (ticker, interval, context_length, horizon_length, model_repo, period, forecast_data, mae_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, interval, context_length, horizon_length, model_repo, period, forecast_data, mae_score))

    conn.commit()
    conn.close()


init_db()
   

