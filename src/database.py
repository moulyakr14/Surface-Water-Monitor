import sqlite3
import pandas as pd
from src.config import DB_PATH, DATA_DIR

def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                site TEXT,
                ph REAL, turbidity_ntu REAL, tds_mg_l REAL,
                temperature_c REAL, dissolved_oxygen_mg_l REAL,
                water_level_m REAL, wqi REAL, quality_category TEXT,
                anomaly INTEGER DEFAULT 0
            )
        ''')
        conn.commit()

def save_readings(df):
    init_db()
    cols = ["timestamp", "site", "ph", "turbidity_ntu", "tds_mg_l",
            "temperature_c", "dissolved_oxygen_mg_l", "water_level_m",
            "wqi", "quality_category", "anomaly"]
    data = df.copy()
    for c in cols:
        if c not in data.columns:
            data[c] = 0 if c == "anomaly" else None
    data[cols].to_sql("readings", get_connection(), if_exists="append", index=False)

def load_readings(limit=10000):
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM readings ORDER BY timestamp DESC LIMIT ?",
            conn, params=(int(limit),)
        ).sort_values("timestamp")
