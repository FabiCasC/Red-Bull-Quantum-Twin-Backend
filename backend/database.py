# SQLite for local dev — Oracle Autonomous Database (OCI) in production
import sqlite3
import os

DB_PATH = 'db/quantum_twin.db'


def get_connection():
    os.makedirs('db', exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    # Oracle Autonomous Database (OCI) in production
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS telemetry_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lap INTEGER, driver TEXT, compound TEXT,
            tyre_life INTEGER, lap_time REAL,
            position INTEGER, stint INTEGER,
            speed REAL, throttle REAL, brake REAL, rpm REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS superclipping_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_laptime REAL, status TEXT,
            velocidad REAL, soc_bateria REAL, mguk_kw REAL,
            throttle_pct REAL, aero_mode INTEGER, temp_bateria REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            severity TEXT, title TEXT, description TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def insert_prediction(data: dict):
    conn = get_connection()
    # Oracle Autonomous Database (OCI) in production
    conn.execute("""
        INSERT INTO superclipping_predictions
        (predicted_laptime, status, velocidad, soc_bateria, mguk_kw, throttle_pct, aero_mode, temp_bateria)
        VALUES (:predicted_laptime, :status, :velocidad, :soc_bateria, :mguk_kw, :throttle_pct, :aero_mode, :temp_bateria)
    """, data)
    conn.commit()
    conn.close()


def insert_alert(data: dict):
    conn = get_connection()
    # Oracle Autonomous Database (OCI) in production
    conn.execute("""
        INSERT INTO alert_history (severity, title, description)
        VALUES (:severity, :title, :description)
    """, data)
    conn.commit()
    conn.close()


def get_recent_predictions(limit=10):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    # Oracle Autonomous Database (OCI) in production
    rows = conn.execute(
        "SELECT * FROM superclipping_predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recent_alerts(limit=10):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    # Oracle Autonomous Database (OCI) in production
    rows = conn.execute(
        "SELECT * FROM alert_history ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
