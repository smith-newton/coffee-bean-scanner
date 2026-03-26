import sqlite3
import os

def init_db():
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)
    
    db_path = os.path.join(data_dir, "logs.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        predicted_class TEXT,
        confidence REAL,
        source TEXT
    )
    """)

    conn.commit()
    conn.close()

def insert_log(timestamp, pred_class, confidence, source):
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    db_path = os.path.join(data_dir, "logs.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
    INSERT INTO predictions (timestamp, predicted_class, confidence, source)
    VALUES (?, ?, ?, ?)
    """, (timestamp, pred_class, confidence, source))

    conn.commit()
    conn.close()