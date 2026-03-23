import sqlite3
import os

DB_PATH = os.getenv("DB_PATH", "login_history.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS login_history (
                user_id   TEXT PRIMARY KEY,
                latitude  REAL,
                longitude REAL,
                ts        REAL
            )
        """)
        conn.commit()
    # Also initialize activity tracking
    init_activity_db()

def get_last_login(user_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT latitude, longitude, ts FROM login_history WHERE user_id=?",
            (user_id,)
        ).fetchone()

def upsert_login(user_id: str, lat: float, lon: float, ts: float):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO login_history(user_id, latitude, longitude, ts)
            VALUES(?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                latitude=excluded.latitude,
                longitude=excluded.longitude,
                ts=excluded.ts
        """, (user_id, lat, lon, ts))
        conn.commit()

def init_activity_db():
    """Initialize user activity tracking table."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES login_history(user_id)
            )
        """)
        # Create index for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_activity 
            ON user_activity_log(user_id, timestamp DESC)
        """)
        conn.commit()

def get_activity_sequence(user_id: str, limit: int = 15) -> list:
    """Get ordered list of resource IDs for a user's recent activities."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("""
            SELECT resource_id, action_type, timestamp 
            FROM user_activity_log 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (user_id, limit))
        activities = cursor.fetchall()
        # Return in chronological order (oldest first)
        return [(activity[0], activity[1]) for activity in reversed(activities)]

def log_resource_access(user_id: str, resource_id: str, action_type: str):
    """Record a resource access event for continuous monitoring."""
    import time
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO user_activity_log(user_id, resource_id, action_type, timestamp)
            VALUES(?, ?, ?, ?)
        """, (user_id, resource_id, action_type, time.time()))
        conn.commit()
