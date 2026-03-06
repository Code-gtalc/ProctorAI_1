"""One-time migration: ensure gaze_calibrations table exists in proctorguard.db."""
import sqlite3
from pathlib import Path

db_path = Path(__file__).resolve().parent / "proctorguard.db"
print(f"DB path: {db_path}  (exists={db_path.exists()})")

conn = sqlite3.connect(str(db_path))
conn.execute("""
    CREATE TABLE IF NOT EXISTS gaze_calibrations (
        user_id       TEXT PRIMARY KEY,
        mean_gaze     TEXT NOT NULL,
        inv_cov       TEXT NOT NULL,
        h_threshold   REAL NOT NULL,
        v_threshold   REAL NOT NULL,
        calibrated_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(user_id)
    )
""")
conn.commit()

# Verify
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
conn.close()

print(f"Tables in DB: {tables}")
if "gaze_calibrations" in tables:
    print("SUCCESS: gaze_calibrations table is present.")
else:
    print("FAILED: gaze_calibrations table was NOT created.")

