"""
threshold_calibration.py
SixthSense AI — Fleet Driver Safety & Analytics Platform
Tata Technologies InnoVent Hackathon 2026

Purpose (spec item 6 — PERSONALIZED THRESHOLDS):
    Every driver's face is different — resting eye openness, eyelid shape,
    and natural yawning frequency all vary. A single fixed EAR/MAR
    threshold either misses drowsiness in some drivers or false-alarms
    on others. This module lets the system learn each driver's own
    baseline over their first few sessions and calibrate their
    personalized ear_threshold / mar_threshold in the `drivers` table
    accordingly.

How calibration works:
    - During each session, the live monitoring loop (Bharathi's app.py)
      should log raw per-frame EAR/MAR samples for that session using
      log_calibration_sample() — but if that's too heavy, per-session
      AVERAGE EAR/MAR (baseline_ear / baseline_mar) is enough; see
      record_session_baseline() for the lightweight path actually
      wired into the session-save flow.
    - Once a driver has accumulated CALIBRATION_SESSION_COUNT (5)
      sessions of baseline data, recalculate_thresholds() computes new
      personalized thresholds from the historical average + a safety
      margin, and writes them into drivers.ear_threshold / mar_threshold
      via database.update_driver_thresholds().
    - After calibration, the system keeps quietly re-averaging in the
      background so thresholds slowly adapt (e.g. as a driver ages or
      changes glasses/lighting conditions) without ever needing a
      driver to be manually reconfigured.

Usage:
    import threshold_calibration as calib

    calib.init_calibration_table()   # call once at startup, alongside database.init_db()

    # After each session ends, alongside database.save_session(...):
    calib.record_session_baseline(driver_id, avg_ear=0.27, avg_mar=0.35,
                                    session_id=session_id)

    # This auto-recalibrates once enough sessions exist, but you can also
    # force it manually:
    calib.recalculate_thresholds(driver_id)

    # Check calibration progress (useful for a "profile calibrating..." UI):
    calib.get_calibration_status(driver_id)
"""

import os
import sqlite3
from contextlib import contextmanager

import database as db

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# How many sessions of baseline data to collect before first calibration
CALIBRATION_SESSION_COUNT = 5

# After initial calibration, recompute every N additional sessions so
# thresholds can slowly adapt (fatigue changes with age, new glasses, etc.)
RECALIBRATION_INTERVAL = 10

# Safety margins applied to the driver's own resting average.
# EAR: drowsy eyes read LOWER than resting, so threshold = avg - margin.
# MAR: yawns read HIGHER than resting mouth position, so threshold = avg + margin.
EAR_SAFETY_MARGIN = 0.05
MAR_SAFETY_MARGIN = 0.15

# Guard rails so a noisy calibration can never produce a dangerously
# unusable threshold (e.g. from bad lighting during setup).
EAR_MIN, EAR_MAX = 0.15, 0.32
MAR_MIN, MAR_MAX = 0.35, 0.85

# Reuses the same DB file as database.py — one SQLite file for everything.
DB_PATH = db.DB_PATH


# ---------------------------------------------------------------------------
# CONNECTION (mirrors database.py's pattern)
# ---------------------------------------------------------------------------

@contextmanager
def _get_connection():
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# TABLE SETUP
# ---------------------------------------------------------------------------

def init_calibration_table():
    """
    Creates the calibration_samples table if it doesn't exist.
    Call once at startup, right after database.init_db().

    This table stores one row per session's average EAR/MAR — the raw
    material used to compute each driver's personalized thresholds.
    """
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_samples (
                sample_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id  INTEGER NOT NULL,
                session_id INTEGER,
                avg_ear    REAL NOT NULL,
                avg_mar    REAL NOT NULL,
                recorded_at TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE SET NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calibration_driver
            ON calibration_samples(driver_id)
        """)
    print("[threshold_calibration] calibration_samples table ready.")


# ---------------------------------------------------------------------------
# RECORDING BASELINE SAMPLES
# ---------------------------------------------------------------------------

def record_session_baseline(driver_id, avg_ear, avg_mar, session_id=None,
                             auto_recalibrate=True):
    """
    Record one session's average EAR/MAR as a calibration sample, then
    automatically recalculate thresholds if enough data has accumulated.

    avg_ear / avg_mar should be the driver's mean EAR/MAR across ALL frames
    in a session where they were alert (not already flagged drowsy) —
    Bharathi's monitoring loop is the natural place to compute this running
    average and pass it in when the session ends.

    Returns a dict with the sample id and whether recalibration ran.
    """
    if avg_ear is None or avg_mar is None:
        raise ValueError("avg_ear and avg_mar are required.")

    with _get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO calibration_samples (driver_id, session_id, avg_ear, avg_mar)
            VALUES (?, ?, ?, ?)
        """, (driver_id, session_id, avg_ear, avg_mar))
        sample_id = cur.lastrowid

    recalibrated = False
    if auto_recalibrate:
        status = get_calibration_status(driver_id)
        sample_count = status["sample_count"]

        should_run = (
            sample_count == CALIBRATION_SESSION_COUNT or
            (sample_count > CALIBRATION_SESSION_COUNT and
             (sample_count - CALIBRATION_SESSION_COUNT) % RECALIBRATION_INTERVAL == 0)
        )
        if should_run:
            recalculate_thresholds(driver_id)
            recalibrated = True

    return {"sample_id": sample_id, "recalibrated": recalibrated}


# ---------------------------------------------------------------------------
# CALIBRATION LOGIC
# ---------------------------------------------------------------------------

def recalculate_thresholds(driver_id):
    """
    Compute personalized EAR/MAR thresholds from this driver's accumulated
    calibration samples and write them to the drivers table.

    EAR threshold = driver's average resting EAR - safety margin
                    (eyes closing further than usual = drowsy)
    MAR threshold = driver's average resting MAR + safety margin
                    (mouth opening wider than usual = yawning)

    Both are clamped to sane guard-rail ranges so a noisy calibration
    session can't produce an unusable threshold.

    Returns the new (ear_threshold, mar_threshold), or None if there's
    no calibration data yet for this driver.
    """
    with _get_connection() as conn:
        row = conn.execute("""
            SELECT AVG(avg_ear) AS mean_ear, AVG(avg_mar) AS mean_mar, COUNT(*) AS n
            FROM calibration_samples
            WHERE driver_id = ?
        """, (driver_id,)).fetchone()

    if row is None or row["n"] == 0:
        print(f"[threshold_calibration] No calibration samples yet for driver {driver_id}.")
        return None

    mean_ear = row["mean_ear"]
    mean_mar = row["mean_mar"]

    new_ear = mean_ear - EAR_SAFETY_MARGIN
    new_mar = mean_mar + MAR_SAFETY_MARGIN

    # Clamp to guard rails
    new_ear = max(EAR_MIN, min(EAR_MAX, new_ear))
    new_mar = max(MAR_MIN, min(MAR_MAX, new_mar))

    new_ear = round(new_ear, 4)
    new_mar = round(new_mar, 4)

    updated = db.update_driver_thresholds(
        driver_id, ear_threshold=new_ear, mar_threshold=new_mar)

    if updated:
        print(f"[threshold_calibration] Driver {driver_id} recalibrated "
              f"from {row['n']} sessions -> ear={new_ear}, mar={new_mar}")
    else:
        print(f"[threshold_calibration] WARNING: could not write thresholds "
              f"for driver {driver_id} (driver may not exist).")

    return (new_ear, new_mar)


def get_calibration_status(driver_id):
    """
    Returns calibration progress for a driver — handy for showing a
    "Calibrating... 3/5 sessions" indicator in the UI.

    {
        "sample_count": int,
        "is_calibrated": bool,        # has reached CALIBRATION_SESSION_COUNT
        "sessions_remaining": int,    # 0 once calibrated
        "current_ear_threshold": float,
        "current_mar_threshold": float
    }
    """
    with _get_connection() as conn:
        row = conn.execute("""
            SELECT COUNT(*) AS n FROM calibration_samples WHERE driver_id = ?
        """, (driver_id,)).fetchone()

    sample_count = row["n"] if row else 0
    is_calibrated = sample_count >= CALIBRATION_SESSION_COUNT
    sessions_remaining = max(0, CALIBRATION_SESSION_COUNT - sample_count)

    driver = db.get_driver(driver_id)
    current_ear = driver["ear_threshold"] if driver else None
    current_mar = driver["mar_threshold"] if driver else None

    return {
        "driver_id": driver_id,
        "sample_count": sample_count,
        "is_calibrated": is_calibrated,
        "sessions_remaining": sessions_remaining,
        "current_ear_threshold": current_ear,
        "current_mar_threshold": current_mar,
    }


def reset_calibration(driver_id):
    """
    Wipe a driver's calibration history and reset their thresholds to the
    system defaults. Useful if a driver's camera setup changes drastically
    (new vehicle, different mounting angle) and old samples would mislead
    the new calibration.
    """
    with _get_connection() as conn:
        conn.execute(
            "DELETE FROM calibration_samples WHERE driver_id = ?", (driver_id,))
    db.update_driver_thresholds(driver_id, ear_threshold=0.25, mar_threshold=0.6)
    print(f"[threshold_calibration] Calibration reset for driver {driver_id}.")


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running threshold_calibration.py self-test...\n")

    # Use a throwaway test DB so we don't touch real data
    test_db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_calib_sixthsense.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    db.DB_PATH = test_db_path

    db.init_db()
    init_calibration_table()

    driver_id = db.add_driver("Anita Sharma", 29, "cab", "TN-09-XY-4321")
    print(f"Created test driver: {driver_id}")

    print("\nStep 1: Recording 4 sessions (below calibration threshold)...")
    sample_data = [
        (0.27, 0.34), (0.28, 0.33), (0.26, 0.36), (0.29, 0.32),
    ]
    for i, (ear, mar) in enumerate(sample_data, 1):
        result = record_session_baseline(driver_id, avg_ear=ear, avg_mar=mar)
        status = get_calibration_status(driver_id)
        print(f"  Session {i}: recorded (recalibrated={result['recalibrated']}) "
              f"-> {status['sample_count']}/{CALIBRATION_SESSION_COUNT}, "
              f"is_calibrated={status['is_calibrated']}")

    print("\nStep 2: Recording the 5th session (should trigger auto-calibration)...")
    result = record_session_baseline(driver_id, avg_ear=0.275, avg_mar=0.35)
    status = get_calibration_status(driver_id)
    print(f"  Session 5: recalibrated={result['recalibrated']}")
    print(f"  New thresholds -> ear={status['current_ear_threshold']}, "
          f"mar={status['current_mar_threshold']}")

    assert result["recalibrated"] is True, "Expected auto-calibration to trigger at 5 sessions"
    assert status["is_calibrated"] is True

    print("\nStep 3: Manual recalculate_thresholds() call...")
    new_thresholds = recalculate_thresholds(driver_id)
    print(f"  Manually recalculated -> {new_thresholds}")

    print("\nStep 4: Testing guard rails with an extreme outlier driver...")
    outlier_id = db.add_driver("Test Outlier", 50, "truck", "XX-00-ZZ-0000")
    for _ in range(5):
        record_session_baseline(outlier_id, avg_ear=0.05, avg_mar=1.5)  # unrealistic values
    outlier_status = get_calibration_status(outlier_id)
    print(f"  Outlier thresholds (should be clamped to guard rails) -> "
          f"ear={outlier_status['current_ear_threshold']} "
          f"(min {EAR_MIN}), mar={outlier_status['current_mar_threshold']} "
          f"(max {MAR_MAX})")
    assert outlier_status["current_ear_threshold"] >= EAR_MIN
    assert outlier_status["current_mar_threshold"] <= MAR_MAX

    print("\nStep 5: Testing reset_calibration()...")
    reset_calibration(driver_id)
    reset_status = get_calibration_status(driver_id)
    print(f"  After reset -> sample_count={reset_status['sample_count']}, "
          f"ear={reset_status['current_ear_threshold']}, "
          f"mar={reset_status['current_mar_threshold']}")
    assert reset_status["sample_count"] == 0
    assert reset_status["current_ear_threshold"] == 0.25

    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    print("\nSelf-test complete. All assertions passed.")
