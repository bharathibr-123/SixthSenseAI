"""
database.py
SixthSense AI — Fleet Driver Safety & Analytics Platform
Tata Technologies InnoVent Hackathon 2026

Handles all SQLite database operations:
- Table creation (drivers, sessions, alerts, driver_analytics)
- CRUD functions for drivers
- Session saving / retrieval
- Alert logging
- Driver analytics (weekly aggregation)

Usage (from app.py or api_routes.py):
    import database as db
    db.init_db()
    db.add_driver("Ramesh Kumar", 34, "truck", "KA-01-AB-1234")
"""

import sqlite3
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DB_NAME = "sixthsense.db"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_NAME)


# ---------------------------------------------------------------------------
# CONNECTION HELPERS
# ---------------------------------------------------------------------------

@contextmanager
def get_connection():
    """
    Context-managed SQLite connection.
    Ensures connections are always closed and commits/rollbacks happen safely.
    Use like:
        with get_connection() as conn:
            conn.execute(...)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # lets us access columns by name
    conn.execute("PRAGMA foreign_keys = ON")  # enforce FK constraints
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row):
    """Convert a sqlite3.Row (or None) into a plain dict (or None)."""
    return dict(row) if row is not None else None


def rows_to_list(rows):
    """Convert a list of sqlite3.Row into a list of dicts."""
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# TABLE CREATION
# ---------------------------------------------------------------------------

def init_db():
    """
    Creates all required tables if they do not already exist.
    Safe to call every time the app starts.
    """
    with get_connection() as conn:
        cur = conn.cursor()

        # ---------------- drivers ----------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                driver_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_name       TEXT NOT NULL,
                age               INTEGER,
                vehicle_type      TEXT CHECK(vehicle_type IN
                                    ('truck','bus','cab','mining')),
                vehicle_number    TEXT,
                profile_photo_path TEXT,
                ear_threshold     REAL DEFAULT 0.25,
                mar_threshold     REAL DEFAULT 0.6,
                created_at        TEXT DEFAULT (datetime('now','localtime'))
            )
        """)

        # ---------------- sessions ----------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id        INTEGER NOT NULL,
                session_date     TEXT DEFAULT (datetime('now','localtime')),
                session_duration INTEGER DEFAULT 0,      -- in seconds
                peak_score       INTEGER DEFAULT 0,      -- 0-100
                total_alerts     INTEGER DEFAULT 0,
                yawn_count       INTEGER DEFAULT 0,
                blink_count      INTEGER DEFAULT 0,
                final_level      TEXT CHECK(final_level IN
                                    ('SAFE','WARNING','DANGER')) DEFAULT 'SAFE',
                snapshots_count  INTEGER DEFAULT 0,
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
                    ON DELETE CASCADE
            )
        """)

        # ---------------- alerts ----------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   INTEGER NOT NULL,
                driver_id    INTEGER NOT NULL,
                alert_type   TEXT CHECK(alert_type IN
                                ('DROWSY','FATIGUE','WARNING','NO_FACE')),
                alert_time   TEXT DEFAULT (datetime('now','localtime')),
                risk_score   INTEGER,
                location_lat REAL,
                location_lng REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
                    ON DELETE CASCADE
            )
        """)

        # ---------------- driver_analytics ----------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS driver_analytics (
                analytics_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id      INTEGER NOT NULL,
                week_number    INTEGER,
                avg_risk_score REAL DEFAULT 0,
                total_sessions INTEGER DEFAULT 0,
                total_alerts   INTEGER DEFAULT 0,
                fatigue_trend  TEXT CHECK(fatigue_trend IN
                                ('IMPROVING','STABLE','WORSENING')) DEFAULT 'STABLE',
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
                    ON DELETE CASCADE
            )
        """)

        # Helpful indexes for common lookups
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_driver ON sessions(driver_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_session ON alerts(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_alerts_driver ON alerts(driver_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_analytics_driver ON driver_analytics(driver_id)")

        _migrate_driver_columns(cur)
        _migrate_alert_columns(cur)

    print(f"[database.py] Database ready at: {DB_PATH}")


def _migrate_alert_columns(cur):
    """Adds alerts.snapshot_path and alerts.video_clip_path, introduced
    after the original schema, so DANGER-event snapshots/clips can be
    linked back to their alert record and served to the frontend (see
    app.py:/snapshots/<filename> and /clips/<filename>)."""
    cur.execute("PRAGMA table_info(alerts)")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "snapshot_path" not in existing_cols:
        cur.execute("ALTER TABLE alerts ADD COLUMN snapshot_path TEXT")
        print("[database.py] Migrated: added alerts.snapshot_path")
    if "video_clip_path" not in existing_cols:
        cur.execute("ALTER TABLE alerts ADD COLUMN video_clip_path TEXT")
        print("[database.py] Migrated: added alerts.video_clip_path")


def _migrate_driver_columns(cur):
    """
    Lightweight migration: adds columns to `drivers` that were introduced
    after the original schema, so existing databases upgrade in place
    instead of needing to be deleted and recreated.

    These columns exist to satisfy the frontend's driver-profile contract
    (src/api/driverApi.js -> getDriverProfile/updateDriverProfile), which
    exposes four named 0-100 "sensitivity" sliders rather than raw
    EAR/MAR floats, plus the fleet map's live lat/lng per driver.
    """
    cur.execute("PRAGMA table_info(drivers)")
    existing_cols = {row[1] for row in cur.fetchall()}

    new_columns = {
        "drowsiness_threshold": "REAL DEFAULT 50",
        "distraction_threshold": "REAL DEFAULT 50",
        "yawn_threshold": "REAL DEFAULT 50",
        "blink_threshold": "REAL DEFAULT 50",
        "last_lat": "REAL",
        "last_lng": "REAL",
        "preferred_language": "TEXT DEFAULT 'en'",
        "emergency_contacts": "TEXT",  # comma-separated WhatsApp numbers, e.g. "whatsapp:+91..., whatsapp:+91..."
    }
    for col, decl in new_columns.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE drivers ADD COLUMN {col} {decl}")
            print(f"[database.py] Migrated: added drivers.{col}")


# ---------------------------------------------------------------------------
# DRIVER CRUD
# ---------------------------------------------------------------------------

def add_driver(driver_name, age, vehicle_type, vehicle_number,
                profile_photo_path=None, ear_threshold=0.25, mar_threshold=0.6):
    """
    Insert a new driver. Returns the new driver_id.
    """
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO drivers
                (driver_name, age, vehicle_type, vehicle_number,
                 profile_photo_path, ear_threshold, mar_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (driver_name, age, vehicle_type, vehicle_number,
              profile_photo_path, ear_threshold, mar_threshold))
        return cur.lastrowid


def get_driver(driver_id):
    """
    Fetch a single driver profile by ID. Returns dict or None.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM drivers WHERE driver_id = ?", (driver_id,)
        ).fetchone()
        return row_to_dict(row)


def get_all_drivers():
    """
    Fetch the full fleet driver list, most recently added first.
    """
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM drivers ORDER BY created_at DESC"
        ).fetchall()
        return rows_to_list(rows)


def update_driver(driver_id, **fields):
    """
    Generic driver update. Pass only the fields you want to change, e.g.:
        update_driver(3, driver_name="New Name", age=40)
    Returns True if a row was updated.
    """
    allowed = {"driver_name", "age", "vehicle_type", "vehicle_number",
               "profile_photo_path", "ear_threshold", "mar_threshold",
               "drowsiness_threshold", "distraction_threshold",
               "yawn_threshold", "blink_threshold", "last_lat", "last_lng",
               "preferred_language", "emergency_contacts"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [driver_id]

    with get_connection() as conn:
        cur = conn.execute(
            f"UPDATE drivers SET {set_clause} WHERE driver_id = ?", values
        )
        return cur.rowcount > 0


def update_driver_thresholds(driver_id, ear_threshold=None, mar_threshold=None):
    """
    Dedicated helper for PUT /api/driver/{id}/thresholds.
    Only updates the thresholds that are provided (not None).
    """
    fields = {}
    if ear_threshold is not None:
        fields["ear_threshold"] = ear_threshold
    if mar_threshold is not None:
        fields["mar_threshold"] = mar_threshold
    if not fields:
        return False
    return update_driver(driver_id, **fields)


def delete_driver(driver_id):
    """
    Delete a driver and (via ON DELETE CASCADE) their sessions/alerts/analytics.
    Returns True if a row was deleted.
    """
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM drivers WHERE driver_id = ?", (driver_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# SESSION CRUD
# ---------------------------------------------------------------------------

def save_session(driver_id, session_duration=0, peak_score=0, total_alerts=0,
                  yawn_count=0, blink_count=0, final_level="SAFE",
                  snapshots_count=0, session_date=None):
    """
    Save a completed monitoring session. Returns the new session_id.
    session_date defaults to now if not provided (pass ISO string to override).
    """
    with get_connection() as conn:
        if session_date:
            cur = conn.execute("""
                INSERT INTO sessions
                    (driver_id, session_date, session_duration, peak_score,
                     total_alerts, yawn_count, blink_count, final_level, snapshots_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (driver_id, session_date, session_duration, peak_score,
                  total_alerts, yawn_count, blink_count, final_level, snapshots_count))
        else:
            cur = conn.execute("""
                INSERT INTO sessions
                    (driver_id, session_duration, peak_score,
                     total_alerts, yawn_count, blink_count, final_level, snapshots_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (driver_id, session_duration, peak_score,
                  total_alerts, yawn_count, blink_count, final_level, snapshots_count))
        return cur.lastrowid


def get_session(session_id):
    """Fetch a single session by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row_to_dict(row)


def get_sessions_for_driver(driver_id, limit=None):
    """
    Fetch all sessions for a driver, most recent first.
    Optionally limit the number of rows returned.
    """
    query = "SELECT * FROM sessions WHERE driver_id = ? ORDER BY session_date DESC"
    params = [driver_id]
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return rows_to_list(rows)


# ---------------------------------------------------------------------------
# ALERT CRUD
# ---------------------------------------------------------------------------

def save_alert(session_id, driver_id, alert_type, risk_score,
                location_lat=None, location_lng=None, alert_time=None,
                snapshot_path=None, video_clip_path=None):
    """
    Log a single alert event (DROWSY/FATIGUE/WARNING/NO_FACE).
    snapshot_path, if given, is the filename (not full URL) saved under
    app.py's SNAPSHOTS_DIR — served back via GET /snapshots/<filename>.
    video_clip_path, if given, is the filename saved under app.py's
    CLIPS_DIR — served back via GET /clips/<filename>.
    Returns the new alert_id.
    """
    with get_connection() as conn:
        if alert_time:
            cur = conn.execute("""
                INSERT INTO alerts
                    (session_id, driver_id, alert_type, alert_time,
                     risk_score, location_lat, location_lng, snapshot_path, video_clip_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, driver_id, alert_type, alert_time,
                  risk_score, location_lat, location_lng, snapshot_path, video_clip_path))
        else:
            cur = conn.execute("""
                INSERT INTO alerts
                    (session_id, driver_id, alert_type,
                     risk_score, location_lat, location_lng, snapshot_path, video_clip_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, driver_id, alert_type,
                  risk_score, location_lat, location_lng, snapshot_path, video_clip_path))
        return cur.lastrowid



def get_alerts_for_session(session_id):
    """Fetch all alerts belonging to one session, chronological order."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE session_id = ? ORDER BY alert_time ASC",
            (session_id,)
        ).fetchall()
        return rows_to_list(rows)


def get_alerts_for_driver(driver_id, limit=None):
    """Fetch all alerts for a driver across all sessions, most recent first."""
    query = "SELECT * FROM alerts WHERE driver_id = ? ORDER BY alert_time DESC"
    params = [driver_id]
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return rows_to_list(rows)


# ---------------------------------------------------------------------------
# DRIVER ANALYTICS
# ---------------------------------------------------------------------------

def _get_iso_week(date_str=None):
    """Return ISO week number for a given date string (or today)."""
    if date_str:
        dt = datetime.fromisoformat(date_str.split(".")[0])
    else:
        dt = datetime.now()
    return dt.isocalendar()[1]  # (year, week, weekday) -> week


def recompute_weekly_analytics(driver_id, week_number=None):
    """
    Recalculate and upsert this driver's analytics row for a given ISO week
    (defaults to the current week) based on their sessions/alerts data.
    Call this after saving a session so the dashboard stays current.
    """
    if week_number is None:
        week_number = _get_iso_week()

    with get_connection() as conn:
        # Aggregate sessions that fall in this ISO week
        sessions = conn.execute(
            "SELECT * FROM sessions WHERE driver_id = ?", (driver_id,)
        ).fetchall()

        week_sessions = [
            s for s in sessions
            if _get_iso_week(s["session_date"]) == week_number
        ]

        total_sessions = len(week_sessions)
        total_alerts = sum(s["total_alerts"] for s in week_sessions)
        avg_risk_score = (
            sum(s["peak_score"] for s in week_sessions) / total_sessions
            if total_sessions > 0 else 0
        )

        # Simple trend logic: compare this week's avg to previous week's avg
        prev_row = conn.execute("""
            SELECT avg_risk_score FROM driver_analytics
            WHERE driver_id = ? AND week_number = ?
        """, (driver_id, week_number - 1)).fetchone()

        fatigue_trend = "STABLE"
        if prev_row is not None:
            prev_avg = prev_row["avg_risk_score"]
            if avg_risk_score > prev_avg + 5:
                fatigue_trend = "WORSENING"
            elif avg_risk_score < prev_avg - 5:
                fatigue_trend = "IMPROVING"

        # Upsert: does a row already exist for this driver + week?
        existing = conn.execute("""
            SELECT analytics_id FROM driver_analytics
            WHERE driver_id = ? AND week_number = ?
        """, (driver_id, week_number)).fetchone()

        if existing:
            conn.execute("""
                UPDATE driver_analytics
                SET avg_risk_score = ?, total_sessions = ?,
                    total_alerts = ?, fatigue_trend = ?
                WHERE analytics_id = ?
            """, (avg_risk_score, total_sessions, total_alerts,
                  fatigue_trend, existing["analytics_id"]))
            return existing["analytics_id"]
        else:
            cur = conn.execute("""
                INSERT INTO driver_analytics
                    (driver_id, week_number, avg_risk_score,
                     total_sessions, total_alerts, fatigue_trend)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (driver_id, week_number, avg_risk_score,
                  total_sessions, total_alerts, fatigue_trend))
            return cur.lastrowid


def get_safety_score(driver_id):
    """
    Feature: Personalized coaching & safety score. Builds on the existing
    driver_analytics table (no new tables needed) — inverts this week's
    avg_risk_score into a 0-100 "higher is better" score (same formula as
    the Reports page's fleet compliance %, for consistency), reuses the
    existing IMPROVING/STABLE/WORSENING trend, and adds a repeated-pattern
    coaching insight by finding this driver's most frequent alert_type
    across their history — e.g. a driver who mostly gets DROWSY alerts
    needs different coaching than one who mostly gets FATIGUE (yawning)
    or NO_FACE (distraction) alerts.
    """
    with get_connection() as conn:
        analytics_row = conn.execute("""
            SELECT * FROM driver_analytics WHERE driver_id = ?
            ORDER BY week_number DESC LIMIT 1
        """, (driver_id,)).fetchone()

        pattern_row = conn.execute("""
            SELECT alert_type, COUNT(*) as cnt FROM alerts
            WHERE driver_id = ?
            GROUP BY alert_type ORDER BY cnt DESC LIMIT 1
        """, (driver_id,)).fetchone()

    # Guard against BOTH the row being absent AND the row existing with a
    # NULL value in a specific column — the latter is a real possibility
    # here given how much accumulated test data can exist across a long
    # development history, potentially from an earlier code path that
    # didn't always populate every column. `if analytics_row else X` only
    # covers the first case; `or X` after covers the second.
    avg_risk_score = (analytics_row["avg_risk_score"] if analytics_row else 0) or 0
    fatigue_trend = (analytics_row["fatigue_trend"] if analytics_row else "STABLE") or "STABLE"
    safety_score = round(max(0, min(100, 100 - avg_risk_score)), 1)

    dominant_pattern = pattern_row["alert_type"] if pattern_row else None
    pattern_count = pattern_row["cnt"] if pattern_row else 0

    coaching_tips = {
        "DROWSY": "Most alerts are drowsiness-related. Consider more frequent rest stops, especially on long or night shifts.",
        "FATIGUE": "Frequent yawning detected. This often precedes drowsiness — a short break now can prevent a DANGER event later.",
        "NO_FACE": "Frequent distraction/face-off-camera events. Review phone and dashboard-glance habits during driving.",
        "DISTRACTION": "Frequent distraction signals detected. Consider minimizing in-cab distractions during active driving.",
    }
    coaching_tip = coaching_tips.get(
        dominant_pattern,
        "Not enough alert history yet for a specific coaching tip — keep driving sessions running to build a profile."
    )

    return {
        "driver_id": driver_id,
        "safety_score": safety_score,
        "fatigue_trend": fatigue_trend,
        "dominant_alert_pattern": dominant_pattern,
        "dominant_pattern_count": pattern_count,
        "coaching_tip": coaching_tip,
    }


def get_driver_analytics(driver_id, weeks=8):
    """
    Return the last N weeks of analytics rows for a driver (most recent first).
    Used to power the /api/driver/{id}/analytics endpoint.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM driver_analytics
            WHERE driver_id = ?
            ORDER BY week_number DESC
            LIMIT ?
        """, (driver_id, weeks)).fetchall()
        return rows_to_list(rows)


def get_daily_summary(driver_id, days=7):
    """
    Return per-day session summaries for the last N days — useful for
    daily (not just weekly) charts on the analytics dashboard.
    """
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                date(session_date) AS day,
                COUNT(*) AS total_sessions,
                SUM(total_alerts) AS total_alerts,
                AVG(peak_score) AS avg_risk_score,
                SUM(yawn_count) AS total_yawns
            FROM sessions
            WHERE driver_id = ? AND session_date >= ?
            GROUP BY date(session_date)
            ORDER BY day DESC
        """, (driver_id, cutoff)).fetchall()
        return rows_to_list(rows)


# ---------------------------------------------------------------------------
# FLEET-FACING QUERIES
# (support the React frontend's fleetApi.js / driverApi.js contract)
# ---------------------------------------------------------------------------

def update_driver_location(driver_id, lat, lng):
    """Persist a driver's latest known GPS position (for the fleet map)."""
    return update_driver(driver_id, last_lat=lat, last_lng=lng)


def update_driver_ux_thresholds(driver_id, **fields):
    """
    Update the four UX-facing 0-100 sensitivity sliders (drowsiness,
    distraction, yawn, blink) shown on the Driver Profile page. Distinct
    from update_driver_thresholds(), which sets the raw EAR/MAR floats
    the detection loop actually reads — see ux_thresholds.py for the
    conversion between the two.
    """
    allowed = {"drowsiness_threshold", "distraction_threshold",
               "yawn_threshold", "blink_threshold"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return False
    return update_driver(driver_id, **fields)


def _bucket_risk_level(score):
    """Shared SAFE/WARNING/DANGER bucketing from a 0-100 score, lowercased
    for the frontend (which lowercases anyway, but keep the API honest)."""
    if score is None:
        return "safe"
    if score >= 66:
        return "danger"
    if score >= 33:
        return "warning"
    return "safe"


def get_fleet_overview():
    """
    One row per driver enriched with their latest-session risk level and
    aggregate stats — the exact shape src/api/fleetApi.js:getFleetDrivers()
    expects: driver_id, driver_name, vehicle_type, vehicle_number,
    risk_level, peak_score, total_alerts, total_sessions, latitude, longitude.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                d.driver_id, d.driver_name, d.vehicle_type, d.vehicle_number,
                d.last_lat AS latitude, d.last_lng AS longitude,
                COALESCE(latest.peak_score, 0) AS peak_score,
                COALESCE(latest.final_level, 'SAFE') AS final_level,
                COALESCE(agg.total_sessions, 0) AS total_sessions,
                COALESCE(agg.total_alerts, 0) AS total_alerts
            FROM drivers d
            LEFT JOIN (
                SELECT s1.driver_id, s1.peak_score, s1.final_level
                FROM sessions s1
                INNER JOIN (
                    SELECT driver_id, MAX(session_date) AS max_date
                    FROM sessions GROUP BY driver_id
                ) s2 ON s1.driver_id = s2.driver_id AND s1.session_date = s2.max_date
            ) latest ON latest.driver_id = d.driver_id
            LEFT JOIN (
                SELECT driver_id, COUNT(*) AS total_sessions, SUM(total_alerts) AS total_alerts
                FROM sessions GROUP BY driver_id
            ) agg ON agg.driver_id = d.driver_id
            ORDER BY d.created_at DESC
        """).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        d["risk_level"] = d.pop("final_level").lower()
        result.append(d)
    return result


def get_fleet_map():
    """Lightweight lat/lng-only feed for the map — src/api/fleetApi.js:getFleetMap()."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT driver_id, last_lat AS latitude, last_lng AS longitude
            FROM drivers WHERE last_lat IS NOT NULL AND last_lng IS NOT NULL
        """).fetchall()
        return rows_to_list(rows)


def get_fleet_alerts(driver_id=None, risk_level=None, alert_type=None,
                      date_from=None, date_to=None, limit=200):
    """
    Fleet-wide alert feed with driver identity joined in — the shape
    src/api/fleetApi.js:getAlerts()/getLiveAlerts() expects: id, driver_id,
    driver_name, vehicle_number, alert_type, risk_level, score, timestamp,
    snapshot. Supports the same filters the Alerts page's UI exposes.
    """
    query = """
        SELECT a.alert_id AS id, a.driver_id, d.driver_name, d.vehicle_number,
               a.alert_type, a.risk_score AS score, a.alert_time AS timestamp,
               a.snapshot_path
        FROM alerts a
        JOIN drivers d ON d.driver_id = a.driver_id
        WHERE 1=1
    """
    params = []
    if driver_id:
        query += " AND a.driver_id = ?"
        params.append(driver_id)
    if alert_type and alert_type != "all":
        query += " AND a.alert_type = ?"
        params.append(alert_type)
    if date_from:
        query += " AND a.alert_time >= ?"
        params.append(date_from)
    if date_to:
        query += " AND a.alert_time <= ?"
        params.append(date_to)
    query += " ORDER BY a.alert_time DESC LIMIT ?"
    params.append(limit)

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        d["risk_level"] = _bucket_risk_level(d["score"])
        raw_path = d.pop("snapshot_path", None)
        d["snapshot"] = f"/snapshots/{raw_path}" if raw_path else None
        results.append(d)

    if risk_level and risk_level != "all":
        results = [r for r in results if r["risk_level"] == risk_level]

    return results


def get_fleet_report(period="weekly"):
    """
    Per-driver + fleet-wide aggregates for the Reports page —
    src/api/fleetApi.js:getFleetReport(). period is 'weekly' (last 7 days)
    or 'monthly' (last 30 days). compliance_pct = 100 - avg risk score,
    a simple, explainable proxy for "how compliant/safe was this driver".
    """
    days = 7 if period == "weekly" else 30
    cutoff = f"-{days} days"

    with get_connection() as conn:
        driver_rows = conn.execute("""
            SELECT
                d.driver_id, d.driver_name, d.vehicle_number,
                COUNT(s.session_id) AS sessions,
                COALESCE(SUM(s.total_alerts), 0) AS total_alerts,
                COALESCE(AVG(s.peak_score), 0) AS avg_risk_score
            FROM drivers d
            LEFT JOIN sessions s
                ON s.driver_id = d.driver_id
                AND s.session_date >= datetime('now', ?, 'localtime')
            GROUP BY d.driver_id
            ORDER BY d.driver_name
        """, (cutoff,)).fetchall()

        trend_rows = conn.execute("""
            SELECT date(session_date) AS day, AVG(peak_score) AS avg_score
            FROM sessions
            WHERE session_date >= datetime('now', ?, 'localtime')
            GROUP BY date(session_date)
            ORDER BY day ASC
        """, (cutoff,)).fetchall()

    drivers = []
    for r in driver_rows:
        d = dict(r)
        avg_score = d["avg_risk_score"] or 0
        d["avg_risk_score"] = round(avg_score, 1)
        d["compliance_pct"] = round(max(0, 100 - avg_score), 1)
        d["risk_level"] = _bucket_risk_level(avg_score)
        drivers.append(d)

    trend = [{
        "label": r["day"],
        "compliance_pct": round(max(0, 100 - (r["avg_score"] or 0)), 1),
    } for r in trend_rows]

    return {"period": period, "drivers": drivers, "trend": trend}


def get_driver_profile_view(driver_id):
    """
    Driver identity + UX-facing thresholds, shaped for
    src/api/driverApi.js:getDriverProfile(): driver_id, driver_name,
    vehicle_type, vehicle_number, photo_url, thresholds: {...}.
    """
    driver = get_driver(driver_id)
    if driver is None:
        return None
    return {
        "driver_id": driver["driver_id"],
        "driver_name": driver["driver_name"],
        "vehicle_type": driver["vehicle_type"],
        "vehicle_number": driver["vehicle_number"],
        "photo_url": driver.get("profile_photo_path") or "",
        "thresholds": {
            "drowsiness_threshold": driver.get("drowsiness_threshold", 50),
            "distraction_threshold": driver.get("distraction_threshold", 50),
            "yawn_threshold": driver.get("yawn_threshold", 50),
            "blink_threshold": driver.get("blink_threshold", 50),
        },
    }


def get_driver_stats_view(driver_id):
    """
    "Current session summary" shape for src/api/fleetApi.js:getDriverStats()
    — the frontend's DriverPage shows this alongside the live /stats poll.
    Uses the most recently saved session as the summary snapshot.
    """
    driver = get_driver(driver_id)
    if driver is None:
        return None
    sessions = get_sessions_for_driver(driver_id, limit=1)
    latest = sessions[0] if sessions else None
    return {
        "driver_id": driver["driver_id"],
        "driver_name": driver["driver_name"],
        "vehicle_type": driver["vehicle_type"],
        "vehicle_number": driver["vehicle_number"],
        "session_duration": round((latest["session_duration"] if latest else 0) / 60, 1),  # seconds -> minutes
        "total_alerts": latest["total_alerts"] if latest else 0,
        "yawn_count": latest["yawn_count"] if latest else 0,
        "blink_count": latest["blink_count"] if latest else 0,
        "peak_score": latest["peak_score"] if latest else 0,
        "risk_level": (latest["final_level"] if latest else "SAFE").lower(),
    }


def get_driver_history_view(driver_id):
    """
    Session history reshaped for src/api/fleetApi.js:getDriverHistory():
    same rows as get_sessions_for_driver(), but session_duration in
    minutes and final_level renamed+lowercased to risk_level.
    """
    sessions = get_sessions_for_driver(driver_id)
    result = []
    for s in sessions:
        s = dict(s)
        s["session_duration"] = round(s["session_duration"] / 60, 1)
        s["risk_level"] = s.pop("final_level").lower()
        s["session_date"] = (s["session_date"] or "")[:10]  # YYYY-MM-DD for chart labels
        result.append(s)
    return result


# ---------------------------------------------------------------------------
# SELF-TEST (run this file directly to sanity-check the module)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running database.py self-test...")

    # Use a throwaway test DB so we don't touch real data
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_sixthsense.db")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    init_db()

    # 1. Add a driver
    did = add_driver("Ramesh Kumar", 34, "truck", "KA-01-AB-1234")
    print("Added driver:", get_driver(did))

    # 2. Update thresholds
    update_driver_thresholds(did, ear_threshold=0.22, mar_threshold=0.55)
    print("After threshold update:", get_driver(did))

    # 3. Add a session
    sid = save_session(did, session_duration=1800, peak_score=72,
                        total_alerts=3, yawn_count=4, blink_count=210,
                        final_level="WARNING", snapshots_count=2)
    print("Saved session:", get_session(sid))

    # 4. Add alerts
    save_alert(sid, did, "DROWSY", 75, 12.9716, 77.5946)
    save_alert(sid, did, "WARNING", 60, 12.9716, 77.5946)
    print("Alerts for session:", get_alerts_for_session(sid))

    # 5. Analytics
    recompute_weekly_analytics(did)
    print("Analytics:", get_driver_analytics(did))

    # 6. Fleet list
    print("All drivers:", get_all_drivers())

    print("\nSelf-test complete. All core functions executed without errors.")
