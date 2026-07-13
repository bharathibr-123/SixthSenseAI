"""
api_routes.py
SixthSense AI — Fleet Driver Safety & Analytics Platform
Tata Technologies InnoVent Hackathon 2026

Purpose:
    All /api/ Flask routes for driver registration, session saving,
    and analytics. Built as a Blueprint so it drops into Bharathi's
    existing app.py without touching her existing routes:

        from api_routes import api_bp
        app.register_blueprint(api_bp)

    That's the only change needed in app.py.

Depends on:
    database.py        (must call database.init_db() once at startup)
    risk_predictor.py   (optional — only used by the bonus prediction route)

Routes implemented:
    DRIVER REGISTRATION
        POST /api/driver/register
        GET  /api/driver/<id>
        GET  /api/fleet/drivers
        PUT  /api/driver/<id>/thresholds

    SESSION SAVING
        POST /api/session/save
        GET  /api/driver/<id>/sessions
        GET  /api/driver/<id>/analytics

    BONUS (ties risk_predictor.py in — remove if not needed)
        GET  /api/driver/<id>/risk-prediction

    CALIBRATION (ties threshold_calibration.py in — remove if not needed)
        GET   /api/driver/<id>/calibration-status
        POST  /api/driver/<id>/calibration-reset

    FRONTEND CONTRACT (matches src/api/fleetApi.js + src/api/driverApi.js
    in the React app — added after reconciling against the real frontend
    source, since these names/shapes don't match the routes above 1:1)
        GET  /api/fleet/map
        GET  /api/fleet/reports
        GET  /api/alerts/live
        GET  /api/driver/<id>/stats
        GET  /api/driver/<id>/history
        GET  /api/driver/<id>/profile
        PUT  /api/driver/<id>/profile
    Note: /api/fleet/drivers above was ALSO changed to return the richer
    shape the frontend expects (risk_level, peak_score, lat/lng, etc.)
    instead of raw driver columns — see get_fleet_overview() in database.py.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import sqlite3

import database as db

try:
    import ux_thresholds as ux
    _UX_THRESHOLDS_AVAILABLE = True
except ImportError:
    _UX_THRESHOLDS_AVAILABLE = False

# Risk prediction is optional — import defensively so api_routes.py still
# works even before risk_predictor.py / scikit-learn are set up.
try:
    import risk_predictor as rp
    _RISK_PREDICTOR_AVAILABLE = True
except ImportError:
    _RISK_PREDICTOR_AVAILABLE = False

# Same defensive pattern for the personalized-threshold calibration module.
try:
    import threshold_calibration as calib
    _CALIBRATION_AVAILABLE = True
except ImportError:
    _CALIBRATION_AVAILABLE = False


api_bp = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def error_response(message, status_code=400):
    """Consistent error JSON shape across every route."""
    return jsonify({"success": False, "error": message}), status_code


def success_response(data=None, status_code=200):
    """Consistent success JSON shape across every route."""
    payload = {"success": True}
    if data is not None:
        payload["data"] = data
    return jsonify(payload), status_code


VALID_VEHICLE_TYPES = {"truck", "bus", "cab", "mining"}
VALID_ALERT_TYPES = {"DROWSY", "FATIGUE", "WARNING", "NO_FACE"}
VALID_LEVELS = {"SAFE", "WARNING", "DANGER"}


# ---------------------------------------------------------------------------
# DRIVER REGISTRATION API
# ---------------------------------------------------------------------------

@api_bp.route("/driver/register", methods=["POST"])
def register_driver():
    """
    Register a new driver.

    Body (JSON):
        {
            "driver_name": "Ramesh Kumar",     (required)
            "age": 34,                          (optional)
            "vehicle_type": "truck",            (required: truck/bus/cab/mining)
            "vehicle_number": "KA-01-AB-1234",  (required)
            "profile_photo_path": "...",        (optional)
            "ear_threshold": 0.25,              (optional, default 0.25)
            "mar_threshold": 0.6                (optional, default 0.6)
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return error_response("Request body must be valid JSON.")

    driver_name = (body.get("driver_name") or "").strip()
    vehicle_type = (body.get("vehicle_type") or "").strip().lower()
    vehicle_number = (body.get("vehicle_number") or "").strip()

    if not driver_name:
        return error_response("driver_name is required.")
    if vehicle_type not in VALID_VEHICLE_TYPES:
        return error_response(
            f"vehicle_type must be one of {sorted(VALID_VEHICLE_TYPES)}.")
    if not vehicle_number:
        return error_response("vehicle_number is required.")

    age = body.get("age")
    if age is not None:
        try:
            age = int(age)
        except (TypeError, ValueError):
            return error_response("age must be an integer.")

    try:
        driver_id = db.add_driver(
            driver_name=driver_name,
            age=age,
            vehicle_type=vehicle_type,
            vehicle_number=vehicle_number,
            profile_photo_path=body.get("profile_photo_path"),
            ear_threshold=body.get("ear_threshold", 0.25),
            mar_threshold=body.get("mar_threshold", 0.6),
        )
    except sqlite3.Error as e:
        return error_response(f"Database error: {e}", 500)

    driver = db.get_driver(driver_id)
    return success_response(driver, status_code=201)


@api_bp.route("/driver/<int:driver_id>", methods=["GET"])
def get_driver_profile(driver_id):
    """Fetch a single driver's profile."""
    driver = db.get_driver(driver_id)
    if driver is None:
        return error_response(f"Driver {driver_id} not found.", 404)
    return success_response(driver)


"""
IMPORTANT — response shape for everything below this line:

The routes above this point use success_response()/error_response(), which
wrap every payload as {"success": true/false, "data"/"error": ...}. That
convention was designed before I had access to the actual React frontend.

Having now read src/api/fleetApi.js and src/api/driverApi.js directly, the
real frontend calls `res.json()` and reads fields straight off the top
level (e.g. `driversRes.drivers`, `stats.driver_name`, `live.risk_score`)
— it never unwraps a `.data` envelope. So every route below (and the
fleet/drivers route above, which I've updated) returns RAW JSON matching
exactly what fleetApi.js / driverApi.js expect, with no success/data
wrapper. This is intentionally inconsistent with the earlier routes in
this file — reconciling this file's own convention is a good follow-up
but out of scope for making the actual UI work today.
"""


@api_bp.route("/fleet/drivers", methods=["GET"])
def list_fleet_drivers():
    """
    Fetch the full fleet driver list, enriched with each driver's latest
    risk level and aggregate stats (peak_score, total_alerts,
    total_sessions, latitude/longitude).

    DUAL-SHAPE RESPONSE — this endpoint has two real consumers with
    contradictory conventions:
      1. The React app's src/api/fleetApi.js reads fields straight off
         the top level: `driversRes.drivers`.
      2. Bharathi's standalone dashboard HTML (sixthsense_ai_fleet_
         dashboard.html) expects the {success, data} envelope and reads
         `body.data.drivers`.
    Rather than break one of them, this response includes both the
    envelope AND the raw top-level keys, so either consumer's code works
    unmodified. A bit redundant, but the fastest safe fix under time
    pressure — worth collapsing to one convention post-hackathon.
    """
    try:
        drivers = db.get_fleet_overview()
    except sqlite3.Error as e:
        return jsonify({"success": False, "error": f"Database error: {e}"}), 500

    payload = {"count": len(drivers), "drivers": drivers}
    return jsonify({"success": True, "data": payload, **payload})


@api_bp.route("/fleet/map", methods=["GET"])
def fleet_map():
    """Lightweight lat/lng feed for the map — src/api/fleetApi.js:getFleetMap()."""
    try:
        vehicles = db.get_fleet_map()
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    return jsonify({"count": len(vehicles), "vehicles": vehicles})


@api_bp.route("/fleet/reports", methods=["GET"])
def fleet_reports():
    """
    Weekly/monthly compliance report — src/api/fleetApi.js:getFleetReport().
    Query param: ?period=weekly|monthly (default weekly). Returns
    {period, drivers, trend} directly — ReportsPage.jsx reads report.drivers
    and report.trend straight off the response, no envelope.
    """
    period = request.args.get("period", default="weekly")
    if period not in ("weekly", "monthly"):
        return jsonify({"error": "period must be 'weekly' or 'monthly'."}), 400
    try:
        report = db.get_fleet_report(period)
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    return jsonify(report)


@api_bp.route("/alerts/live", methods=["GET"])
def alerts_live():
    """
    Fleet-wide alert feed — src/api/fleetApi.js:getLiveAlerts()/getAlerts().
    Optional query params match the Alerts page's filters exactly:
    driver_id, risk_level, alert_type, from, to.
    """
    driver_id = request.args.get("driver_id", type=int)
    risk_level = request.args.get("risk_level")
    alert_type = request.args.get("alert_type")
    date_from = request.args.get("from")
    date_to = request.args.get("to")

    try:
        alerts = db.get_fleet_alerts(
            driver_id=driver_id, risk_level=risk_level, alert_type=alert_type,
            date_from=date_from, date_to=date_to)
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

    return jsonify({"count": len(alerts), "alerts": alerts})


@api_bp.route("/driver/<int:driver_id>/stats", methods=["GET"])
def driver_stats_frontend(driver_id):
    """
    "Current session summary" — src/api/fleetApi.js:getDriverStats().
    Distinct from /api/driver/<id>/analytics above (weekly/daily rollups);
    this is the latest-session snapshot DriverPage.jsx shows next to the
    live /stats poll. Returns the raw object directly, no envelope.
    """
    try:
        stats = db.get_driver_stats_view(driver_id)
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    if stats is None:
        return jsonify({"error": f"Driver {driver_id} not found."}), 404
    return jsonify(stats)


@api_bp.route("/driver/<int:driver_id>/history", methods=["GET"])
def driver_history_frontend(driver_id):
    """
    Session history reshaped for src/api/fleetApi.js:getDriverHistory() —
    session_duration in minutes, risk_level (lowercase) instead of
    final_level. Returns a raw array directly (DriverPage.jsx accepts
    either an array or {sessions: [...]}; array is simplest here).
    """
    driver = db.get_driver(driver_id)
    if driver is None:
        return jsonify({"error": f"Driver {driver_id} not found."}), 404
    try:
        history = db.get_driver_history_view(driver_id)
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    return jsonify(history)


@api_bp.route("/driver/<int:driver_id>/snapshots", methods=["GET"])
def driver_snapshots(driver_id):
    """
    All of this driver's DANGER-event snapshots, most recent first — feeds
    the Snapshots Gallery on DriverPage.jsx. Filters db.get_alerts_for_driver()
    down to alerts that actually have a saved snapshot_path (most WARNING
    alerts don't), and turns the bare filename into a servable URL via
    app.py's GET /snapshots/<filename> route.
    """
    driver = db.get_driver(driver_id)
    if driver is None:
        return jsonify({"error": f"Driver {driver_id} not found."}), 404
    try:
        alerts = db.get_alerts_for_driver(driver_id)
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

    snapshots = [
        {
            "alert_id": a.get("alert_id"),
            "alert_type": a.get("alert_type"),
            "alert_time": a.get("alert_time"),
            "risk_score": a.get("risk_score"),
            "snapshot_url": f"/snapshots/{a['snapshot_path']}",
            "clip_url": f"/clips/{a['video_clip_path']}" if a.get("video_clip_path") else None,
        }
        for a in alerts
        if a.get("snapshot_path")
    ]
    return jsonify({"count": len(snapshots), "snapshots": snapshots})


@api_bp.route("/driver/<int:driver_id>/profile", methods=["GET"])
def driver_profile_get(driver_id):
    """
    Driver identity + UX-facing thresholds —
    src/api/driverApi.js:getDriverProfile(). Returns the raw profile
    object directly: driver_id, driver_name, vehicle_type, vehicle_number,
    photo_url, thresholds: {drowsiness/distraction/yawn/blink_threshold}.
    """
    try:
        profile = db.get_driver_profile_view(driver_id)
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    if profile is None:
        return jsonify({"error": f"Driver {driver_id} not found."}), 404
    return jsonify(profile)


@api_bp.route("/driver/<int:driver_id>/profile", methods=["PUT"])
def driver_profile_put(driver_id):
    """
    Save edited profile + thresholds —
    src/api/driverApi.js:updateDriverProfile(). Body:
        {driver_name, vehicle_type, vehicle_number, thresholds: {...}}
    Updates both the UX-facing 0-100 sliders AND (if ux_thresholds.py is
    available) the underlying EAR/MAR floats the detection loop actually
    reads, so a saved slider change has a real effect on app.py, not just
    a cosmetic one. Returns the updated profile directly, no envelope —
    DriverProfilePage.jsx does setProfile(updated) with the raw response.
    """
    driver = db.get_driver(driver_id)
    if driver is None:
        return jsonify({"error": f"Driver {driver_id} not found."}), 404

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    identity_fields = {}
    for key in ("driver_name", "vehicle_type", "vehicle_number", "emergency_contacts"):
        if key in body and body[key]:
            identity_fields[key] = body[key]
    if identity_fields:
        db.update_driver(driver_id, **identity_fields)

    thresholds = body.get("thresholds", {})
    if thresholds:
        db.update_driver_ux_thresholds(driver_id, **thresholds)

        # Also recompute the real CV thresholds (ear/mar) so this save
        # actually changes detection behavior, not just the display value.
        if _UX_THRESHOLDS_AVAILABLE:
            updated_driver = db.get_driver(driver_id)
            cv_params = ux.ux_thresholds_to_cv_params(
                drowsiness_pct=updated_driver.get("drowsiness_threshold", 50),
                distraction_pct=updated_driver.get("distraction_threshold", 50),
                yawn_pct=updated_driver.get("yawn_threshold", 50),
                blink_pct=updated_driver.get("blink_threshold", 50),
            )
            db.update_driver_thresholds(
                driver_id,
                ear_threshold=cv_params["ear_threshold"],
                mar_threshold=cv_params["mar_threshold"],
            )

    return jsonify(db.get_driver_profile_view(driver_id))


@api_bp.route("/driver/<int:driver_id>/thresholds", methods=["PUT"])
def update_thresholds(driver_id):
    """
    Update a driver's personalized EAR/MAR thresholds.

    Body (JSON):
        {
            "ear_threshold": 0.22,   (optional — at least one required)
            "mar_threshold": 0.58    (optional — at least one required)
        }
    """
    driver = db.get_driver(driver_id)
    if driver is None:
        return error_response(f"Driver {driver_id} not found.", 404)

    body = request.get_json(silent=True)
    if not body:
        return error_response("Request body must be valid JSON.")

    ear_threshold = body.get("ear_threshold")
    mar_threshold = body.get("mar_threshold")

    if ear_threshold is None and mar_threshold is None:
        return error_response(
            "Provide at least one of ear_threshold or mar_threshold.")

    for name, value in (("ear_threshold", ear_threshold), ("mar_threshold", mar_threshold)):
        if value is not None:
            try:
                float(value)
            except (TypeError, ValueError):
                return error_response(f"{name} must be a number.")

    updated = db.update_driver_thresholds(
        driver_id, ear_threshold=ear_threshold, mar_threshold=mar_threshold)

    if not updated:
        return error_response("No changes were applied.", 500)

    return success_response(db.get_driver(driver_id))


# ---------------------------------------------------------------------------
# SESSION SAVING API
# ---------------------------------------------------------------------------

@api_bp.route("/session/save", methods=["POST"])
def save_session_route():
    """
    Save a completed monitoring session and its alerts in one call.

    Body (JSON):
        {
            "driver_id": 1,                (required)
            "session_duration": 1800,      (seconds, optional, default 0)
            "peak_score": 72,              (0-100, optional, default 0)
            "yawn_count": 4,                (optional, default 0)
            "blink_count": 210,             (optional, default 0)
            "final_level": "WARNING",       (optional: SAFE/WARNING/DANGER)
            "snapshots_count": 2,           (optional, default 0)
            "alerts": [                     (optional list of alert events)
                {
                    "alert_type": "DROWSY",
                    "risk_score": 75,
                    "location_lat": 12.9716,
                    "location_lng": 77.5946,
                    "alert_time": "2026-07-02 10:15:00"   (optional)
                }
            ]
        }

    total_alerts is derived automatically from len(alerts) unless the alerts
    list is omitted, in which case you may pass "total_alerts" directly.
    """
    body = request.get_json(silent=True)
    if not body:
        return error_response("Request body must be valid JSON.")

    driver_id = body.get("driver_id")
    if driver_id is None:
        return error_response("driver_id is required.")

    driver = db.get_driver(driver_id)
    if driver is None:
        return error_response(f"Driver {driver_id} not found.", 404)

    final_level = body.get("final_level", "SAFE")
    if final_level not in VALID_LEVELS:
        return error_response(f"final_level must be one of {sorted(VALID_LEVELS)}.")

    alerts_list = body.get("alerts", [])
    if not isinstance(alerts_list, list):
        return error_response("alerts must be a list.")

    total_alerts = body.get("total_alerts", len(alerts_list))

    try:
        session_id = db.save_session(
            driver_id=driver_id,
            session_duration=body.get("session_duration", 0),
            peak_score=body.get("peak_score", 0),
            total_alerts=total_alerts,
            yawn_count=body.get("yawn_count", 0),
            blink_count=body.get("blink_count", 0),
            final_level=final_level,
            snapshots_count=body.get("snapshots_count", 0),
        )

        saved_alerts = []
        for alert in alerts_list:
            alert_type = alert.get("alert_type")
            if alert_type not in VALID_ALERT_TYPES:
                return error_response(
                    f"Invalid alert_type '{alert_type}'. Must be one of "
                    f"{sorted(VALID_ALERT_TYPES)}.")
            alert_id = db.save_alert(
                session_id=session_id,
                driver_id=driver_id,
                alert_type=alert_type,
                risk_score=alert.get("risk_score"),
                location_lat=alert.get("location_lat"),
                location_lng=alert.get("location_lng"),
                alert_time=alert.get("alert_time"),
            )
            saved_alerts.append(alert_id)

        # Keep the weekly analytics rollup current
        db.recompute_weekly_analytics(driver_id)

    except sqlite3.Error as e:
        return error_response(f"Database error: {e}", 500)

    session = db.get_session(session_id)
    session["alert_ids"] = saved_alerts
    return success_response(session, status_code=201)


@api_bp.route("/driver/<int:driver_id>/sessions", methods=["GET"])
def get_driver_sessions(driver_id):
    """
    Fetch all sessions for a driver, most recent first.
    Optional query param: ?limit=20
    """
    driver = db.get_driver(driver_id)
    if driver is None:
        return error_response(f"Driver {driver_id} not found.", 404)

    limit = request.args.get("limit", type=int)

    try:
        sessions = db.get_sessions_for_driver(driver_id, limit=limit)
    except sqlite3.Error as e:
        return error_response(f"Database error: {e}", 500)

    return success_response({"count": len(sessions), "sessions": sessions})


@api_bp.route("/driver/<int:driver_id>/analytics", methods=["GET"])
def get_driver_analytics_route(driver_id):
    """
    Fetch weekly + daily analytics for a driver's dashboard.
    Optional query params:
        ?weeks=8   (number of weekly rows, default 8)
        ?days=7    (number of days for the daily summary, default 7)
    """
    driver = db.get_driver(driver_id)
    if driver is None:
        return error_response(f"Driver {driver_id} not found.", 404)

    weeks = request.args.get("weeks", default=8, type=int)
    days = request.args.get("days", default=7, type=int)

    try:
        weekly = db.get_driver_analytics(driver_id, weeks=weeks)
        daily = db.get_daily_summary(driver_id, days=days)
    except sqlite3.Error as e:
        return error_response(f"Database error: {e}", 500)

    return success_response({"weekly": weekly, "daily": daily})


@api_bp.route("/driver/<int:driver_id>/safety-score", methods=["GET"])
def get_driver_safety_score_route(driver_id):
    """
    Feature: Personalized coaching & safety score — a single 0-100 score
    plus a repeated-pattern coaching tip, for a compact "how am I doing
    overall" card on the driver's page (distinct from the weekly/daily
    analytics arrays above, which are for charting trends over time).

    Returns the raw object directly, no {"success", "data"} envelope —
    matching driver_stats_frontend() and driver_history_frontend() above,
    which src/api/fleetApi.js:getSafetyScore() reads the same way
    (getDriverStats/getDriverHistory's sibling, not api_routes.py's
    success_response() convention used elsewhere in this file).
    """
    driver = db.get_driver(driver_id)
    if driver is None:
        return jsonify({"error": f"Driver {driver_id} not found."}), 404

    try:
        result = db.get_safety_score(driver_id)
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

    return jsonify(result)


# ---------------------------------------------------------------------------
# BONUS: ACCIDENT RISK PREDICTION
# ---------------------------------------------------------------------------

@api_bp.route("/driver/<int:driver_id>/risk-prediction", methods=["GET"])
def get_risk_prediction(driver_id):
    """
    Predict this driver's accident-risk probability for the next 30 minutes,
    based on their most recent session. Requires risk_predictor.py.

    Optional query param: ?hour=14 to override the hour used (defaults to
    the driver's most recent session hour if available, else current hour).
    """
    if not _RISK_PREDICTOR_AVAILABLE:
        return error_response(
            "risk_predictor module is not available on this server.", 503)

    driver = db.get_driver(driver_id)
    if driver is None:
        return error_response(f"Driver {driver_id} not found.", 404)

    sessions = db.get_sessions_for_driver(driver_id, limit=1)
    if not sessions:
        return error_response(
            f"Driver {driver_id} has no sessions yet — nothing to predict from.", 404)

    latest_session = sessions[0]

    hour_override = request.args.get("hour", type=int)
    if hour_override is None:
        try:
            hour_override = datetime.fromisoformat(
                latest_session["session_date"].split(".")[0]).hour
        except (ValueError, KeyError):
            hour_override = datetime.now().hour

    try:
        probability = rp.predict_accident_risk_from_session(
            latest_session, hour_of_day=hour_override)
    except Exception as e:
        return error_response(f"Prediction failed: {e}", 500)

    return success_response({
        "driver_id": driver_id,
        "based_on_session_id": latest_session["session_id"],
        "hour_used": hour_override,
        "accident_risk_probability": probability,
        "risk_level": rp.risk_level_label(probability),
    })


# ---------------------------------------------------------------------------
# BONUS: PERSONALIZED THRESHOLD CALIBRATION
# ---------------------------------------------------------------------------

@api_bp.route("/driver/<int:driver_id>/calibration-status", methods=["GET"])
def get_calibration_status_route(driver_id):
    """
    Check a driver's threshold calibration progress — e.g. to show a
    "Calibrating your profile... 3/5 sessions" indicator in the UI.
    Requires threshold_calibration.py.
    """
    if not _CALIBRATION_AVAILABLE:
        return error_response(
            "threshold_calibration module is not available on this server.", 503)

    driver = db.get_driver(driver_id)
    if driver is None:
        return error_response(f"Driver {driver_id} not found.", 404)

    try:
        status = calib.get_calibration_status(driver_id)
    except sqlite3.Error as e:
        return error_response(f"Database error: {e}", 500)

    return success_response(status)


@api_bp.route("/driver/<int:driver_id>/calibration-reset", methods=["POST"])
def reset_calibration_route(driver_id):
    """
    Wipe a driver's calibration history and revert their EAR/MAR thresholds
    to system defaults. Use when a driver's camera setup changes drastically
    (new vehicle, different mounting angle) so old samples don't mislead
    the new calibration. Requires threshold_calibration.py.
    """
    if not _CALIBRATION_AVAILABLE:
        return error_response(
            "threshold_calibration module is not available on this server.", 503)

    driver = db.get_driver(driver_id)
    if driver is None:
        return error_response(f"Driver {driver_id} not found.", 404)

    try:
        calib.reset_calibration(driver_id)
    except sqlite3.Error as e:
        return error_response(f"Database error: {e}", 500)

    return success_response(calib.get_calibration_status(driver_id))


# ---------------------------------------------------------------------------
# SELF-TEST (spins up a temporary Flask app and hits every route)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from flask import Flask

    print("Running api_routes.py self-test...\n")

    # Use a throwaway test DB so we don't touch real data
    test_db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_api_sixthsense.db")
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    db.DB_PATH = test_db_path
    db.init_db()

    if _RISK_PREDICTOR_AVAILABLE:
        rp.MODEL_PATH = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "test_rf.joblib")
        rp.load_or_train_model()

    if _CALIBRATION_AVAILABLE:
        calib.init_calibration_table()

    app = Flask(__name__)
    app.register_blueprint(api_bp)
    client = app.test_client()

    def show(label, resp):
        print(f"{label}: [{resp.status_code}] {resp.get_json()}")

    # 1. Register driver
    resp = client.post("/api/driver/register", json={
        "driver_name": "Suresh Patil", "age": 41,
        "vehicle_type": "bus", "vehicle_number": "MH-12-CD-5678"
    })
    show("POST /api/driver/register", resp)
    driver_id = resp.get_json()["data"]["driver_id"]

    # 2. Get driver profile
    show("GET /api/driver/<id>", client.get(f"/api/driver/{driver_id}"))

    # 3. List fleet
    show("GET /api/fleet/drivers", client.get("/api/fleet/drivers"))

    # 4. Update thresholds
    show("PUT /api/driver/<id>/thresholds", client.put(
        f"/api/driver/{driver_id}/thresholds",
        json={"ear_threshold": 0.21, "mar_threshold": 0.57}))

    # 5. Save a session with alerts
    resp = client.post("/api/session/save", json={
        "driver_id": driver_id, "session_duration": 3600, "peak_score": 80,
        "yawn_count": 6, "blink_count": 300, "final_level": "DANGER",
        "snapshots_count": 3,
        "alerts": [
            {"alert_type": "DROWSY", "risk_score": 82,
             "location_lat": 18.5204, "location_lng": 73.8567},
            {"alert_type": "FATIGUE", "risk_score": 70},
        ]
    })
    show("POST /api/session/save", resp)

    # 6. Get sessions
    show("GET /api/driver/<id>/sessions", client.get(f"/api/driver/{driver_id}/sessions"))

    # 7. Get analytics
    show("GET /api/driver/<id>/analytics", client.get(f"/api/driver/{driver_id}/analytics"))

    # 8. Bonus: risk prediction
    if _RISK_PREDICTOR_AVAILABLE:
        show("GET /api/driver/<id>/risk-prediction",
             client.get(f"/api/driver/{driver_id}/risk-prediction"))

    # 9. Calibration status + reset
    if _CALIBRATION_AVAILABLE:
        for i in range(5):
            calib.record_session_baseline(driver_id, avg_ear=0.27, avg_mar=0.34)
        show("GET /api/driver/<id>/calibration-status",
             client.get(f"/api/driver/{driver_id}/calibration-status"))
        show("POST /api/driver/<id>/calibration-reset",
             client.post(f"/api/driver/{driver_id}/calibration-reset"))

    # 10. Error cases
    show("GET /api/driver/9999 (not found)", client.get("/api/driver/9999"))
    show("POST /api/driver/register (missing fields)",
         client.post("/api/driver/register", json={"driver_name": ""}))

    # Cleanup
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    test_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "test_rf.joblib")
    if os.path.exists(test_model_path):
        os.remove(test_model_path)

    print("\nSelf-test complete. All routes exercised without errors.")
