"""
app.py
SixthSense AI — Fleet Driver Safety & Analytics Platform
Tata Technologies InnoVent Hackathon 2026

This is the main Flask application: the live camera/detection loop plus
every route the React frontend (App.jsx) and fleet dashboard talk to.
It wires together all five supporting modules built alongside it:

    database.py               — persistent storage (SQLite)
    offline_voice.py          — pre-cached multilingual voice alerts
    risk_predictor.py         — Random Forest accident-risk prediction
    threshold_calibration.py  — personalized EAR/MAR learning
    api_routes.py             — the /api/ REST Blueprint (driver/session/analytics)

Feature coverage (from the original project brief):
    - Real-time face detection (MediaPipe Face Mesh)
    - EAR (drowsiness) + MAR (yawn) detection
    - Head pose estimation (pitch/yaw/roll)
    - Risk Score Engine (0-100) -> SAFE / WARNING / DANGER
    - Live camera streaming (MJPEG over Flask)
    - Multilingual voice alerts (offline_voice.py)
    - Emergency WhatsApp alert with GPS (best-effort; needs credentials)
    - Weather monitoring (Open-Meteo, no API key required)
    - Stress detection via head-movement variance
    - Session history (now persisted server-side via database.py,
      superseding the earlier localStorage-only approach)
    - Snapshot capture on DANGER events
    - Find Nearby (rest stop / parking / hospital / police) via
      OpenStreetMap Overpass API (no API key required)

IMPORTANT — read before running:
    I (the ML/backend half of this project) do not have access to a
    physical webcam or Bharathi's original detection-loop code in this
    environment, so this file is a clean-room reference implementation
    of every feature from the project brief, not a merge of her existing
    code. Variable names, thresholds, and route names are my own
    reasonable choices — please diff this against anything Bharathi
    already has running and reconcile before relying on both.

    Everything NOT dependent on a physical camera (risk scoring math,
    all /api/ routes, weather, nearby-places, database wiring) has been
    self-tested in this environment. The camera capture loop itself
    (cv2.VideoCapture(0)) cannot be exercised here and needs testing on
    real hardware.
"""

import os
import time
import math
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS  # pip install flask-cors — needed since the
                              # React frontend runs on a different port

import database as db
import offline_voice as voice
import risk_predictor as rp
import threshold_calibration as calib
import ux_thresholds as ux
from api_routes import api_bp

# ===========================================================================
# CONFIG
# ===========================================================================

CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

# Default thresholds (overridden per-driver once calibrated — see
# threshold_calibration.py)
DEFAULT_EAR_THRESHOLD = 0.25
DEFAULT_MAR_THRESHOLD = 0.6

# Consecutive-frame counts before we call something a genuine event, not
# a blink/noise. At ~15-20fps this is roughly 1-1.5 seconds of eye closure.
EAR_CONSEC_FRAMES = 20
MAR_CONSEC_FRAMES = 15
NO_FACE_CONSEC_FRAMES = 30

# Risk score decay/rise rates per frame (keeps the 0-100 score smooth
# instead of jumping between extremes every frame)
RISK_RISE_STEP = 4
RISK_FALL_STEP = 2

# Head-pose stress detection: rolling window of yaw/pitch samples; high
# variance = restless/stressed head movement rather than genuine steering
STRESS_WINDOW_SECONDS = 20

SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
os.makedirs(CLIPS_DIR, exist_ok=True)

# Feature: incident video clips. Assumes ~15fps camera loop — not exact,
# but close enough that a "few seconds before/after" clip is genuinely
# useful context around a DANGER event, not precisely 3.000 seconds.
PRE_INCIDENT_FRAMES = 45
POST_INCIDENT_FRAMES = 45

# ===========================================================================
# FLASK APP SETUP
# ===========================================================================

app = Flask(__name__)
CORS(app)  # allow the React dev server (different origin) to call these APIs

# One-time module initialization
db.init_db()
calib.init_calibration_table()
voice.prepare_all_voice_alerts()   # needs internet the first time only
rp.load_or_train_model()           # trains once, caches to disk after
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

app.register_blueprint(api_bp)

# ===========================================================================
# MEDIAPIPE FACE MESH SETUP
# ===========================================================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark indices (MediaPipe Face Mesh topology)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 39, 181, 0, 17]  # left, right, top-left, bottom-left, top-mid, bottom-mid

# 3D model points for head-pose solvePnP (generic face model, in mm)
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),          # Nose tip        (landmark 1)
    (0.0, -330.0, -65.0),     # Chin            (landmark 152)
    (-225.0, 170.0, -135.0),  # Left eye corner (landmark 33)
    (225.0, 170.0, -135.0),   # Right eye corner(landmark 263)
    (-150.0, -150.0, -125.0), # Left mouth      (landmark 61)
    (150.0, -150.0, -125.0),  # Right mouth     (landmark 291)
], dtype=np.float64)
HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]


def _euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_ear(landmarks, indices, w, h):
    """
    Eye Aspect Ratio: ratio of vertical eye opening to horizontal eye width.
    Drops sharply when the eye closes.
    """
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    horizontal = 2.0 * _euclidean(p1, p4)
    if horizontal == 0:
        return 0.3
    return vertical / horizontal


def compute_mar(landmarks, indices, w, h):
    """
    Mouth Aspect Ratio: ratio of vertical mouth opening to horizontal
    mouth width. Rises sharply during a yawn.
    """
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    left, right, top_l, bottom_l, top_m, bottom_m = pts
    vertical = _euclidean(top_l, bottom_l) + _euclidean(top_m, bottom_m)
    horizontal = 2.0 * _euclidean(left, right)
    if horizontal == 0:
        return 0.3
    return vertical / horizontal


def compute_head_pose(landmarks, w, h):
    """
    Estimate head pitch/yaw/roll (degrees) via solvePnP against a generic
    3D face model. Returns (pitch, yaw, roll).
    """
    image_points = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in HEAD_POSE_LANDMARKS],
        dtype=np.float64,
    )
    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(-rotation_mat[2, 0], sy))
        yaw = math.degrees(math.atan2(rotation_mat[1, 0], rotation_mat[0, 0]))
        roll = math.degrees(math.atan2(rotation_mat[2, 1], rotation_mat[2, 2]))
    else:
        pitch = math.degrees(math.atan2(-rotation_mat[2, 0], sy))
        yaw = 0.0
        roll = 0.0
    return pitch, yaw, roll


# ===========================================================================
# SESSION STATE (single active driver at a time — one vehicle, one camera)
# ===========================================================================

class MonitoringSession:
    """
    Holds all live state for the current monitoring session: counters,
    risk score, calibration accumulators, and alert log. Reset every time
    a new session starts.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.driver_id = None
        self.driver = None
        self.ear_threshold = DEFAULT_EAR_THRESHOLD
        self.mar_threshold = DEFAULT_MAR_THRESHOLD

        self.start_time = None
        self.risk_score = 0
        self.risk_level = "SAFE"

        self.eye_closed_frames = 0
        self.mouth_open_frames = 0
        self.no_face_frames = 0

        self.blink_count = 0
        self.yawn_count = 0
        self.stress_events_count = 0
        self.stress_currently_active = False

        # Feature: incident video clips. frame_buffer is a small rolling
        # window of recent frames (memory-only, never written to disk
        # unless an incident actually triggers) — this is what supplies
        # the "before" context once a DANGER event fires. incident_frames
        # accumulates pre+post frames for the clip currently being built;
        # incident_frames_remaining counts down the post-incident capture
        # window before the clip gets finalized and written to disk.
        self.frame_buffer = deque(maxlen=PRE_INCIDENT_FRAMES)
        self.incident_recording_active = False
        self.incident_frames = []
        self.incident_frames_remaining = 0
        self.incident_alert_index = None
        self.total_alerts = 0
        self.snapshots_count = 0
        self.peak_score = 0

        self.alerts = []  # list of dicts: alert_type, risk_score, lat, lng, time

        # Running EAR/MAR averages, used for both display and end-of-session
        # calibration baseline (threshold_calibration.py)
        self.ear_sum = 0.0
        self.mar_sum = 0.0
        self.ear_mar_samples = 0

        # Head-pose stress detection: rolling (timestamp, yaw, pitch) window
        self.pose_window = deque()

        self.last_gps = (None, None)  # (lat, lng) — set by frontend via /api/monitoring/gps
        self.last_gps_time = None
        self.speed_kmh = 0.0

        # Per-session overrides derived from the driver's saved UX sliders
        # (Driver Profile page -> /api/driver/<id>/profile) via
        # ux_thresholds.py. Fall back to the module-level defaults if a
        # driver has never touched their sliders.
        self.ear_consec_frames = EAR_CONSEC_FRAMES
        self.distraction_yaw_cutoff = 12.0
        self.distraction_pitch_cutoff = 10.0

        # Demo Mode — see run_demo_sequence(). While active, process_frame()
        # skips real MediaPipe detection and lets the demo thread drive
        # risk_score/risk_level directly, so a presenter can show the full
        # SAFE -> WARNING -> DANGER -> alert cycle on command without
        # needing to actually act drowsy on camera.
        self.demo_active = False
        self.demo_stop_requested = False

        # Voice command ("I'm okay") — see /api/monitoring/acknowledge.
        # Deliberately does NOT touch risk_score/risk_level (those stay
        # 100% driven by real camera detection, honestly) — it only pauses
        # new alert log entries + repeat voice nags for a short window, so
        # a driver who's just spoken up isn't immediately re-alarmed for
        # the same episode while genuinely still adjusting.
        self.alert_grace_until = 0

        # Tracks the highest risk_level an alert has already been logged
        # for during the CURRENT episode, so alerts fire on genuine
        # severity transitions (SAFE -> WARNING -> DANGER) instead of on
        # an early frame-counter crossing that doesn't yet reflect the
        # real risk_score. Resets to "SAFE" once risk_level drops back to
        # SAFE, so a later episode can alert again from WARNING.
        self.last_alert_level = "SAFE"

    def reset(self, driver_id, driver):
        with self.lock:
            self.__init__()
            self.active = True
            self.driver_id = driver_id
            self.driver = driver
            self.ear_threshold = driver.get("ear_threshold") or DEFAULT_EAR_THRESHOLD
            self.mar_threshold = driver.get("mar_threshold") or DEFAULT_MAR_THRESHOLD
            self.start_time = time.time()

            # Apply this driver's Distraction / Blink Duration sliders (set
            # via the Driver Profile page) to this session's detection
            # sensitivity. Drowsiness/Yawn sliders already flow through via
            # ear_threshold/mar_threshold above, kept in sync whenever the
            # profile is saved — see api_routes.py:driver_profile_put().
            cv_params = ux.ux_thresholds_to_cv_params(
                drowsiness_pct=driver.get("drowsiness_threshold", 50),
                distraction_pct=driver.get("distraction_threshold", 50),
                yawn_pct=driver.get("yawn_threshold", 50),
                blink_pct=driver.get("blink_threshold", 50),
            )
            self.ear_consec_frames = cv_params["ear_consec_frames"]
            self.distraction_yaw_cutoff = cv_params["distraction_yaw_std_cutoff"]
            self.distraction_pitch_cutoff = cv_params["distraction_pitch_std_cutoff"]

    def stop(self):
        with self.lock:
            self.active = False


session = MonitoringSession()


# ===========================================================================
# RISK SCORE ENGINE
# ===========================================================================

_LEVEL_SEVERITY = {"SAFE": 0, "WARNING": 1, "DANGER": 2}


def classify_risk_level(score):
    """Same SAFE/WARNING/DANGER cutoffs used live in update_risk_score(),
    exposed standalone so a session's peak_score can be classified after
    the fact (see stop_monitoring()), without needing to re-derive the
    thresholds in two places."""
    if score >= 66:
        return "DANGER"
    elif score >= 33:
        return "WARNING"
    return "SAFE"


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance between two lat/lng points, in kilometers."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def get_driving_context():
    """
    Feature: Contextual alert sensitivity. Classifies current driving
    conditions from data already being collected (GPS-derived speed +
    system clock), and returns a multiplier applied to the eyes/mouth
    consecutive-frame thresholds in process_frame() — lower multiplier =
    more sensitive (fewer frames needed to trigger), higher = less
    sensitive (more frames needed, tolerating brief head turns).

    Night driving is a well-documented higher-fatigue-risk window, so
    sensitivity increases (multiplier < 1). Low-speed city driving
    involves frequent legitimate head movement (mirror checks,
    intersections) that would otherwise cause false positives, so
    sensitivity decreases slightly (multiplier > 1) there. Highway/day is
    the baseline (multiplier == 1).
    """
    hour = datetime.now().hour
    is_night = hour < 6 or hour >= 20

    if session.speed_kmh >= 50:
        road = "highway"
    elif session.speed_kmh > 0:
        road = "city"
    else:
        road = "unknown"  # no recent GPS fix, or vehicle stationary

    multiplier = 1.0
    if is_night:
        multiplier *= 0.8  # more sensitive at night
    if road == "city":
        multiplier *= 1.2  # less sensitive in low-speed stop-and-go traffic
    elif road == "highway":
        multiplier *= 0.9  # monotonous highway driving is also a known fatigue risk

    return {
        "road": road,
        "time_of_day": "night" if is_night else "day",
        "speed_kmh": round(session.speed_kmh, 1),
        "sensitivity_multiplier": round(multiplier, 2),
    }


def update_risk_score(eyes_closed, yawning, no_face, stress_detected):
    """
    Smoothly moves the 0-100 risk score toward a target based on this
    frame's signals, rather than snapping — makes the on-screen number
    and voice alerts feel stable instead of flickering.
    """
    target = 0
    if no_face:
        target = 70
    elif eyes_closed and yawning:
        target = 95
    elif eyes_closed:
        target = 80
    elif yawning:
        target = 55
    elif stress_detected:
        target = 40

    if target > session.risk_score:
        session.risk_score = min(target, session.risk_score + RISK_RISE_STEP)
    else:
        session.risk_score = max(target, session.risk_score - RISK_FALL_STEP)

    session.risk_score = int(max(0, min(100, session.risk_score)))
    session.peak_score = max(session.peak_score, session.risk_score)
    session.risk_level = classify_risk_level(session.risk_score)

    return session.risk_score, session.risk_level


def detect_stress_from_head_movement(yaw, pitch):
    """
    Stress heuristic: high variance in yaw/pitch over a rolling window
    suggests restless head movement (distinct from a single deliberate
    mirror-check). Returns True if the recent window looks stressed.
    """
    now = time.time()
    session.pose_window.append((now, yaw, pitch))
    while session.pose_window and now - session.pose_window[0][0] > STRESS_WINDOW_SECONDS:
        session.pose_window.popleft()

    if len(session.pose_window) < 10:
        return False

    yaws = [p[1] for p in session.pose_window]
    pitches = [p[2] for p in session.pose_window]
    yaw_std = float(np.std(yaws))
    pitch_std = float(np.std(pitches))
    # Cutoffs come from this driver's Distraction slider (ux_thresholds.py),
    # set in MonitoringSession.reset() — falls back to sane defaults if a
    # driver has never touched the slider.
    return (yaw_std > session.distraction_yaw_cutoff) or (pitch_std > session.distraction_pitch_cutoff)


def maybe_trigger_alert(alert_type, skip_whatsapp=False):
    """
    Log an in-memory alert for this session and play the matching voice
    alert. Called by process_frame() only on a genuine risk_level severity
    increase for the current episode (see session.last_alert_level) —
    always AFTER update_risk_score() has finalized this frame's
    score/level, so the alert's logged risk_score always matches what's
    shown live everywhere else, and a DANGER-tagged alert always
    coincides with the snapshot capture in the same frame.

    skip_whatsapp: set True by Demo Mode (run_demo_sequence()) so that
    rehearsing the demo never fires a real WhatsApp message even if
    Twilio credentials are configured — a real DANGER detection should
    still trigger the real emergency alert, so this defaults to False.
    """
    if time.time() < session.alert_grace_until:
        # Driver just acknowledged via the "I'm okay" voice command —
        # skip logging/voicing a new alert for this short window. The
        # underlying risk_score/risk_level computation is untouched by
        # this and keeps running honestly off real camera data; only the
        # alert log + voice nag are paused.
        return

    session.total_alerts += 1
    lat, lng = session.last_gps
    session.alerts.append({
        "alert_type": alert_type,
        "risk_score": session.risk_score,
        "location_lat": lat,
        "location_lng": lng,
        "alert_time": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "snapshot_path": None,  # filled in by process_frame() if this frame ends up DANGER
        "video_clip_path": None,  # filled in later by process_frame() once the incident clip finishes recording
    })

    lang = (session.driver or {}).get("preferred_language", "en")
    level = "danger" if session.risk_level == "DANGER" else "warning"
    voice.play_alert(lang=lang, level=level)

    if session.risk_level == "DANGER" and not skip_whatsapp:
        trigger_emergency_whatsapp(reason=alert_type)


def maybe_save_snapshot(frame):
    """
    Save a JPEG snapshot of the current frame — called on DANGER events.
    Returns just the filename (not full path), for storing in
    alerts.snapshot_path and serving via GET /snapshots/<filename>.
    """
    if session.driver_id is None:
        return None
    filename = f"driver{session.driver_id}_{int(time.time())}.jpg"
    path = os.path.join(SNAPSHOTS_DIR, filename)
    cv2.imwrite(path, frame)
    session.snapshots_count += 1
    return filename


def save_incident_clip(frames):
    """
    Feature: incident video clips. Writes a short MP4 (pre + post
    DANGER-trigger frames, from session.incident_frames) to disk via
    OpenCV's VideoWriter. Returns just the filename, same convention as
    maybe_save_snapshot(), for storing in alerts.video_clip_path and
    serving via GET /clips/<filename>.

    Assumes all frames are the same width/height (true here, since
    they all come from the same camera feed) — takes the size from the
    first frame.
    """
    if session.driver_id is None or not frames:
        return None
    filename = f"driver{session.driver_id}_{int(time.time())}.mp4"
    path = os.path.join(CLIPS_DIR, filename)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 15.0, (w, h))
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()
    return filename


# ===========================================================================
# CAMERA / DETECTION LOOP
# ===========================================================================

def process_frame(frame):
    """
    Run face mesh + EAR/MAR/head-pose/risk scoring on a single BGR frame.
    Draws overlay annotations and returns the annotated frame. Mutates
    the module-level `session` object with updated counters/alerts.
    """
    h, w = frame.shape[:2]

    if session.demo_active:
        # Demo Mode: skip real detection entirely — run_demo_sequence()
        # (a background thread) is driving session.risk_score/risk_level
        # directly on a scripted timeline. Just draw the current state.
        level = session.risk_level
        score = session.risk_score
        color = {"SAFE": (0, 200, 0), "WARNING": (0, 165, 255), "DANGER": (0, 0, 255)}[level]
        cv2.rectangle(frame, (0, 0), (w, 34), color, -1)
        cv2.putText(frame, f"{level}  ({score})", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "DEMO MODE — SIMULATED", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    eyes_closed = False
    yawning = False
    face_missing = results.multi_face_landmarks is None
    no_face = False  # debounced/sustained version, set below — this is what actually feeds risk scoring
    stress_detected = False
    alerts_before_this_frame = len(session.alerts)

    if face_missing:
        session.no_face_frames += 1
        # Require sustained absence (NO_FACE_CONSEC_FRAMES) before treating
        # this as a real signal — a single dropped frame is normal camera
        # noise, not evidence the driver's face has actually left the frame.
        no_face = session.no_face_frames >= NO_FACE_CONSEC_FRAMES
    else:
        session.no_face_frames = 0
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = compute_ear(landmarks, LEFT_EYE, w, h)
        right_ear = compute_ear(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        mar = compute_mar(landmarks, MOUTH, w, h)
        pitch, yaw, roll = compute_head_pose(landmarks, w, h)

        # Accumulate for personalized-threshold calibration (see
        # threshold_calibration.py) — only while NOT already flagged
        # drowsy, so a real drowsy episode doesn't skew the "resting"
        # baseline we're trying to learn.
        if ear >= session.ear_threshold and mar <= session.mar_threshold:
            session.ear_sum += ear
            session.mar_sum += mar
            session.ear_mar_samples += 1

        stress_detected = detect_stress_from_head_movement(yaw, pitch)
        if stress_detected and not session.stress_currently_active:
            session.stress_events_count += 1
        session.stress_currently_active = stress_detected

        # Eyes — consecutive-frame threshold comes from this driver's Blink
        # Duration slider (ux_thresholds.py), set in MonitoringSession.reset()
        # Contextual alert sensitivity — see get_driving_context(). Applies
        # the multiplier to this driver's calibrated base threshold rather
        # than mutating session.ear_consec_frames itself, so the personal
        # calibration value stays intact between frames/contexts.
        consec_needed = max(3, int(session.ear_consec_frames * get_driving_context()["sensitivity_multiplier"]))
        if ear < session.ear_threshold:
            session.eye_closed_frames += 1
            eyes_closed = session.eye_closed_frames >= consec_needed
        else:
            if 2 <= session.eye_closed_frames < consec_needed:
                session.blink_count += 1  # counts as a normal blink, not a drowsy episode
            # Decay instead of hard-reset: a single noisy frame where EAR
            # briefly reads above threshold — common even with genuinely
            # closed eyes, due to landmark jitter — shouldn't erase several
            # seconds of accumulated eye-closure evidence in one frame.
            # That hard-reset was the reason a real, sustained eye closure
            # could get knocked back down before the risk score ever
            # climbed high enough to reach DANGER. Decrementing instead
            # still brings the counter back down quickly on a genuine
            # eyes-open stretch (3x the rise rate), but tolerates isolated
            # single-frame blips without losing all accumulated progress.
            session.eye_closed_frames = max(0, session.eye_closed_frames - 3)
            eyes_closed = session.eye_closed_frames >= consec_needed

        # Mouth / yawns — same contextual multiplier as the eyes above
        mar_consec_needed = max(3, int(MAR_CONSEC_FRAMES * get_driving_context()["sensitivity_multiplier"]))
        if mar > session.mar_threshold:
            session.mouth_open_frames += 1
            yawning = session.mouth_open_frames >= mar_consec_needed
            if session.mouth_open_frames == mar_consec_needed:
                session.yawn_count += 1
        else:
            # Same debounce reasoning as the eye-closed counter above —
            # a single frame where MAR briefly dips at the start/end of a
            # yawn shouldn't erase the whole yawn mid-detection.
            session.mouth_open_frames = max(0, session.mouth_open_frames - 3)
            yawning = session.mouth_open_frames >= mar_consec_needed

        # Draw overlay
        for idx in LEFT_EYE + RIGHT_EYE:
            pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(frame, pt, 1, (0, 255, 0), -1)
        for idx in MOUTH:
            pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            cv2.circle(frame, pt, 1, (0, 165, 255), -1)

        cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Yaw: {yaw:.0f} Pitch: {pitch:.0f}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    score, level = update_risk_score(eyes_closed, yawning, no_face, stress_detected)

    # Fire an alert on a genuine severity INCREASE (SAFE -> WARNING or
    # WARNING/SAFE -> DANGER) for this episode, now that score/level are
    # authoritative for this frame — not on an early frame-counter
    # crossing that happens before the score has actually caught up.
    # This is what makes the alert's logged risk_score/level always match
    # what's shown live elsewhere, and what makes DANGER-level alerts
    # always coincide with the snapshot capture below.
    if _LEVEL_SEVERITY[level] > _LEVEL_SEVERITY[session.last_alert_level]:
        if no_face:
            alert_type = "NO_FACE"
        elif eyes_closed and yawning:
            alert_type = "DROWSY"  # combined episode — eyes are the more urgent signal
        elif eyes_closed:
            alert_type = "DROWSY"
        elif yawning:
            alert_type = "FATIGUE"
        elif stress_detected:
            alert_type = "DISTRACTION"
        else:
            alert_type = "RISK"  # shouldn't normally happen if level rose, but a safe fallback
        maybe_trigger_alert(alert_type)
        session.last_alert_level = level
    elif _LEVEL_SEVERITY[level] < _LEVEL_SEVERITY[session.last_alert_level]:
        # Risk has genuinely subsided — drop the watermark so a later,
        # separate episode can alert again from WARNING rather than being
        # permanently silenced for the rest of the session.
        session.last_alert_level = level

    if level == "DANGER":
        snapshot_filename = maybe_save_snapshot(frame)
        # Link this snapshot to whichever alert fired THIS frame, if any.
        if len(session.alerts) > alerts_before_this_frame:
            session.alerts[-1]["snapshot_path"] = snapshot_filename

    color = {"SAFE": (0, 200, 0), "WARNING": (0, 165, 255), "DANGER": (0, 0, 255)}[level]
    cv2.rectangle(frame, (0, 0), (w, 34), color, -1)
    cv2.putText(frame, f"{level}  ({score})", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Feature: incident video clips. A NEW DANGER-level alert this frame
    # starts a clip using the rolling buffer as "before" context; once
    # started, every subsequent frame (this one included) gets appended
    # as "after" context until the post-incident window is used up, at
    # which point the clip is written to disk and linked to the alert
    # that triggered it — same linking pattern as the snapshot above.
    if level == "DANGER" and len(session.alerts) > alerts_before_this_frame and not session.incident_recording_active:
        session.incident_frames = list(session.frame_buffer)
        session.incident_recording_active = True
        session.incident_frames_remaining = POST_INCIDENT_FRAMES
        session.incident_alert_index = len(session.alerts) - 1

    if session.incident_recording_active:
        session.incident_frames.append(frame.copy())
        session.incident_frames_remaining -= 1
        if session.incident_frames_remaining <= 0:
            clip_filename = save_incident_clip(session.incident_frames)
            idx = session.incident_alert_index
            if idx is not None and idx < len(session.alerts):
                session.alerts[idx]["video_clip_path"] = clip_filename
            session.incident_recording_active = False
            session.incident_frames = []
            session.incident_alert_index = None

    session.frame_buffer.append(frame.copy())

    return frame


def gen_frames():
    """MJPEG generator for the /video_feed route."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[app.py] WARNING: could not open camera at index", CAMERA_INDEX)
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if session.active:
                frame = process_frame(frame)
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()


# ===========================================================================
# WEATHER (Open-Meteo — free, no API key)
# ===========================================================================

def fetch_weather(lat, lng):
    """
    Current weather via Open-Meteo. Used to flag conditions (heavy rain,
    fog, extreme heat) that compound driver fatigue risk.
    """
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lng, "current_weather": "true"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json().get("current_weather", {})
        return {
            "temperature_c": data.get("temperature"),
            "windspeed_kmh": data.get("windspeed"),
            "weather_code": data.get("weathercode"),
        }
    except requests.RequestException as e:
        return {"error": f"Weather service unavailable: {e}"}


# ===========================================================================
# FIND NEARBY (OpenStreetMap Overpass — free, no API key)
# ===========================================================================

NEARBY_QUERY_TAGS = {
    "hospital": 'node["amenity"="hospital"]',
    "police": 'node["amenity"="police"]',
    "parking": 'node["amenity"="parking"]',
    "rest_stop": 'node["highway"="rest_area"]',
}


def find_nearby(lat, lng, category, radius_m=8000):
    """
    Query OpenStreetMap's Overpass API for points of interest near the
    driver's current GPS location. category must be one of
    NEARBY_QUERY_TAGS' keys.
    """
    tag_filter = NEARBY_QUERY_TAGS.get(category)
    if tag_filter is None:
        return {"error": f"Unknown category '{category}'. Choose from {list(NEARBY_QUERY_TAGS)}."}

    query = f"""
    [out:json][timeout:10];
    {tag_filter}(around:{radius_m},{lat},{lng});
    out center 10;
    """
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter", data={"data": query}, timeout=12
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        results = [{
            "name": el.get("tags", {}).get("name", "Unnamed"),
            "lat": el.get("lat"),
            "lng": el.get("lon"),
        } for el in elements]
        return {"category": category, "count": len(results), "results": results}
    except requests.RequestException as e:
        return {"error": f"Nearby-places service unavailable: {e}"}


# ===========================================================================
# EMERGENCY WHATSAPP ALERT (best-effort — needs real credentials to send)
# ===========================================================================

# Configure via environment variables — never hardcode credentials.
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM")   # e.g. "whatsapp:+14155238886"
EMERGENCY_CONTACT_WHATSAPP = os.environ.get("EMERGENCY_CONTACT_WHATSAPP")  # e.g. "whatsapp:+91XXXXXXXXXX"


def trigger_emergency_whatsapp(reason):
    """
    Send a WhatsApp alert with the driver's GPS location on a DANGER
    event, via Twilio's WhatsApp API. No-ops with a log message if
    credentials aren't configured — so a hackathon demo without Twilio
    set up doesn't crash, it just skips this step visibly.

    Feature: multi-contact alerts. Sends to every number in this driver's
    emergency_contacts (comma-separated, saved via the profile editor),
    falling back to the single global EMERGENCY_CONTACT_WHATSAPP env var
    if the driver hasn't configured any personal contacts — so existing
    setups with only the env var configured keep working unchanged.
    """
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM]):
        print(f"[app.py] Emergency WhatsApp alert SKIPPED (Twilio not configured). "
              f"Reason: {reason}, GPS: {session.last_gps}")
        return False

    driver_contacts_raw = (session.driver or {}).get("emergency_contacts") or ""
    contacts = [c.strip() for c in driver_contacts_raw.split(",") if c.strip()]
    if not contacts and EMERGENCY_CONTACT_WHATSAPP:
        contacts = [EMERGENCY_CONTACT_WHATSAPP]
    if not contacts:
        print(f"[app.py] Emergency WhatsApp alert SKIPPED (no contacts configured for this driver "
              f"and no EMERGENCY_CONTACT_WHATSAPP fallback set). Reason: {reason}")
        return False

    lat, lng = session.last_gps
    maps_link = f"https://maps.google.com/?q={lat},{lng}" if lat and lng else "location unavailable"
    driver_name = (session.driver or {}).get("driver_name", "Unknown driver")
    body = (f"⚠️ SixthSense AI Alert\nDriver: {driver_name}\nReason: {reason}\n"
            f"Risk score: {session.risk_score}\nLocation: {maps_link}")

    try:
        from twilio.rest import Client  # pip install twilio — imported lazily,
                                          # only needed if credentials are set
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        sent_count = 0
        for contact in contacts:
            try:
                client.messages.create(from_=TWILIO_WHATSAPP_FROM, to=contact, body=body)
                sent_count += 1
            except Exception as e:
                print(f"[app.py] Emergency WhatsApp alert FAILED for {contact}: {e}")
        return sent_count > 0
    except Exception as e:
        print(f"[app.py] Emergency WhatsApp alert FAILED: {e}")
        return False


# ===========================================================================
# ROUTES
# ===========================================================================

@app.route("/video_feed")
def video_feed():
    """MJPEG live camera stream for the frontend <img> tag."""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/monitoring/start", methods=["POST"])
def start_monitoring():
    """
    Begin a monitoring session for a driver. Body: {"driver_id": 1}
    Pulls that driver's calibrated thresholds from the database.
    """
    body = request.get_json(silent=True) or {}
    driver_id = body.get("driver_id")
    if driver_id is None:
        return jsonify({"success": False, "error": "driver_id is required."}), 400

    driver = db.get_driver(driver_id)
    if driver is None:
        return jsonify({"success": False, "error": f"Driver {driver_id} not found."}), 404

    session.reset(driver_id, driver)
    return jsonify({"success": True, "data": {
        "driver_id": driver_id,
        "ear_threshold": session.ear_threshold,
        "mar_threshold": session.mar_threshold,
        "started_at": datetime.now().isoformat(),
    }})


@app.route("/api/monitoring/stop", methods=["POST"])
def stop_monitoring():
    """
    End the current session: persist it, log all alerts, feed the
    calibration module, refresh analytics, and return a summary
    (including an accident-risk prediction for the session just ended).
    """
    if not session.active or session.driver_id is None:
        return jsonify({"success": False, "error": "No active session."}), 400

    session.stop()
    duration = int(time.time() - session.start_time)
    driver_id = session.driver_id

    session_id = db.save_session(
        driver_id=driver_id,
        session_duration=duration,
        peak_score=session.peak_score,
        total_alerts=session.total_alerts,
        yawn_count=session.yawn_count,
        blink_count=session.blink_count,
        # Classify from peak_score, NOT session.risk_level — risk_level is
        # whatever the score happens to be at this exact instant, and it
        # naturally decays back down (RISK_FALL_STEP) once the triggering
        # episode ends. If the driver's eyes reopened and some time passed
        # before they clicked Stop, risk_level could already be back to
        # SAFE/WARNING even though the session genuinely peaked at DANGER
        # — which is exactly what Session History should show.
        final_level=classify_risk_level(session.peak_score),
        snapshots_count=session.snapshots_count,
    )

    for alert in session.alerts:
        db.save_alert(session_id=session_id, driver_id=driver_id, **alert)

    db.recompute_weekly_analytics(driver_id)

    # Feed the personalized-calibration module with this session's resting
    # EAR/MAR average (only meaningful if we collected enough clean samples)
    calibration_result = None
    if session.ear_mar_samples >= 30:  # ignore very short/aborted sessions
        avg_ear = session.ear_sum / session.ear_mar_samples
        avg_mar = session.mar_sum / session.ear_mar_samples
        calibration_result = calib.record_session_baseline(
            driver_id=driver_id, avg_ear=avg_ear, avg_mar=avg_mar, session_id=session_id)

    risk_prediction = None
    try:
        saved_session = db.get_session(session_id)
        probability = rp.predict_accident_risk_from_session(saved_session)
        risk_prediction = {
            "accident_risk_probability": probability,
            "risk_level": rp.risk_level_label(probability),
        }
    except Exception as e:
        print(f"[app.py] Risk prediction skipped: {e}")

    return jsonify({"success": True, "data": {
        "session_id": session_id,
        "duration_seconds": duration,
        "peak_score": session.peak_score,
        "total_alerts": session.total_alerts,
        "calibration": calibration_result,
        "next_30min_risk_prediction": risk_prediction,
    }})


@app.route("/api/monitoring/live-prediction", methods=["GET"])
def live_prediction():
    """
    Feature: Live Fatigue Early-Warning. Unlike /api/driver/<id>/risk-
    prediction (which only looks at the last COMPLETED session), this
    predicts the next-30-min accident risk from the CURRENTLY IN-PROGRESS
    session — elapsed time so far, yawns/blinks/alerts so far this
    session — so a fleet manager (or the driver) can see a forecast climb
    BEFORE the camera detection itself reaches WARNING/DANGER, not just
    after. Requires an active session; harmless to poll every ~30s.
    """
    if not session.active:
        return jsonify({"success": False, "error": "No active monitoring session."}), 400

    elapsed_sec = time.time() - session.start_time if session.start_time else 0
    synthetic_session = {
        "session_duration": elapsed_sec,
        "yawn_count": session.yawn_count,
        "blink_count": session.blink_count,
        "total_alerts": session.total_alerts,
    }
    try:
        probability = rp.predict_accident_risk_from_session(synthetic_session)
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {e}"}), 500

    return jsonify({"success": True, "data": {
        "accident_risk_probability": probability,
        "risk_level": rp.risk_level_label(probability),
        "elapsed_minutes": round(elapsed_sec / 60.0, 1),
        "based_on": "in-progress session (live)",
    }})


@app.route("/api/monitoring/status", methods=["GET"])
def monitoring_status():
    """Live poll endpoint for the frontend to show the current risk score/level."""
    return jsonify({"success": True, "data": {
        "active": session.active,
        "driver_id": session.driver_id,
        "risk_score": session.risk_score,
        "risk_level": session.risk_level,
        "blink_count": session.blink_count,
        "yawn_count": session.yawn_count,
        "total_alerts": session.total_alerts,
        "demo_active": session.demo_active,
        "driving_context": get_driving_context(),
        "stress_detected": session.stress_currently_active,
        "stress_events_count": session.stress_events_count,
    }})


def run_demo_sequence():
    """
    Runs in a background thread. Scripts a full SAFE -> WARNING -> DANGER
    -> alert -> recovery cycle over ~24 seconds, driving session.risk_score
    / risk_level / yawn_count / blink_count directly — independent of the
    real camera loop, which process_frame() skips while session.demo_active
    is True. Lets a presenter show the whole detection story on command,
    without needing to actually act drowsy on camera for a live demo.

    Deliberately reuses the same alert-append + voice-alert path as a real
    detection so the alert log and voice output are genuine — but skips the
    WhatsApp emergency trigger, so rehearsing the demo doesn't repeatedly
    fire a real WhatsApp message.
    """
    timeline = [
        # (duration_seconds, target_score, level, note)
        (3.0, 12, "SAFE", None),
        (4.0, 45, "WARNING", "distraction_sim"),
        (3.0, 65, "WARNING", None),
        (4.0, 88, "DANGER", "drowsy_sim"),
        (3.0, 88, "DANGER", None),   # hold at peak so it's clearly visible
        (5.0, 10, "SAFE", None),     # recover
    ]

    steps_per_leg = 20
    for duration, target_score, level, note in timeline:
        if session.demo_stop_requested:
            break
        start_score = session.risk_score
        step_sleep = duration / steps_per_leg
        for i in range(steps_per_leg):
            if session.demo_stop_requested:
                break
            frac = (i + 1) / steps_per_leg
            session.risk_score = int(round(start_score + (target_score - start_score) * frac))
            session.risk_level = level
            session.peak_score = max(session.peak_score, session.risk_score)
            time.sleep(step_sleep)

        if note == "distraction_sim":
            session.blink_count += 1
        elif note == "drowsy_sim":
            session.yawn_count += 1
            # Reuse the real alert-logging + voice-playing path so the
            # alert log and voice output are genuine, not reimplemented —
            # but explicitly suppress the WhatsApp emergency trigger, so
            # rehearsing the demo never fires a real WhatsApp message even
            # if Twilio credentials are configured on this machine.
            maybe_trigger_alert("DROWSY (demo)", skip_whatsapp=True)

    session.demo_active = False
    session.demo_stop_requested = False


@app.route("/api/monitoring/acknowledge", methods=["POST"])
def acknowledge_alert():
    """
    Voice command ("I'm okay") support. Called by the frontend after its
    browser-side speech recognition hears the driver say "I'm okay" during
    an active WARNING/DANGER alert. Opens a short grace window during
    which maybe_trigger_alert() won't log a new alert or replay the voice
    nag — but deliberately does NOT touch risk_score/risk_level, which
    stay 100% driven by real camera detection. Saying "I'm okay" pauses
    the alarm for a moment; it can't talk the system into reporting SAFE
    when the camera still sees otherwise.
    """
    if not session.active:
        return jsonify({"success": False, "error": "No active monitoring session."}), 400

    GRACE_SECONDS = 20
    session.alert_grace_until = time.time() + GRACE_SECONDS
    return jsonify({"success": True, "data": {"grace_seconds": GRACE_SECONDS}})


@app.route("/api/demo/start", methods=["POST"])
def start_demo():
    """
    Kick off the scripted SAFE -> WARNING -> DANGER demo sequence. Requires
    an active monitoring session (so the demo has a real driver_id/session
    to attach its simulated alert to) — start one via /api/monitoring/start
    first.
    """
    if not session.active:
        return jsonify({"success": False, "error": "Start a monitoring session before running Demo Mode."}), 400
    if session.demo_active:
        return jsonify({"success": False, "error": "Demo Mode is already running."}), 400

    session.demo_active = True
    session.demo_stop_requested = False
    threading.Thread(target=run_demo_sequence, daemon=True).start()
    return jsonify({"success": True, "data": {"demo_active": True}})


@app.route("/api/demo/stop", methods=["POST"])
def stop_demo():
    """Stop an in-progress demo sequence early and reset to SAFE."""
    if not session.demo_active:
        return jsonify({"success": False, "error": "Demo Mode is not running."}), 400

    session.demo_stop_requested = True
    session.risk_score = 0
    session.risk_level = "SAFE"
    return jsonify({"success": True, "data": {"demo_active": False}})


@app.route("/stats", methods=["GET"])
def root_stats():
    """
    Root-level (no /api prefix) live risk poll — matches
    src/api/driverApi.js:getCurrentRiskStats(), which fetches
    `${API_BASE}/stats` directly and reads {risk_score, risk_level} off
    the raw response, no envelope. Kept separate from
    /api/monitoring/status above (same underlying data, different shape/
    path) since that route already has its own consumer.
    """
    return jsonify({
        "risk_score": session.risk_score,
        "risk_level": session.risk_level.lower(),
    })


@app.route("/api/monitoring/gps", methods=["POST"])
def update_gps():
    """
    Frontend pushes the vehicle's current GPS coordinates here periodically
    (e.g. from a phone mount or OBD GPS module). Used to geotag alerts,
    power the emergency WhatsApp / find-nearby features, AND persisted to
    the driver's row so the Fleet Dashboard map (/api/fleet/map,
    /api/fleet/drivers) shows an up-to-date position.
    Body: {"lat": 12.9716, "lng": 77.5946}
    """
    body = request.get_json(silent=True) or {}
    lat, lng = body.get("lat"), body.get("lng")
    if lat is None or lng is None:
        return jsonify({"success": False, "error": "lat and lng are required."}), 400

    # Derive speed from the distance/time between this fix and the last
    # one — feeds get_driving_context() for contextual alert sensitivity.
    # Only trust it as real motion if the two fixes are a sensible amount
    # of time apart (avoids a divide-by-near-zero speed spike if two
    # pushes land unusually close together).
    now = time.time()
    prev_lat, prev_lng = session.last_gps
    if prev_lat is not None and session.last_gps_time is not None:
        dt_hours = (now - session.last_gps_time) / 3600.0
        if dt_hours > (2.0 / 3600.0):  # at least 2 real seconds apart
            dist_km = haversine_km(prev_lat, prev_lng, lat, lng)
            session.speed_kmh = round(dist_km / dt_hours, 1)

    session.last_gps = (lat, lng)
    session.last_gps_time = now
    if session.driver_id is not None:
        db.update_driver_location(session.driver_id, lat, lng)
    return jsonify({"success": True})


@app.route("/api/weather", methods=["GET"])
def weather_route():
    """?lat=..&lng=.. — current weather at the vehicle's location."""
    lat, lng = request.args.get("lat", type=float), request.args.get("lng", type=float)
    if lat is None or lng is None:
        return jsonify({"success": False, "error": "lat and lng query params are required."}), 400
    return jsonify({"success": True, "data": fetch_weather(lat, lng)})


@app.route("/api/nearby", methods=["GET"])
def nearby_route():
    """?lat=..&lng=..&category=hospital|police|parking|rest_stop"""
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    category = request.args.get("category", default="hospital")
    if lat is None or lng is None:
        return jsonify({"success": False, "error": "lat and lng query params are required."}), 400
    return jsonify({"success": True, "data": find_nearby(lat, lng, category)})


@app.route("/snapshots/<path:filename>", methods=["GET"])
def serve_snapshot(filename):
    """
    Serves DANGER-event snapshots saved by maybe_save_snapshot(). Without
    this route, alerts.snapshot is always None and AlertsPage.jsx's
    snapshot preview modal has nothing to show — see
    FRONTEND_RECONCILIATION.md.
    """
    from flask import send_from_directory
    return send_from_directory(SNAPSHOTS_DIR, filename)


@app.route("/clips/<path:filename>", methods=["GET"])
def serve_clip(filename):
    """
    Serves incident video clips saved by save_incident_clip() — the
    before/after DANGER-event MP4s. Same pattern as serve_snapshot()
    above, just for CLIPS_DIR instead of SNAPSHOTS_DIR.
    """
    from flask import send_from_directory
    return send_from_directory(CLIPS_DIR, filename)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple liveness probe for the frontend's Demo Mode fallback logic."""
    return jsonify({"success": True, "status": "ok", "time": datetime.now().isoformat()})


# ===========================================================================
# UNIFIED FRONTEND SERVING
# ---------------------------------------------------------------------------
# Serves the built React app (frontend/dist, produced by `npm run build`)
# directly from this same Flask process, so the whole project — driver
# camera view, fleet dashboard, alerts, reports — runs from one command
# (`python app.py`) on one port, instead of needing a second `npm run dev`
# terminal. This route is registered last and only matches paths that
# aren't already handled above (/api/*, /video_feed, /snapshots/*, /stats),
# since Flask/Werkzeug always prefers the more specific routes over this
# catch-all regardless of registration order.
#
# If frontend/dist doesn't exist yet (i.e. `npm run build` hasn't been run),
# this serves a plain instruction page instead of crashing.
# ===========================================================================
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "dist")


@app.route("/", defaults={"req_path": ""})
@app.route("/<path:req_path>")
def serve_frontend(req_path):
    from flask import send_from_directory

    if not os.path.isdir(FRONTEND_DIST):
        return (
            "<h2>Frontend not built yet</h2>"
            "<p>Run <code>cd ../frontend &amp;&amp; npm install &amp;&amp; npm run build</code>, "
            "then restart <code>python app.py</code>.</p>",
            200,
        )

    # Serve a real static file (JS/CSS/images) if the path matches one
    candidate = os.path.join(FRONTEND_DIST, req_path)
    if req_path and os.path.isfile(candidate):
        return send_from_directory(FRONTEND_DIST, req_path)

    # Otherwise fall back to index.html so React Router can handle the
    # client-side route (e.g. /driver/1, /alerts, /reports)
    return send_from_directory(FRONTEND_DIST, "index.html")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"[app.py] SixthSense AI — everything running on one server")
    print(f"[app.py] Open the app:         http://localhost:{port}")
    print(f"[app.py] Live video feed:      http://localhost:{port}/video_feed")
    print(f"[app.py] API base:             http://localhost:{port}/api/")
    serve(app, host="0.0.0.0", port=port)
