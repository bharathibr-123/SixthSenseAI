# SixthSense AI — Integration Guide (for Bharathi's app.py)

This covers how to wire in the 5 backend modules built for the ML/backend
side of the project: `database.py`, `offline_voice.py`, `risk_predictor.py`,
`api_routes.py`, `threshold_calibration.py`.

---

## 1. Files to drop into the project root

Copy these next to your existing `app.py`:

```
SixthSenseAI/
├── app.py                    ← your existing file (unchanged, mostly)
├── database.py
├── offline_voice.py
├── risk_predictor.py
├── api_routes.py
├── threshold_calibration.py
├── models/                   ← auto-created, holds the trained risk model
└── sounds/                   ← auto-created, holds the 6 pre-generated MP3s
```

## 2. What to add to `app.py`

At the top, alongside your existing imports:

```python
import database as db
import offline_voice as voice
import risk_predictor as rp
import threshold_calibration as calib
from api_routes import api_bp
```

Right after you create your Flask `app` object, and BEFORE `app.run(...)`:

```python
# One-time setup — safe to call on every startup
db.init_db()
calib.init_calibration_table()
voice.prepare_all_voice_alerts()   # needs internet the very first run only
rp.load_or_train_model()           # trains once, then loads cached model

app.register_blueprint(api_bp)     # adds all /api/... routes
```

That's it — no other changes needed. Your existing routes, camera loop,
and UI are untouched.

## 3. Wiring into your live monitoring loop

Three places in your existing drowsiness-detection loop need small hooks:

### a) Playing voice alerts (replaces/supplements your current gTTS calls)

```python
# Wherever you currently trigger a spoken warning:
voice.play_alert(lang="en", level="danger")   # or "warning"
# lang can be "en" / "kn" / "hi" depending on the driver's selected language
```

This plays instantly from the cached MP3 — no network call during driving.

### b) Saving a session when monitoring stops

```python
session_id = db.save_session(
    driver_id=current_driver_id,
    session_duration=elapsed_seconds,
    peak_score=max_risk_score_seen,
    total_alerts=len(alerts_this_session),
    yawn_count=yawn_counter,
    blink_count=blink_counter,
    final_level=final_risk_level,       # "SAFE" / "WARNING" / "DANGER"
    snapshots_count=len(saved_snapshots),
)

for alert in alerts_this_session:
    db.save_alert(
        session_id=session_id,
        driver_id=current_driver_id,
        alert_type=alert["type"],       # DROWSY / FATIGUE / WARNING / NO_FACE
        risk_score=alert["score"],
        location_lat=alert.get("lat"),
        location_lng=alert.get("lng"),
    )

db.recompute_weekly_analytics(current_driver_id)
```

### c) Feeding the threshold calibration (the one new piece you'll need to add)

Track a running average of EAR and MAR across the session (only while the
driver is NOT flagged drowsy — you don't want a real drowsy episode
skewing their "resting" baseline). At session end:

```python
calib.record_session_baseline(
    driver_id=current_driver_id,
    avg_ear=session_ear_running_average,
    avg_mar=session_mar_running_average,
    session_id=session_id,
)
```

This auto-calibrates personalized thresholds after 5 sessions — after
that, pull the driver's current thresholds at the START of each session
(instead of using the hardcoded 0.25 / 0.6 defaults):

```python
driver = db.get_driver(current_driver_id)
ear_threshold = driver["ear_threshold"]
mar_threshold = driver["mar_threshold"]
```

## 4. New API endpoints available to your frontend

All under `/api/`, all return `{"success": true/false, "data"/"error": ...}`:

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/driver/register` | Add a new driver |
| GET | `/api/driver/<id>` | Get driver profile |
| GET | `/api/fleet/drivers` | List all drivers |
| PUT | `/api/driver/<id>/thresholds` | Manually override thresholds |
| POST | `/api/session/save` | Save a completed session (+ its alerts) |
| GET | `/api/driver/<id>/sessions` | List a driver's sessions |
| GET | `/api/driver/<id>/analytics` | Weekly + daily analytics |
| GET | `/api/driver/<id>/risk-prediction` | Accident-risk probability, next 30 min |
| GET | `/api/driver/<id>/calibration-status` | Calibration progress (e.g. "3/5") |
| POST | `/api/driver/<id>/calibration-reset` | Reset calibration to defaults |

## 5. Before the demo

- [ ] Run `python3 offline_voice.py` once with real internet to generate
      the 6 MP3s into `sounds/` (my sandbox couldn't reach Google's TTS
      servers to do this for you).
- [ ] Have a Kannada/Hindi speaker sanity-check the alert phrasing in
      `offline_voice.py` → `ALERT_MESSAGES` (it was machine-translated).
- [ ] Run each module's built-in self-test once on your machine
      (`python3 database.py`, `python3 risk_predictor.py`, etc.) to confirm
      everything works in your environment before combining.
- [ ] Decide whether to keep the bonus risk-prediction and calibration
      routes in the demo, or trim scope if time is tight — they're
      self-contained and easy to remove from `api_routes.py` if needed.
