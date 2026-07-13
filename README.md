# 👁️ SixthSense AI

**Edge AI driver fatigue detection & fleet safety platform**
Tata Technologies InnoVent-27 · Category: AI at the Edge / Edge AI for ADAS
Team SentriX AI

> A real-time, camera-based system that watches for the early signs of driver drowsiness — before they become a crash statistic.

---

## The Problem

Fatigue-related lapses are one of the leading causes of commercial-vehicle accidents in India, especially on long highway hauls, night shifts, and mining/logistics routes with poor connectivity. Most fleets have no way to catch a driver's attention slipping until it's already too late. SixthSense AI is built to close that gap — cheaply, offline-capable, and at the edge.

## What It Does

SixthSense AI runs on a camera pointed at the driver and continuously analyzes their eyes, mouth, and head position to detect drowsiness and distraction in real time — then acts on it, both in-cabin and back at the fleet office.

- 👀 **Real-time drowsiness detection** — MediaPipe Face Mesh tracks facial landmarks to compute Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head pose, catching microsleeps, prolonged eye closure, and yawning as they happen.
- 🧠 **Predictive risk scoring** — A Random Forest model estimates the probability of a fatigue-related lapse in the next 30 minutes, using time of day, session duration, yawn count, blink rate, and alert history.
- 🎯 **Per-driver calibration** — Every driver's resting eye/mouth geometry is different. The system learns each driver's individual baseline over their first 5 sessions and auto-tunes their personal thresholds, with guard rails against noisy single-session data.
- 🗣️ **Offline multilingual voice alerts** — Alerts are pre-generated and cached in English, Kannada, and Hindi, so they play with zero network dependency — built for highway and mine-site conditions with unreliable signal.
- 📲 **Emergency WhatsApp alerts** — Twilio-based emergency notifications for critical fatigue events, with a graceful no-op if not configured.
- 📊 **Fleet dashboard** — A React web app for fleet managers: live driver risk leaderboard, incident charts, alerts feed, and per-driver profile analytics.
- 🔌 **REST API** — 10 Flask endpoints covering driver registration, session/alert logging, weekly & daily analytics, and live risk-prediction/calibration data for the frontend.


## Tech Stack

**Backend / ML:** Python 3.11 · Flask · SQLite · OpenCV · MediaPipe (Face Mesh) · scikit-learn (Random Forest) · gTTS + pygame · Twilio · joblib

**Frontend:** React · React Router · React-Leaflet · Chart.js · Lucide Icons · Vite

**Design:** Figma-ready HTML/CSS mockups, SVG architecture & data-flow diagrams, full brand system

## System Architecture

```
                    ┌─────────────────────────┐
                    │      Live Camera Loop     │
                    │  MediaPipe Face Mesh →    │
                    │   EAR / MAR / Head Pose   │
                    └──────────┬────────────────┘
                               │
                 ┌─────────────┼─────────────────┐
                 ▼             ▼                 ▼
         offline_voice.py  api_routes.py   threshold_calibration.py
         (spoken alerts,   (Flask REST      (per-driver EAR/MAR
          zero internet)    Blueprint)        adaptive baseline)
                 │             │                 │
                 │             ▼                 │
                 │        database.py ◄───────────┘
                 │        (SQLite: drivers, sessions,
                 │         alerts, driver_analytics)
                 │             ▲
                 │             │
                 └────► risk_predictor.py
                        (Random Forest — 30-min
                         fatigue-lapse risk)
```

## Project Structure

```
SixthSenseAI/
├── backend/          Flask API + ML pipeline
│   ├── app.py                    Main app: live camera loop + server
│   ├── api_routes.py             10 REST endpoints (Blueprint)
│   ├── database.py               SQLite schema + CRUD
│   ├── risk_predictor.py         Random Forest fatigue risk model
│   ├── threshold_calibration.py  Per-driver adaptive calibration
│   ├── offline_voice.py          Multilingual offline voice alerts
│   ├── ux_thresholds.py          UX-slider → CV-threshold conversion
│   └── requirements.txt
├── frontend/         React fleet dashboard (Vite)
│   └── src/components/           Fleet Dashboard, Driver Monitoring,
│                                  Driver Profile, Alerts, Reports
└── design/           Brand system, diagrams, mockups, pitch deck
```

## Getting Started

### Backend
```bash
cd backend
pip install -r requirements.txt
python3 app.py
```
Runs at `http://localhost:5000`. First boot needs internet once to generate voice-alert audio files; after that, the detection loop itself is fully offline-capable.

> ⚠️ **Pinned to `mediapipe==0.10.13`** — newer pip builds (0.10.30+) dropped the classic `mp.solutions.face_mesh` API this project depends on. Don't upgrade without checking `hasattr(mediapipe, "solutions")` first.

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Point it at the backend via the API base URL config if it's not running on `localhost:5000`.

### Design Mockups
Open any `.html` file in `design/` directly in a browser — no build step required. Three of them (`fleet_dashboard`, `driver_monitoring`, `driver_profile_analytics`) fetch live data from the Flask API when it's running, and fall back to demo data otherwise, with an on-screen badge showing which mode is active.

## Validation & Testing

Every backend module ships with a runnable self-test exercising its full functionality against a throwaway database:

| Module | Coverage |
|---|---|
| `database.py` | Full CRUD chain: driver → session → alert → analytics |
| `risk_predictor.py` | Training, hand-checked scenarios, save/load caching |
| `api_routes.py` | All 10 routes incl. error cases, via Flask test client |
| `threshold_calibration.py` | 5-session convergence, guard-rail clamping, reset |

Run the full non-camera suite with:
```bash
python3 backend/test_app_no_camera.py
```

## Honest Status — What's Verified vs. What Needs Real-World Testing

**Verified in sandbox testing:** database CRUD, the full session → calibration → risk-prediction pipeline, all REST routes, the risk-score state machine, EAR/MAR math against synthetic landmark data, GPS persistence, UX-threshold conversion math, and a clean install into a fresh virtualenv with a passing test run.

**Not yet verified — needs real hardware/environment:**
- Live webcam capture (`cv2.VideoCapture`) — no camera was available in the build sandbox
- Voice-alert audio generation — needs live internet access to Google's TTS service
- `/api/weather` and `/api/nearby` — untested against live external APIs from the sandbox
- The React frontend has not yet been run end-to-end against a live backend
- WhatsApp emergency alerts — implemented via Twilio but untested with real credentials; correctly no-ops when unconfigured

**Recommended before demo day:** run the backend on a machine with a real webcam and internet connection, connect the frontend to it, and click through the fleet dashboard, driver monitoring, and alerts flow end-to-end.

## Scope & Limitations

- The risk model predicts **fatigue-related lapse risk**, not accidents in general — no telematics inputs (speed, braking, weather) are used yet.
- Trained on **synthetic data** encoding published drowsy-driving research (peak risk 2–5 AM, secondary post-lunch dip), not real fleet incident data. A `train_on_real_data()` hook is ready for retraining once real sessions accumulate.
- Kannada/Hindi voice alert phrasing was machine-translated and should be reviewed by a native speaker before field deployment.
- Problem-statement statistics in the pitch deck are illustrative estimates, not yet sourced from verified MoRTH data.

## What's Next

- Retrain the risk model on real field session/alert data as it accumulates
- Wire live per-session EAR/MAR averages from the camera loop into the calibration module
- Add telematics inputs (speed, braking) to the risk model if OBD data becomes available
- Native-speaker review of Kannada/Hindi voice alerts
