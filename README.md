# SixthSense AI — Combined Project Package
Tata Technologies InnoVent Hackathon 2026 · Team Apex

Everything for the project in one place: backend, frontend, and design.
Read this first, then the folder-specific README in each section.

```
SixthSenseAI/
├── backend/      ← Python/Flask API + ML (this teammate's work)
├── frontend/     ← React app (src/) — the real production UI
└── design/       ← Bharathi's design deliverables: HTML mockups, SVG
                     diagrams, branding, PPT template
```

---

## Quick start

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
python3 app.py
```
Runs on `http://localhost:5000`. First boot needs internet once (to
generate voice-alert MP3s); after that it's fully offline-capable for
the actual detection loop.

`requirements.txt` is pinned to the exact versions tested together in
the sandbox this was built in, including `mediapipe==0.10.13` — **do not
casually upgrade mediapipe**, newer pip builds (0.10.30+) dropped the
classic `mp.solutions.face_mesh` API this code depends on. Verified: a
completely fresh virtualenv installing only this file, then importing
`app.py` and running the full non-camera test suite, works cleanly.

### 2. Frontend
```bash
cd frontend
npm install react react-dom react-router-dom react-leaflet leaflet \
            lucide-react chart.js react-chartjs-2
npm start   # or your bundler's dev command — no package.json is included yet
```
Point it at the backend via `window.__SIXTHSENSE_API_BASE__` in your
`index.html` if it's not on `localhost:5000`.

**No `package.json`/bundler config included** — this is just the `src/`
tree as extracted from the uploaded files. You'll need to drop it into a
Vite or Create React App scaffold (or send me the existing one and I'll
verify it matches).

### 3. Design
Open any `.html` file directly in a browser — no build step needed. See
`design/README.md` for which three are wired to live API calls
(`fleet_dashboard`, `driver_monitoring`, `driver_profile_analytics`) and
how to point them at your running backend.

---

## What to read next

- **`backend/FRONTEND_RECONCILIATION.md`** — the most important doc if
  something doesn't connect. Explains the two different API response
  conventions in play (React app vs. design-deliverable HTML dashboards)
  and exactly which routes were added/changed to satisfy both.
- **`backend/INTEGRATION_GUIDE.md`** — how the 5 backend modules wire
  into `app.py`'s live camera loop.
- **`backend/README.md`** — module-by-module summary of the ML/backend
  contribution, written for hackathon judges.
- **`design/README.md`** — Bharathi's own notes on every design file,
  including which HTML dashboards need `CORS`/`API_BASE_URL`/`DRIVER_ID`
  configured to go from demo data to live data.

---

## Honest status — what's verified vs. what still needs real-world testing

**Verified in an automated sandbox** (see `backend/test_app_no_camera.py`,
runnable yourself): database CRUD, the full session→calibration→risk-
prediction pipeline, all REST routes (including the dual-shape
`/api/fleet/drivers` that satisfies both frontends), the risk-score state
machine, EAR/MAR math against synthetic landmark data, GPS persistence,
the UX-slider → CV-threshold conversion math, and a clean install of
`requirements.txt` into a fresh virtualenv followed by a successful
`app.py` import + full test run.

**NOT yet verified — needs real hardware/environment:**
- `cv2.VideoCapture(0)` — the actual camera loop. No webcam was available
  in the sandbox this was built in.
- The voice-alert MP3 generation (`offline_voice.py`) — needs a real
  internet connection to Google's TTS servers; the sandbox's network was
  restricted to a small domain whitelist.
- `/api/weather` and `/api/nearby` — same restricted-network issue;
  the code is correct but untested against the live Open-Meteo/Overpass
  APIs from this environment.
- The React frontend has not been run through an actual bundler/dev
  server against the live backend — only the API contract was verified
  by reading the source code side-by-side with the backend routes.
- Emergency WhatsApp alerts — implemented via Twilio, but no credentials
  were available to test an actual send; it correctly no-ops without
  crashing when unconfigured.

**Recommended next step before the demo:** run `backend/app.py` on a real
machine with a webcam and internet, open the React frontend against it,
and click through all five pages once end-to-end.
