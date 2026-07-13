# SixthSense AI — Design & Frontend Deliverables
Tata Technologies InnoVent Hackathon 2026 · Team Apex
Built by Bharathi B R (UI/UX & Visual Design)

This zip contains every design asset produced for the project, from the
first logo concepts through to the final API-wired dashboard screens.

---

## ⚡ For the backend/ML teammate — read this first

Three of the HTML files fetch live data from your Flask API
(`api_routes.py`). They fall back to demo data automatically if the
server isn't reachable, so they're safe to open standalone at any time.

**To make them actually connect to your live backend:**

1. **Enable CORS** in `app.py` (opening an .html file directly counts as
   a cross-origin request):
   ```
   pip install flask-cors
   ```
   ```python
   from flask_cors import CORS
   CORS(app)
   ```
2. **Check `API_BASE_URL`** at the top of the `<script>` block in each
   file below — defaults to `http://localhost:5000`. Change it if your
   Flask app runs elsewhere.
3. **Check `DRIVER_ID`** in the two driver-specific files — defaults to
   `1`. Change it to match a real `driver_id` in your `sixthsense.db`
   (register one via `POST /api/driver/register` first if the table is
   empty).

Files with live API wiring:
| File | Endpoints it calls |
|---|---|
| `sixthsense_ai_fleet_dashboard.html` | `GET /api/fleet/drivers`, `GET /api/driver/<id>/risk-prediction` |
| `sixthsense_ai_driver_monitoring.html` | `GET /api/driver/<id>/risk-prediction`, `GET /api/driver/<id>/calibration-status` |
| `sixthsense_ai_driver_profile_analytics.html` | `GET /api/driver/<id>/risk-prediction`, `GET /api/driver/<id>/calibration-status` |

Each shows a small **"DEMO DATA" / "LIVE"** badge so it's always obvious
which mode it's in — useful during the actual demo so nobody's confused
about whether the backend is connected.

---

## 📁 File-by-file guide

### Figma-ready screens (open in any browser — real HTML/CSS, not flat images)
- **`sixthsense_ai_fleet_dashboard.html`** — Fleet manager's main dashboard: live fleet map, incident chart, vehicle table, alerts feed, at-risk driver leaderboard. **Wired to live API.**
- **`sixthsense_ai_driver_monitoring.html`** — Live camera monitoring view for a single driver mid-trip: AI overlay on the camera feed, live metrics, voice alert log, session timeline, AI insights (risk prediction + calibration). **Wired to live API.**
- **`sixthsense_ai_driver_profile_analytics.html`** — A driver's long-term profile: achievements, fleet comparison, 12-week safety trend, alert/trip history, AI insights. **Wired to live API.**
- **`sixthsense_ai_mobile_app.html`** — Driver-facing mobile app home screen, shown in an actual phone frame.
- **`sixthsense_ai_alert_popups.html`** — All alert notification states in one view: toast stack, critical emergency modal with countdown, and the driver's phone push notification.

### Diagrams (SVG — drag straight into Figma, PPT, or Google Slides)
- **`sixthsense_ai_system_architecture.svg`** — Full system architecture: edge device → connectivity → Flask/SQLite backend → external services → apps.
- **`sixthsense_ai_data_flow_diagram.svg`** — Standard DFD notation showing exactly what data moves where, including the calibration feedback loop and risk-prediction flow.
- **`sixthsense_ai_hardware_setup.svg`** — Physical hardware install diagram: camera, edge compute unit, GPS/4G module, power, cloud.

### Branding
- **`sixthsense_ai_logo_final.svg`** — Final logo: primary color icon, monochrome white/black variants, horizontal lockup with wordmark.
- **`sixthsense_ai_logo_concepts.svg`** — The three original concept sketches (eye-in-shield, radar-eye, hexagon-sensor) kept for reference — the final logo merges the shield and radar-eye ideas.
- **`sixthsense_ai_color_palette.svg`** — Full color system: hex/RGB values, usage rules, and text-contrast pairings.
- **`sixthsense_ai_icon_set.svg`** — 12 single-color line icons covering every core feature.

### Presentation
- **`SixthSense_AI_PPT_Template.pptx`** — Real, editable 5-slide deck: title, problem statistics, feature showcase, before/after comparison, closing. Speaker notes included on every slide.

---

## Known gaps / honest notes
- The PPT's feature showcase slide currently covers the original 6 features only — it doesn't yet mention personalized calibration or 30-minute risk prediction, even though both now appear in the diagrams and dashboard screens. Worth a quick update before presenting.
- The stats on the PPT's "Why this matters" slide are illustrative estimates, not sourced from a specific verified report — swap in real MoRTH figures before presenting.
- `/api/fleet/drivers` doesn't return live route/location data, so the dashboard's vehicle table shows vehicle type instead of route when running live.
