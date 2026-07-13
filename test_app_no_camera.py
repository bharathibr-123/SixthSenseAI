"""
test_app_no_camera.py
Exercises everything in app.py that does NOT require a physical webcam:
  - Module import / startup wiring
  - Risk score engine math
  - EAR/MAR computation against synthetic landmark data
  - All non-video Flask routes via test client
This does NOT test gen_frames()/cv2.VideoCapture — that needs real hardware.
"""
import os
import sys

# Use throwaway DB/model paths so this doesn't touch real project data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database as db
db.DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_app_sixthsense.db")
if os.path.exists(db.DB_PATH):
    os.remove(db.DB_PATH)

import risk_predictor as rp
rp.MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "test_app_rf.joblib")

import app as sixthsense_app  # this runs app.py's module-level init

print("\n=== Step 1: Module import / startup completed without crashing ===")

client = sixthsense_app.app.test_client()

def show(label, resp):
    print(f"{label}: [{resp.status_code}] {resp.get_json()}")

print("\n=== Step 2: Health check ===")
show("GET /api/health", client.get("/api/health"))

print("\n=== Step 3: Register a driver (via api_routes blueprint) ===")
resp = client.post("/api/driver/register", json={
    "driver_name": "Test Driver", "age": 35, "vehicle_type": "truck",
    "vehicle_number": "KA-05-TEST-001"
})
show("POST /api/driver/register", resp)
driver_id = resp.get_json()["data"]["driver_id"]

print("\n=== Step 4: Start monitoring session ===")
show("POST /api/monitoring/start", client.post("/api/monitoring/start", json={"driver_id": driver_id}))

print("\n=== Step 5: Push GPS ===")
show("POST /api/monitoring/gps", client.post("/api/monitoring/gps", json={"lat": 12.9716, "lng": 77.5946}))

print("\n=== Step 6: Simulate risk score engine directly (no camera) ===")
s = sixthsense_app.session
# Simulate 20 frames of eyes-closed to cross EAR_CONSEC_FRAMES threshold
for i in range(25):
    score, level = sixthsense_app.update_risk_score(
        eyes_closed=True, yawning=False, no_face=False, stress_detected=False)
print(f"  After 25 drowsy frames -> score={score}, level={level}")
assert level == "DANGER", f"Expected DANGER, got {level}"

for i in range(40):
    score, level = sixthsense_app.update_risk_score(
        eyes_closed=False, yawning=False, no_face=False, stress_detected=False)
print(f"  After 40 recovery frames -> score={score}, level={level}")
assert level == "SAFE", f"Expected SAFE after recovery, got {level}"

print("\n=== Step 7: EAR/MAR math against synthetic landmarks ===")
class FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Build a fake "open eye" landmark set: indices used are
# LEFT_EYE = [362, 385, 387, 263, 373, 380]
w, h = 640, 480
fake_landmarks = {}
# open eye shape: horizontal corners far apart, vertical pairs moderately apart
coords = {
    362: (0.40, 0.40), 385: (0.42, 0.38), 387: (0.44, 0.38),
    263: (0.46, 0.40), 373: (0.44, 0.42), 380: (0.42, 0.42),
}
for idx, (x, y) in coords.items():
    fake_landmarks[idx] = FakeLandmark(x, y)

class LandmarkList:
    def __init__(self, d):
        self.d = d
    def __getitem__(self, i):
        return self.d[i]

ear_open = sixthsense_app.compute_ear(LandmarkList(fake_landmarks), sixthsense_app.LEFT_EYE, w, h)
print(f"  Open-eye EAR: {ear_open:.3f} (expect moderate, > 0.15)")
assert ear_open > 0.1

# closed eye: vertical points collapse toward each other
coords_closed = {
    362: (0.40, 0.40), 385: (0.42, 0.399), 387: (0.44, 0.399),
    263: (0.46, 0.40), 373: (0.44, 0.401), 380: (0.42, 0.401),
}
fake_landmarks_closed = {idx: FakeLandmark(x, y) for idx, (x, y) in coords_closed.items()}
ear_closed = sixthsense_app.compute_ear(LandmarkList(fake_landmarks_closed), sixthsense_app.LEFT_EYE, w, h)
print(f"  Closed-eye EAR: {ear_closed:.4f} (expect near 0)")
assert ear_closed < ear_open

print("\n=== Step 8: Weather + Nearby (external APIs — may fail if sandboxed network blocks them) ===")
show("GET /api/weather", client.get("/api/weather?lat=12.9716&lng=77.5946"))
show("GET /api/nearby", client.get("/api/nearby?lat=12.9716&lng=77.5946&category=hospital"))

print("\n=== Step 9: Monitoring status ===")
show("GET /api/monitoring/status", client.get("/api/monitoring/status"))

print("\n=== Step 10: Stop monitoring session (persists to DB, runs calibration + risk prediction) ===")
show("POST /api/monitoring/stop", client.post("/api/monitoring/stop"))

print("\n=== Step 11: Confirm session was actually persisted ===")
show("GET /api/driver/<id>/sessions", client.get(f"/api/driver/{driver_id}/sessions"))

print("\n=== Step 12: Emergency WhatsApp (should no-op cleanly, no Twilio configured) ===")
result = sixthsense_app.trigger_emergency_whatsapp(reason="TEST")
print(f"  trigger_emergency_whatsapp() returned: {result} (expect False — no credentials set)")
assert result is False

print("\n=== Step 13: React frontend contract routes (fleetApi.js / driverApi.js) ===")

# --- /api/fleet/drivers must satisfy BOTH consumers at once ---
resp = client.get("/api/fleet/drivers")
body = resp.get_json()
show("GET /api/fleet/drivers", resp)
assert "drivers" in body, "React app needs top-level 'drivers'"
assert body.get("success") is True and "drivers" in body.get("data", {}), \
    "Bharathi's HTML dashboards need {success, data: {drivers}}"
print("  OK: dual-shape response satisfies both the React app AND the HTML dashboards")

# --- /api/fleet/map ---
resp = client.get("/api/fleet/map")
show("GET /api/fleet/map", resp)
assert "vehicles" in resp.get_json()

# --- /api/fleet/reports ---
resp = client.get("/api/fleet/reports?period=weekly")
body = resp.get_json()
show("GET /api/fleet/reports", resp)
assert "drivers" in body and "trend" in body and "period" in body

# --- /api/alerts/live ---
resp = client.get("/api/alerts/live")
body = resp.get_json()
show("GET /api/alerts/live", resp)
assert "alerts" in body
if body["alerts"]:
    a = body["alerts"][0]
    assert "driver_name" in a and "vehicle_number" in a and "risk_level" in a and "timestamp" in a
    print(f"  Sample alert shape OK: {list(a.keys())}")

# --- /api/driver/<id>/stats ---
resp = client.get(f"/api/driver/{driver_id}/stats")
body = resp.get_json()
show("GET /api/driver/<id>/stats", resp)
assert "driver_name" in body and "session_duration" in body and "risk_level" in body
assert body["risk_level"] == body["risk_level"].lower(), "risk_level must be lowercase for frontend"

# --- /api/driver/<id>/history ---
resp = client.get(f"/api/driver/{driver_id}/history")
body = resp.get_json()
show("GET /api/driver/<id>/history", resp)
assert isinstance(body, list)
if body:
    assert "risk_level" in body[0] and "session_duration" in body[0]

# --- /api/driver/<id>/profile GET ---
resp = client.get(f"/api/driver/{driver_id}/profile")
body = resp.get_json()
show("GET /api/driver/<id>/profile", resp)
assert "thresholds" in body
assert set(body["thresholds"].keys()) == {
    "drowsiness_threshold", "distraction_threshold", "yawn_threshold", "blink_threshold"}

# --- /api/driver/<id>/profile PUT ---
resp = client.put(f"/api/driver/{driver_id}/profile", json={
    "driver_name": "Test Driver Updated",
    "thresholds": {"drowsiness_threshold": 20, "yawn_threshold": 80,
                   "distraction_threshold": 30, "blink_threshold": 10},
})
body = resp.get_json()
show("PUT /api/driver/<id>/profile", resp)
assert body["driver_name"] == "Test Driver Updated"
assert body["thresholds"]["drowsiness_threshold"] == 20

# Confirm the PUT actually changed the underlying EAR/MAR CV thresholds too
updated_driver = db.get_driver(driver_id)
print(f"  Underlying CV thresholds after profile save: "
      f"ear={updated_driver['ear_threshold']}, mar={updated_driver['mar_threshold']}")
assert updated_driver["ear_threshold"] != 0.25, "drowsiness slider should have changed ear_threshold"

print("\n=== Step 14: Root-level /stats (driverApi.js contract) ===")
resp = client.get("/stats")
body = resp.get_json()
show("GET /stats", resp)
assert "risk_score" in body and "risk_level" in body
assert "success" not in body, "/stats must be raw, no envelope"

print("\n=== Step 15: GPS persists to the driver record (for the fleet map) ===")
client.post("/api/monitoring/start", json={"driver_id": driver_id})
client.post("/api/monitoring/gps", json={"lat": 13.05, "lng": 77.6})
d = db.get_driver(driver_id)
print(f"  Driver location after GPS push: lat={d['last_lat']}, lng={d['last_lng']}")
assert d["last_lat"] == 13.05 and d["last_lng"] == 77.6
client.post("/api/monitoring/stop")

print("\n=== Step 16: ux_thresholds.py conversion sanity (imported into app.py + api_routes.py) ===")
import ux_thresholds as ux
assert sixthsense_app.ux is ux
print("  OK: app.py shares the same ux_thresholds module")

print("\nAll non-camera tests passed, including full frontend-contract reconciliation.")
