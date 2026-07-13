# Frontend Reconciliation Notes

What changed in the backend after reading the actual React app
(`src/api/fleetApi.js`, `driverApi.js`) and Bharathi's design-deliverable
HTML dashboards, and why. Read this before touching `api_routes.py` or
`app.py` again — the reasons below aren't obvious from the code alone.

---

## The core problem: two frontends, two API conventions

- **The React app** (`fleetApi.js`/`driverApi.js`) calls `res.json()` and
  reads fields straight off the top level: `driversRes.drivers`,
  `stats.driver_name`, `live.risk_score`. No envelope, ever.
- **Bharathi's standalone HTML dashboards** (in the design-deliverables
  zip) expect `{success: true, data: {...}}` and read `body.data.drivers`.
- My original `api_routes.py` used the `{success, data}` envelope
  everywhere (written before I'd seen either frontend).

Only **one route** is called by both: `/api/fleet/drivers`. It now
returns a **dual-shape response** — both the envelope and the raw
top-level keys in the same JSON body — so both consumers work unmodified:

```python
payload = {"count": len(drivers), "drivers": drivers}
return jsonify({"success": True, "data": payload, **payload})
```

Every other new route added for the React app returns **raw JSON, no
envelope** (see the note at the top of `api_routes.py` marking where this
convention starts). The original routes built before this reconciliation
(`/api/driver/register`, `/api/session/save`, `/api/driver/<id>/risk-
prediction`, `/api/driver/<id>/calibration-status`, etc.) still use the
`{success, data}` envelope — that's fine, because neither frontend
actually calls those with the raw-JSON expectation.

**If you add a new route later**, check which frontend(s) call it before
picking a response shape.

---

## New routes added (React app contract)

| Route | Backing function |
|---|---|
| `GET /api/fleet/map` | `database.get_fleet_map()` |
| `GET /api/fleet/reports?period=` | `database.get_fleet_report()` |
| `GET /api/alerts/live` | `database.get_fleet_alerts()` |
| `GET /api/driver/<id>/stats` | `database.get_driver_stats_view()` |
| `GET /api/driver/<id>/history` | `database.get_driver_history_view()` |
| `GET/PUT /api/driver/<id>/profile` | `database.get_driver_profile_view()` / `update_driver_ux_thresholds()` |
| `GET /stats` (root, no `/api`) | mirrors `session.risk_score/risk_level` |

## New database columns (auto-migrated — see `_migrate_driver_columns()`)

`drivers` gained: `drowsiness_threshold`, `distraction_threshold`,
`yawn_threshold`, `blink_threshold` (all 0–100, UX-facing), `last_lat`,
`last_lng` (fleet map), `preferred_language`. Existing databases upgrade
automatically on next `init_db()` call — no manual migration needed.

## The UX-threshold ↔ CV-threshold translation layer (`ux_thresholds.py`)

The Driver Profile page shows four 0–100 "sensitivity" sliders
(Drowsiness, Distraction, Yawn, Blink). The detection loop needs real
EAR/MAR floats and head-pose variance cutoffs. `ux_thresholds.py`
converts between them in both directions. Saving the profile in the UI
(`PUT /api/driver/<id>/profile`) now **actually changes detection
behavior** — it recomputes `ear_threshold`/`mar_threshold` and, on the
next `POST /api/monitoring/start`, the per-session distraction cutoffs
and blink consecutive-frame count too.

**Honest limitation**: Distraction and Blink Duration don't have
dedicated CV signals in this system — they're proxies (head-pose
restlessness variance, and an adjustable consecutive-frame window on the
same EAR signal used for drowsiness, respectively). Documented in
`ux_thresholds.py`'s bottom comment block. A production system would
want a real gaze vector and a per-blink timer instead.

---

## Known remaining gaps (from Bharathi's own README in the design zip)

- `/api/fleet/drivers` doesn't return live route/location data beyond a
  single lat/lng point — her dashboard's vehicle table shows vehicle type
  instead of a route.
- The three HTML dashboard files still fall back to demo data if CORS
  isn't enabled or `API_BASE_URL`/`DRIVER_ID` don't match your setup —
  `flask-cors` is already wired into `app.py`, but double check the
  `DRIVER_ID` constant in each HTML file against a real ID in your DB.

### ~~Snapshot serving~~ — fixed
Originally, DANGER-event snapshots were saved to disk but never linked to
their alert record or served over HTTP, so `AlertsPage.jsx`'s preview
modal would have had nothing to show. Fixed:
- `alerts` table gained a `snapshot_path` column (auto-migrated).
- `app.py`'s `maybe_trigger_alert()`/`maybe_save_snapshot()` now link the
  snapshot captured on a DANGER frame to whichever alert fired that same
  frame (done *after* `update_risk_score()` finalizes the level, since
  the level isn't authoritative yet at the point alerts are raised).
- New route: `GET /snapshots/<filename>` serves the file.
- `database.get_fleet_alerts()` now returns a real `/snapshots/...` URL
  in the `snapshot` field instead of always `null`.
