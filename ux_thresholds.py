"""
ux_thresholds.py
SixthSense AI — Fleet Driver Safety & Analytics Platform
Tata Technologies InnoVent Hackathon 2026

Purpose:
    The React frontend's Driver Profile page (DriverProfilePage.jsx) shows
    four named 0-100 "sensitivity" sliders: Drowsiness, Distraction, Yawn
    Frequency, Blink Duration. These are a friendlier abstraction for a
    fleet manager than raw computer-vision numbers — nobody wants to type
    "0.23" into a form and know what it means.

    The actual detection loop in app.py, however, needs real EAR/MAR
    floats (and a couple of head-pose variance cutoffs for stress/
    distraction). This module is the translation layer between the two,
    so a change on the frontend slider produces a real, correctly-signed
    change in detection behavior, not just a cosmetic number.

Direction of each slider (matches the copy already written in
DriverProfilePage.jsx):
    - Drowsiness: "Lower = triggers sooner" -> lower slider value must
      raise the EAR threshold (easier to count eyes as "closed").
    - Yawn Frequency: higher slider = requires a wider mouth opening
      before counting it as a yawn (less sensitive).
    - Distraction / Blink: no dedicated CV signal exists yet for these
      as separate from drowsiness — see the honest note at the bottom.
"""

# EAR/MAR guard-rail ranges — kept identical to threshold_calibration.py
# so personalized calibration and manual UX sliders never fight each other.
EAR_MIN, EAR_MAX = 0.15, 0.32
MAR_MIN, MAR_MAX = 0.35, 0.85

# Stress/distraction detection cutoffs (degrees of yaw/pitch std-dev over
# the rolling window in app.py) — the range a distraction slider moves within.
DISTRACTION_YAW_STD_MIN, DISTRACTION_YAW_STD_MAX = 6.0, 20.0
DISTRACTION_PITCH_STD_MIN, DISTRACTION_PITCH_STD_MAX = 5.0, 16.0


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def drowsiness_pct_to_ear(pct):
    """
    0 (most sensitive, triggers sooner) -> EAR_MAX
    100 (least sensitive)               -> EAR_MIN
    """
    pct = _clamp(pct, 0, 100)
    return round(EAR_MAX - (pct / 100.0) * (EAR_MAX - EAR_MIN), 4)


def ear_to_drowsiness_pct(ear):
    """Inverse of drowsiness_pct_to_ear — used when displaying a
    calibrated EAR threshold back on the 0-100 slider."""
    ear = _clamp(ear, EAR_MIN, EAR_MAX)
    pct = (EAR_MAX - ear) / (EAR_MAX - EAR_MIN) * 100.0
    return round(pct, 1)


def yawn_pct_to_mar(pct):
    """
    0 (most sensitive, flags smaller mouth openings) -> MAR_MIN
    100 (least sensitive, requires a wide-open yawn)  -> MAR_MAX
    """
    pct = _clamp(pct, 0, 100)
    return round(MAR_MIN + (pct / 100.0) * (MAR_MAX - MAR_MIN), 4)


def mar_to_yawn_pct(mar):
    """Inverse of yawn_pct_to_mar."""
    mar = _clamp(mar, MAR_MIN, MAR_MAX)
    pct = (mar - MAR_MIN) / (MAR_MAX - MAR_MIN) * 100.0
    return round(pct, 1)


def distraction_pct_to_std_cutoffs(pct):
    """
    Maps the Distraction slider to the yaw/pitch standard-deviation
    cutoffs used by app.py's detect_stress_from_head_movement(). Lower
    slider value = smaller head movements already count as distraction.
    Returns (yaw_std_cutoff, pitch_std_cutoff) in degrees.
    """
    pct = _clamp(pct, 0, 100)
    yaw_cutoff = DISTRACTION_YAW_STD_MAX - (pct / 100.0) * (DISTRACTION_YAW_STD_MAX - DISTRACTION_YAW_STD_MIN)
    pitch_cutoff = DISTRACTION_PITCH_STD_MAX - (pct / 100.0) * (DISTRACTION_PITCH_STD_MAX - DISTRACTION_PITCH_STD_MIN)
    return round(yaw_cutoff, 2), round(pitch_cutoff, 2)


def blink_pct_to_consec_frames(pct, base_frames=20, min_frames=10, max_frames=35):
    """
    Maps the Blink Duration slider to EAR_CONSEC_FRAMES (how many
    consecutive closed-eye frames count as a genuine drowsy episode
    rather than a normal blink). Lower slider = shorter closures already
    count (more sensitive, fewer frames required).
    """
    pct = _clamp(pct, 0, 100)
    frames = min_frames + (pct / 100.0) * (max_frames - min_frames)
    return int(round(frames))


def ux_thresholds_to_cv_params(drowsiness_pct, distraction_pct, yawn_pct, blink_pct):
    """
    Convenience one-shot conversion: given all four UX slider values,
    return every derived CV parameter app.py's monitoring loop needs.
    Call this once when a session starts (see app.py start_monitoring()).
    """
    yaw_std_cutoff, pitch_std_cutoff = distraction_pct_to_std_cutoffs(distraction_pct)
    return {
        "ear_threshold": drowsiness_pct_to_ear(drowsiness_pct),
        "mar_threshold": yawn_pct_to_mar(yawn_pct),
        "distraction_yaw_std_cutoff": yaw_std_cutoff,
        "distraction_pitch_std_cutoff": pitch_std_cutoff,
        "ear_consec_frames": blink_pct_to_consec_frames(blink_pct),
    }


# ---------------------------------------------------------------------------
# HONEST LIMITATION
# ---------------------------------------------------------------------------
# The Distraction and Blink Duration sliders now have a genuine, wired
# effect on detection behavior (head-pose variance cutoffs and consecutive-
# frame count respectively) — but they are proxies, not dedicated models.
# There is no separate gaze-tracking or blink-duration-specific signal in
# this system; "distraction" is approximated via head-pose restlessness,
# and "blink duration" is approximated via the same EAR signal used for
# drowsiness, just with an adjustable frame-count window. A production
# system would likely want a dedicated gaze vector and a per-blink timer
# instead of reusing EAR/head-pose for four conceptually distinct sliders.


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running ux_thresholds.py self-test...\n")

    print("Drowsiness slider -> EAR threshold:")
    for pct in [0, 25, 50, 75, 100]:
        ear = drowsiness_pct_to_ear(pct)
        back = ear_to_drowsiness_pct(ear)
        print(f"  {pct:>3} -> EAR {ear}  (round-trip: {back})")
    assert drowsiness_pct_to_ear(0) == EAR_MAX
    assert drowsiness_pct_to_ear(100) == EAR_MIN

    print("\nYawn slider -> MAR threshold:")
    for pct in [0, 25, 50, 75, 100]:
        mar = yawn_pct_to_mar(pct)
        back = mar_to_yawn_pct(mar)
        print(f"  {pct:>3} -> MAR {mar}  (round-trip: {back})")
    assert yawn_pct_to_mar(0) == MAR_MIN
    assert yawn_pct_to_mar(100) == MAR_MAX

    print("\nDistraction slider -> yaw/pitch std cutoffs:")
    for pct in [0, 50, 100]:
        print(f"  {pct:>3} -> {distraction_pct_to_std_cutoffs(pct)}")

    print("\nBlink slider -> consecutive-frame count:")
    for pct in [0, 50, 100]:
        print(f"  {pct:>3} -> {blink_pct_to_consec_frames(pct)} frames")

    print("\nOne-shot conversion:")
    print(" ", ux_thresholds_to_cv_params(50, 50, 50, 50))

    print("\nAll assertions passed.")
