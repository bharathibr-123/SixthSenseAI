"""
risk_predictor.py
SixthSense AI — Fleet Driver Safety & Analytics Platform
Tata Technologies InnoVent Hackathon 2026

Purpose:
    Predicts the probability that a driver will have an accident-causing
    lapse (micro-sleep, severe distraction, etc.) in the NEXT 30 MINUTES,
    based on their current session behaviour and history.

Model: Random Forest Classifier (scikit-learn)
Features:
    - hour_of_day        (0-23)   — circadian risk (night/post-lunch dip)
    - session_duration_min         — time-on-task fatigue
    - yawn_count                   — yawning frequency this session
    - blink_rate                   — blinks per minute (too low = drowsy,
                                      too high = stressed/strained)
    - past_alerts_count            — alerts already raised this session

Since no real accident-labelled dataset exists yet, we generate a
SYNTHETIC dataset using domain-informed rules (circadian fatigue curves,
time-on-task effects, yawning/blink-rate correlations with drowsiness
research) plus random noise, then train the Random Forest on that.
This gives a reasonable, explainable baseline for the hackathon demo.
Swap in real labelled data (from the `alerts`/`sessions` tables, or
telematics/accident logs) later via train_on_real_data().

Usage:
    import risk_predictor as rp

    rp.load_or_train_model()          # call once at startup

    probability = rp.predict_accident_risk(
        hour_of_day=2,
        session_duration_min=95,
        yawn_count=6,
        blink_rate=8,
        past_alerts_count=3
    )
    # -> float between 0.0 and 1.0
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "risk_predictor_rf.joblib")

FEATURE_NAMES = [
    "hour_of_day",
    "session_duration_min",
    "yawn_count",
    "blink_rate",
    "past_alerts_count",
]

RANDOM_SEED = 42

# Module-level cached model (loaded once, reused across predict calls)
_model = None


# ---------------------------------------------------------------------------
# SYNTHETIC DATASET GENERATION
# ---------------------------------------------------------------------------

def _circadian_risk(hour_of_day):
    """
    Returns a base risk multiplier (0-1) based on time of day, following
    well-documented drowsy-driving research: risk peaks late night/early
    morning (2-5 AM) and dips slightly after lunch (1-3 PM).
    """
    # Late night / early morning window — highest risk
    night_peak = np.exp(-0.5 * ((hour_of_day - 3.5) / 1.8) ** 2)
    # Post-lunch dip — moderate secondary risk
    afternoon_dip = 0.5 * np.exp(-0.5 * ((hour_of_day - 14) / 1.5) ** 2)
    return np.clip(night_peak + afternoon_dip, 0, 1)


def generate_synthetic_dataset(n_samples=6000, random_seed=RANDOM_SEED):
    """
    Build a synthetic, domain-informed training dataset.

    Returns:
        X (pd.DataFrame): features, columns = FEATURE_NAMES
        y (pd.Series): binary label, 1 = high accident risk in next 30 min
    """
    rng = np.random.default_rng(random_seed)

    hour_of_day = rng.integers(0, 24, n_samples)
    session_duration_min = rng.gamma(shape=2.0, scale=45, size=n_samples)   # skewed, mostly 0-180 min
    session_duration_min = np.clip(session_duration_min, 0, 360)

    yawn_count = rng.poisson(lam=1.5, size=n_samples)
    # correlate yawns loosely with session duration (fatigue builds over time)
    yawn_count = yawn_count + (session_duration_min > 120).astype(int) * rng.poisson(1.5, n_samples)

    # Normal blink rate ~12-20/min. Drowsy drivers trend lower (long closures
    # reduce blink frequency); highly stressed/strained drivers trend higher.
    blink_rate = rng.normal(loc=16, scale=5, size=n_samples)
    blink_rate = np.clip(blink_rate, 2, 40)

    past_alerts_count = rng.poisson(lam=0.8, size=n_samples)
    past_alerts_count = past_alerts_count + (session_duration_min > 150).astype(int) * rng.poisson(1.0, n_samples)

    # ---- Compute a continuous "true risk" score from domain rules ----
    circadian = _circadian_risk(hour_of_day)                       # 0-1
    fatigue_time = np.clip(session_duration_min / 240, 0, 1)       # 0-1, saturates at 4h
    yawn_factor = np.clip(yawn_count / 8, 0, 1)                    # 0-1
    # blink deviation from healthy range (16) — both too low and too high add risk
    blink_factor = np.clip(np.abs(blink_rate - 16) / 14, 0, 1)
    alert_factor = np.clip(past_alerts_count / 5, 0, 1)

    true_risk_score = (
        0.30 * circadian +
        0.25 * fatigue_time +
        0.20 * yawn_factor +
        0.15 * blink_factor +
        0.10 * alert_factor
    )

    # Add noise so the relationship isn't perfectly deterministic
    noise = rng.normal(0, 0.08, n_samples)
    true_risk_score = np.clip(true_risk_score + noise, 0, 1)

    # Binary label: sample from a Bernoulli using true_risk_score as probability
    # (this keeps the label probabilistic/realistic rather than a hard threshold)
    y = rng.binomial(1, true_risk_score)

    X = pd.DataFrame({
        "hour_of_day": hour_of_day,
        "session_duration_min": session_duration_min,
        "yawn_count": yawn_count,
        "blink_rate": blink_rate,
        "past_alerts_count": past_alerts_count,
    })

    return X, pd.Series(y, name="high_risk_next_30min")


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_model(n_samples=6000, save=True):
    """
    Train a Random Forest on the synthetic dataset and cache it in memory.
    Prints validation accuracy / ROC-AUC so you can sanity-check quality.

    Returns the trained model.
    """
    global _model

    X, y = generate_synthetic_dataset(n_samples=n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"[risk_predictor] Trained on {len(X_train)} samples, "
          f"validated on {len(X_test)}.")
    print(f"[risk_predictor] Validation accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")

    # Feature importance — useful to show judges the model isn't a black box
    importances = dict(zip(FEATURE_NAMES, clf.feature_importances_.round(3)))
    print(f"[risk_predictor] Feature importances: {importances}")

    _model = clf

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        print(f"[risk_predictor] Model saved to {MODEL_PATH}")

    return clf


def load_or_train_model(force_retrain=False):
    """
    Load the cached model from disk if it exists; otherwise train a fresh
    one (and save it). Call this once at Flask app startup.
    """
    global _model

    if not force_retrain and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        print(f"[risk_predictor] Loaded existing model from {MODEL_PATH}")
    else:
        train_model(save=True)

    return _model


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

def predict_accident_risk(hour_of_day, session_duration_min, yawn_count,
                           blink_rate, past_alerts_count):
    """
    Predict the probability (0.0-1.0) of an accident-risk lapse in the
    next 30 minutes, given current session stats.

    Automatically loads/trains the model on first use if not already loaded.
    """
    global _model
    if _model is None:
        load_or_train_model()

    row = pd.DataFrame([{
        "hour_of_day": hour_of_day,
        "session_duration_min": session_duration_min,
        "yawn_count": yawn_count,
        "blink_rate": blink_rate,
        "past_alerts_count": past_alerts_count,
    }])[FEATURE_NAMES]

    probability = _model.predict_proba(row)[0, 1]
    return float(round(probability, 4))


def predict_accident_risk_from_session(session_dict, hour_of_day=None):
    """
    Convenience wrapper: build the prediction directly from a `sessions`
    table row (as returned by database.get_session /
    database.get_sessions_for_driver), so callers don't need to hand-map
    field names.

    session_dict expected keys: session_duration (seconds), yawn_count,
    blink_count, total_alerts. session_duration is converted to minutes;
    blink_count is converted to an approximate blink_rate (per minute)
    using session_duration.

    hour_of_day: pass the current hour (0-23) explicitly, e.g. from
    datetime.now().hour. Defaults to the current system hour if omitted.
    """
    from datetime import datetime

    if hour_of_day is None:
        hour_of_day = datetime.now().hour

    duration_sec = session_dict.get("session_duration", 0) or 0
    duration_min = max(duration_sec / 60.0, 0.1)  # avoid divide-by-zero

    blink_count = session_dict.get("blink_count", 0) or 0
    blink_rate = blink_count / duration_min

    return predict_accident_risk(
        hour_of_day=hour_of_day,
        session_duration_min=duration_min,
        yawn_count=session_dict.get("yawn_count", 0) or 0,
        blink_rate=blink_rate,
        past_alerts_count=session_dict.get("total_alerts", 0) or 0,
    )


def risk_level_label(probability):
    """
    Convert a raw probability into a human-friendly risk band, matching
    the SAFE / WARNING / DANGER vocabulary already used elsewhere in the app.
    """
    if probability >= 0.66:
        return "DANGER"
    elif probability >= 0.33:
        return "WARNING"
    else:
        return "SAFE"


# ---------------------------------------------------------------------------
# HOOK FOR FUTURE REAL DATA
# ---------------------------------------------------------------------------

def train_on_real_data(csv_path, label_column="high_risk_next_30min", save=True):
    """
    Once real labelled accident/near-miss data is available (e.g. exported
    from the `sessions` + `alerts` tables and manually/automatically
    labelled), retrain on it here instead of synthetic data.

    csv_path must contain columns: FEATURE_NAMES + [label_column]
    """
    global _model

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_NAMES + [label_column] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    X = df[FEATURE_NAMES]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"[risk_predictor] Real-data model — accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}")

    _model = clf
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        print(f"[risk_predictor] Real-data model saved to {MODEL_PATH}")

    return clf


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running risk_predictor.py self-test...\n")

    print("Step 1: Training model on synthetic dataset...")
    train_model(n_samples=6000)

    print("\nStep 2: Sample predictions —")

    scenarios = [
        {"label": "Well-rested, daytime, short session",
         "params": dict(hour_of_day=10, session_duration_min=20, yawn_count=0,
                         blink_rate=16, past_alerts_count=0)},
        {"label": "3 AM, long session, high yawns, low blink rate, past alerts",
         "params": dict(hour_of_day=3, session_duration_min=180, yawn_count=8,
                         blink_rate=6, past_alerts_count=4)},
        {"label": "Post-lunch dip, moderate fatigue",
         "params": dict(hour_of_day=14, session_duration_min=90, yawn_count=3,
                         blink_rate=12, past_alerts_count=1)},
    ]

    for s in scenarios:
        prob = predict_accident_risk(**s["params"])
        level = risk_level_label(prob)
        print(f"  {s['label']}")
        print(f"    -> probability={prob:.3f}  level={level}")

    print("\nStep 3: Testing predict_accident_risk_from_session() wrapper...")
    fake_session = {
        "session_duration": 5400,  # 90 min in seconds
        "yawn_count": 5,
        "blink_count": 540,        # 6/min -> low, drowsy signal
        "total_alerts": 2,
    }
    prob = predict_accident_risk_from_session(fake_session, hour_of_day=2)
    print(f"  From session dict -> probability={prob:.3f}  level={risk_level_label(prob)}")

    print("\nStep 4: Testing load_or_train_model() cache reuse...")
    load_or_train_model()  # should load from disk, not retrain
    print(f"  Model file exists: {os.path.exists(MODEL_PATH)}")

    print("\nSelf-test complete. All core functions executed without errors.")
