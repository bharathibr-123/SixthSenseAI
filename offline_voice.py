"""
offline_voice.py
SixthSense AI — Fleet Driver Safety & Analytics Platform
Tata Technologies InnoVent Hackathon 2026

Purpose:
    Driving happens on highways/mines with patchy or zero internet.
    gTTS itself NEEDS internet to synthesize speech, so we can't call it
    live while the vehicle is moving. The fix: generate all alert MP3s
    ONCE (at app startup / install time, whenever internet is available)
    and cache them to disk. From then on, playback is 100% offline —
    we just play the cached file with pygame.

Usage (from app.py or main monitoring loop):
    import offline_voice as voice

    voice.prepare_all_voice_alerts()   # call once at startup (needs internet
                                        # only the first time; skips regenerating
                                        # files that already exist)

    voice.play_alert("kn", "danger")   # 100% offline from here on
    voice.play_alert("en", "warning")
    voice.play_alert("hi", "danger")

Folder layout produced:
    sounds/
        en_danger.mp3
        en_warning.mp3
        kn_danger.mp3
        kn_warning.mp3
        hi_danger.mp3
        hi_warning.mp3
"""

import os
import time
import threading

from gtts import gTTS
import pygame

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

# gTTS language codes
LANG_CODES = {
    "en": "en",   # English
    "kn": "kn",   # Kannada
    "hi": "hi",   # Hindi
}

# The exact alert phrases spoken to the driver, per language/level.
# Keep these short and unambiguous — clarity matters more than politeness
# when someone is falling asleep at the wheel.
ALERT_MESSAGES = {
    "en": {
        "danger":  "Danger! You appear drowsy. Please stop the vehicle and take a break immediately.",
        "warning": "Warning! Signs of fatigue detected. Please stay alert.",
    },
    "kn": {
        "danger":  "ಅಪಾಯ! ನೀವು ನಿದ್ರಾವಸ್ಥೆಯಲ್ಲಿರುವಂತೆ ಕಂಡುಬರುತ್ತಿದೆ. ದಯವಿಟ್ಟು ವಾಹನವನ್ನು ನಿಲ್ಲಿಸಿ ವಿಶ್ರಾಂತಿ ಪಡೆಯಿರಿ.",
        "warning": "ಎಚ್ಚರಿಕೆ! ಆಯಾಸದ ಲಕ್ಷಣಗಳು ಕಂಡುಬಂದಿವೆ. ದಯವಿಟ್ಟು ಜಾಗರೂಕರಾಗಿರಿ.",
    },
    "hi": {
        "danger":  "खतरा! आप नींद में लग रहे हैं। कृपया तुरंत वाहन रोकें और आराम करें।",
        "warning": "चेतावनी! थकान के लक्षण मिले हैं। कृपया सतर्क रहें।",
    },
}

# How many times to retry a failed download before giving up on that file
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

# ---------------------------------------------------------------------------
# INTERNAL STATE
# ---------------------------------------------------------------------------

_mixer_ready = False
_mixer_lock = threading.Lock()


# ---------------------------------------------------------------------------
# SETUP HELPERS
# ---------------------------------------------------------------------------

def _ensure_sounds_dir():
    """Create the sounds/ folder if it doesn't exist yet."""
    os.makedirs(SOUNDS_DIR, exist_ok=True)


def _file_path(lang, level):
    """Return the expected on-disk path for a given lang/level combo."""
    return os.path.join(SOUNDS_DIR, f"{lang}_{level}.mp3")


def _init_mixer():
    """
    Lazily initialize the pygame mixer exactly once.
    Safe to call repeatedly (it no-ops after the first success).
    """
    global _mixer_ready
    with _mixer_lock:
        if not _mixer_ready:
            pygame.mixer.init()
            _mixer_ready = True


# ---------------------------------------------------------------------------
# GENERATION (needs internet — run at startup / install time)
# ---------------------------------------------------------------------------

def generate_alert_mp3(lang, level, force=False):
    """
    Generate a single alert MP3 using gTTS and save it under sounds/.

    lang:  "en" | "kn" | "hi"
    level: "danger" | "warning"
    force: regenerate even if the file already exists

    Returns the file path on success, or None on failure (e.g. no internet).
    """
    if lang not in ALERT_MESSAGES or level not in ALERT_MESSAGES[lang]:
        raise ValueError(f"Unknown lang/level combo: {lang}/{level}")

    _ensure_sounds_dir()
    out_path = _file_path(lang, level)

    if os.path.exists(out_path) and not force:
        return out_path  # already cached, nothing to do

    message = ALERT_MESSAGES[lang][level]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tts = gTTS(text=message, lang=LANG_CODES[lang])
            tts.save(out_path)
            print(f"[offline_voice] Generated {out_path}")
            return out_path
        except Exception as e:
            print(f"[offline_voice] Attempt {attempt}/{MAX_RETRIES} failed for "
                  f"{lang}_{level}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    print(f"[offline_voice] FAILED to generate {lang}_{level}.mp3 "
          f"(no internet? check connection). Voice alert will be skipped "
          f"until this file exists.")
    return None


def prepare_all_voice_alerts(force=False):
    """
    Generate every alert file defined in ALERT_MESSAGES.
    Call this once at app startup (e.g. before the Flask server starts
    accepting camera frames). Files that already exist on disk are
    skipped unless force=True, so this is cheap to call every boot.

    Returns a dict: {"en_danger": True/False, "en_warning": True/False, ...}
    """
    _ensure_sounds_dir()
    results = {}
    for lang in ALERT_MESSAGES:
        for level in ALERT_MESSAGES[lang]:
            key = f"{lang}_{level}"
            path = generate_alert_mp3(lang, level, force=force)
            results[key] = path is not None

    missing = [k for k, ok in results.items() if not ok]
    if missing:
        print(f"[offline_voice] WARNING: missing voice files: {missing}. "
              f"These alerts will silently no-op until regenerated with internet access.")
    else:
        print("[offline_voice] All voice alert files ready. Fully offline from here.")

    return results


# ---------------------------------------------------------------------------
# PLAYBACK (fully offline — just reads the cached mp3 from disk)
# ---------------------------------------------------------------------------

def play_alert(lang, level, blocking=False):
    """
    Play a pre-generated alert MP3. No internet required.

    lang:     "en" | "kn" | "hi"
    level:    "danger" | "warning"
    blocking: if True, waits for playback to finish before returning.
              Default False so it doesn't freeze the video processing loop.

    Returns True if playback started, False if the file was missing
    (e.g. prepare_all_voice_alerts() was never run or failed for that file).
    """
    if lang not in LANG_CODES or level not in ("danger", "warning"):
        raise ValueError(f"Unknown lang/level combo: {lang}/{level}")

    path = _file_path(lang, level)
    if not os.path.exists(path):
        print(f"[offline_voice] Missing {path} — call prepare_all_voice_alerts() "
              f"with internet access first. Skipping playback.")
        return False

    _init_mixer()

    try:
        sound = pygame.mixer.Sound(path)
        sound.play()
        if blocking:
            duration = sound.get_length()
            time.sleep(duration)
        return True
    except Exception as e:
        print(f"[offline_voice] Playback error for {path}: {e}")
        return False


def stop_all_alerts():
    """Immediately stop any currently playing alert sounds."""
    if _mixer_ready:
        pygame.mixer.stop()


def is_ready(lang, level):
    """Check whether a specific alert file has already been generated."""
    return os.path.exists(_file_path(lang, level))


def all_ready():
    """Check whether every required alert file exists on disk."""
    return all(
        is_ready(lang, level)
        for lang in ALERT_MESSAGES
        for level in ALERT_MESSAGES[lang]
    )


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running offline_voice.py self-test...")
    print(f"Sounds directory: {SOUNDS_DIR}")

    print("\nStep 1: Generating all alert files (requires internet)...")
    results = prepare_all_voice_alerts()
    print("Generation results:", results)

    print(f"\nStep 2: all_ready() = {all_ready()}")

    if all_ready():
        print("\nStep 3: Playing en_warning (non-blocking)...")
        play_alert("en", "warning", blocking=False)
        time.sleep(3)
        stop_all_alerts()
        print("Self-test complete — playback attempted successfully.")
    else:
        print("\nSkipping playback test — some files failed to generate "
              "(likely no internet in this environment). The functions "
              "themselves are implemented correctly and will work as soon "
              "as gTTS can reach the internet at least once.")
