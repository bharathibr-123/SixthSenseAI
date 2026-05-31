from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import threading
import pygame
import os
import time
import json
import urllib.request
from datetime import datetime
from gtts import gTTS
from detector import detect_frame
from risk_score import RiskEngine

app    = Flask(__name__)
engine = RiskEngine()

# Shared state
latest_result = {"score":0,"level":"SAFE","alert":False,"message":"",
                 "ear_frames":0,"yawn_count":0,"blink_count":0,"session_time":"00:00"}
latest_data   = {"face_detected":False,"ear":0.0,"mar":0.0,"pitch":0.0,"yaw":0.0}
state_lock    = threading.Lock()
jpeg_lock     = threading.Lock()
jpeg_buffer   = None
_lang         = ["en"]
last_spoken   = ""
is_speaking   = False
snapshots_taken    = 0
last_snapshot_time = 0

ALERT_MESSAGES = {
    "en": {
        "DROWSY":   "DANGER! Driver is falling asleep! Stop the vehicle now!",
        "FATIGUE":  "DANGER! High fatigue detected! Take a break immediately!",
        "WARNING":  "Warning! Signs of fatigue detected. Stay alert.",
        "NO_FACE":  "DANGER! Driver not visible! Eyes on the road!",
        "SAFE_BACK":"Driver is alert. Safe to continue."
    },
    "kn": {
        "DROWSY":   "ಅಪಾಯ! ಚಾಲಕರು ನಿದ್ರಿಸುತ್ತಿದ್ದಾರೆ! ವಾಹನ ನಿಲ್ಲಿಸಿ!",
        "FATIGUE":  "ಅಪಾಯ! ಹೆಚ್ಚಿನ ಆಯಾಸ! ತಕ್ಷಣ ವಿರಾಮ ತೆಗೆದುಕೊಳ್ಳಿ!",
        "WARNING":  "ಎಚ್ಚರಿಕೆ! ಆಯಾಸದ ಲಕ್ಷಣಗಳು. ಜಾಗರೂಕರಾಗಿರಿ.",
        "NO_FACE":  "ಅಪಾಯ! ಚಾಲಕರು ಕಾಣಿಸುತ್ತಿಲ್ಲ! ರಸ್ತೆ ನೋಡಿ!",
        "SAFE_BACK":"ಚಾಲಕರು ಎಚ್ಚರವಾಗಿದ್ದಾರೆ. ಸುರಕ್ಷಿತ."
    },
    "hi": {
        "DROWSY":   "खतरा! चालक सो रहा है! गाड़ी रोकें!",
        "FATIGUE":  "खतरा! अत्यधिक थकान! तुरंत आराम करें!",
        "WARNING":  "चेतावनी! थकान के संकेत. सतर्क रहें।",
        "NO_FACE":  "खतरा! चालक दिख नहीं रहा!",
        "SAFE_BACK":"चालक सतर्क है। सुरक्षित।"
    }
}
LANG_NAMES = {"en":"English","kn":"ಕನ್ನಡ","hi":"हिंदी"}

pygame.mixer.init()

def speak(alert_type):
    global last_spoken, is_speaking
    if is_speaking: return
    lang = _lang[0]
    msg  = ALERT_MESSAGES.get(lang, ALERT_MESSAGES["en"]).get(alert_type, "")
    if not msg or msg == last_spoken: return
    is_speaking = True
    last_spoken = msg
    try:
        tts = gTTS(text=msg, lang=lang)
        tts.save("sounds/alert.mp3")
        pygame.mixer.music.load("sounds/alert.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Audio: {e}")
    finally:
        is_speaking = False

def do_snapshot(frame, score):
    global last_snapshot_time, snapshots_taken
    now = time.time()
    if now - last_snapshot_time < 10: return
    last_snapshot_time  = now
    snapshots_taken    += 1
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn  = f"snapshots/DANGER_{ts}_{score}.jpg"
    snap = frame.copy()
    h, w = snap.shape[:2]
    cv2.rectangle(snap,(0,h-40),(w,h),(0,0,180),-1)
    cv2.putText(snap,f"DANGER|{score}|{datetime.now().strftime('%H:%M:%S')}",
                (8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
    cv2.imwrite(fn, snap)

def processing_loop():
    global latest_result, latest_data, jpeg_buffer, last_spoken
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,    1)
    prev_level = "SAFE"

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame = cv2.flip(frame, 1)
        try:
            data   = detect_frame(frame)
            result = engine.update(data)
        except Exception as e:
            print(f"Detection: {e}")
            time.sleep(0.05)
            continue

        with state_lock:
            latest_result = result
            latest_data   = data

        level = result["level"]

        if level != prev_level:
            atype = None
            if not data["face_detected"]:
                atype = "NO_FACE"
            elif level == "DANGER":
                atype = "DROWSY" if result.get("ear_frames",0)>=5 else "FATIGUE"
            elif level == "WARNING":
                atype = "WARNING"
            elif level == "SAFE" and prev_level != "SAFE":
                atype = "SAFE_BACK"
                last_spoken = ""  # reset so next DANGER fires fresh
                is_speaking = False
            if atype:
                threading.Thread(target=speak, args=(atype,), daemon=True).start()

        if level == "DANGER":
            threading.Thread(target=do_snapshot,
                args=(frame.copy(), result["score"]), daemon=True).start()

        prev_level = level

        col = (0,220,100) if level=="SAFE" else (0,165,255) if level=="WARNING" else (0,0,255)
        h, w = frame.shape[:2]
        cv2.rectangle(frame,(0,0),(w,6),(20,20,20),-1)
        bw = int(result["score"]/100.0*w)
        cv2.rectangle(frame,(0,0),(bw,6),col,-1)
        dot = (0,220,100) if data["face_detected"] else (0,0,255)
        cv2.circle(frame,(w-18,20),8,dot,-1)
        if level == "DANGER":
            cv2.rectangle(frame,(0,0),(w-1,h-1),(0,0,200),4)

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,70])
        if ok:
            with jpeg_lock:
                jpeg_buffer = buf.tobytes()
        time.sleep(0.04)

proc_thread = threading.Thread(target=processing_loop, daemon=True)
proc_thread.start()

def generate_frames():
    while True:
        with jpeg_lock:
            fb = jpeg_buffer
        if fb:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + fb + b'\r\n')
        time.sleep(0.04)

@app.route('/')
def index():
    return render_template('index.html', current_lang=_lang[0], lang_names=LANG_NAMES)

@app.route('/video_feed')
def video_feed():
    r = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    r.headers['Cache-Control'] = 'no-cache'
    return r

@app.route('/stats')
def stats():
    with state_lock:
        return jsonify({
            "score":        latest_result["score"],
            "level":        latest_result["level"],
            "alert":        latest_result["alert"],
            "message":      latest_result["message"],
            "ear":          latest_data["ear"],
            "mar":          latest_data["mar"],
            "yawn_count":   latest_result["yawn_count"],
            "blink_count":  latest_result["blink_count"],
            "session_time": latest_result["session_time"],
            "face":         latest_data["face_detected"],
            "lang":         _lang[0],
            "snapshots":    snapshots_taken
        })

@app.route('/set_lang/<lang>')
def set_lang(lang):
    global last_spoken, is_speaking
    if lang in ALERT_MESSAGES:
        _lang[0]    = lang
        last_spoken = ""
        is_speaking = False
        print(f"Lang: {LANG_NAMES[lang]}")
    return jsonify({"lang":_lang[0],"name":LANG_NAMES[_lang[0]]})

@app.route('/stress_check')
def stress_check():
    with state_lock:
        pitch = abs(latest_data.get("pitch", 0))
        yaw   = abs(latest_data.get("yaw",   0))
        score = latest_result.get("score",   0)
    stress = 0
    if pitch > 15: stress += 30
    if yaw   > 20: stress += 30
    if score > 60: stress += 40
    stress = min(100, stress)
    level  = "HIGH" if stress > 60 else "MEDIUM" if stress > 30 else "LOW"
    return jsonify({"stress_score": stress, "level": level})

@app.route('/weather')
def weather():
    try:
        loc  = json.loads(urllib.request.urlopen("http://ip-api.com/json/", timeout=3).read())
        city = loc.get("city","Mysuru")
        lat  = loc.get("lat", 12.29)
        lon  = loc.get("lon", 76.64)
        w    = json.loads(urllib.request.urlopen(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true",
            timeout=5).read())
        cur   = w.get("current_weather",{})
        temp  = cur.get("temperature",0)
        wind  = cur.get("windspeed",0)
        wcode = cur.get("weathercode",0)
        if wcode in [51,53,55,61,63,65,80,81,82]:
            cond,warn,alrt = "🌧️ Rainy","Rain detected! Reduce speed.",True
        elif wcode in [71,73,75,77]:
            cond,warn,alrt = "🌫️ Foggy","Low visibility! Drive slowly.",True
        elif wind > 40:
            cond,warn,alrt = "💨 Windy","Strong winds! Hold steering firmly.",True
        else:
            cond,warn,alrt = "☀️ Clear","Weather is clear. Drive safely!",False
        return jsonify({"city":city,"temp":temp,"condition":cond,"warning":warn,"alert":alrt})
    except Exception as e:
        return jsonify({"city":"--","temp":"--","condition":"--","warning":"Could not fetch weather.","alert":False})

@app.route('/snapshots_list')
def snapshots_list():
    files = []
    if os.path.exists('snapshots'):
        files = sorted([f for f in os.listdir('snapshots') if f.endswith('.jpg')], reverse=True)
    return jsonify({'snapshots': files, 'count': len(files)})

@app.route('/snapshots/<filename>')
def get_snapshot(filename):
    path = os.path.join('snapshots', filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return 'Not found', 404

@app.route('/reset')
def reset():
    global last_spoken, snapshots_taken, is_speaking
    engine.reset()
    last_spoken     = ""
    is_speaking     = False
    snapshots_taken = 0
    return jsonify({"status":"reset"})

if __name__ == '__main__':
    os.makedirs("snapshots", exist_ok=True)
    print("="*50)
    print("  SixthSense AI -- http://127.0.0.1:5000")
    print("="*50)
    try:
        from waitress import serve
        serve(app, host="127.0.0.1", port=5000, threads=8)
    except ImportError:
        app.run(debug=False, threaded=True, use_reloader=False)