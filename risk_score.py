import time
import numpy as np

# ─────────────────────────────────────────
#  THRESHOLDS
# ─────────────────────────────────────────
EAR_THRESHOLD   = 0.28   # Below this = eye closing
MAR_THRESHOLD   = 0.50   # Above this = yawning
PITCH_THRESHOLD = 20.0   # Degrees - head nodding
YAW_THRESHOLD   = 25.0   # Degrees - head turning

EAR_CONSEC_FRAMES = 5    # frames before counting drowsy
MAR_CONSEC_FRAMES = 8    # frames before counting yawn

# ─────────────────────────────────────────
#  RISK ENGINE
# ─────────────────────────────────────────
class RiskEngine:
    def __init__(self):
        self.ear_counter    = 0
        self.mar_counter    = 0
        self.blink_count    = 0
        self.yawn_count     = 0
        self.distract_count = 0

        self.session_start   = time.time()
        self.last_alert_time = 0
        self.alert_cooldown  = 5   # seconds between alerts

        self.score_history  = []
        self.history_size   = 6   # smaller = faster response
        self.current_score  = 0
        self.current_level  = "SAFE"
        self.alert_needed   = False
        self.alert_message  = ""

    # ─────────────────────────────────────
    #  MAIN UPDATE
    # ─────────────────────────────────────
    def update(self, data):
        self.alert_needed  = False
        self.alert_message = ""

        # ── NO FACE = instant DANGER ──
        if not data["face_detected"]:
            self.distract_count += 1
            # Force score to DANGER immediately
            self.score_history  = [92] * self.history_size
            self.current_score  = 92
            self.current_level  = "DANGER"
            self._check_alert("DISTRACTION",
                "DANGER! Driver not visible! Eyes on the road!")
            return self._result()

        ear   = data["ear"]
        mar   = data["mar"]
        pitch = abs(data["pitch"])
        yaw   = abs(data["yaw"])

        # ── EAR counter ──
        if ear < EAR_THRESHOLD:
            self.ear_counter += 1
        else:
            if self.ear_counter >= EAR_CONSEC_FRAMES:
                self.blink_count += 1
            self.ear_counter = 0

        # ── MAR counter ──
        if mar > MAR_THRESHOLD:
            self.mar_counter += 1
        else:
            if self.mar_counter >= MAR_CONSEC_FRAMES:
                self.yawn_count += 1
            self.mar_counter = 0

        # ── Head pose ──
        head_distracted = (pitch > PITCH_THRESHOLD or yaw > YAW_THRESHOLD)
        if head_distracted:
            self.distract_count += 1

        # ─────────────────────────────────
        #  SCORE CALCULATION (max = 100)
        # ─────────────────────────────────
        score = 0

        # Eye closure → up to 60 points (hits DANGER alone)
        if ear < EAR_THRESHOLD:
            ratio     = self.ear_counter / EAR_CONSEC_FRAMES
            eye_score = min(60, ratio * 75)
            score    += eye_score

        # Yawn current → up to 20 points
        if mar > MAR_THRESHOLD:
            mar_score = min(20, ((mar - MAR_THRESHOLD) / 0.3) * 20)
            score    += mar_score

        # Yawn history → up to 20 points
        score += min(20, self.yawn_count * 7)

        # Head pose → up to 25 points
        if head_distracted:
            if pitch > PITCH_THRESHOLD:
                score += min(15, ((pitch - PITCH_THRESHOLD) / 10) * 15)
            if yaw > YAW_THRESHOLD:
                score += min(10, ((yaw - YAW_THRESHOLD) / 10) * 10)

        score = min(100, max(0, score))

        # Smooth score
        self._update_score(score)

        # ── Level & alerts ──
        if self.current_score >= 70:
            self.current_level = "DANGER"
            if self.ear_counter >= EAR_CONSEC_FRAMES:
                self._check_alert("DROWSY",
                    "DANGER! Driver is falling asleep! Stop the vehicle now!")
            elif head_distracted:
                self._check_alert("DISTRACTION",
                    "DANGER! Driver is distracted! Eyes on the road!")
            else:
                self._check_alert("FATIGUE",
                    "DANGER! High fatigue detected! Take a break immediately!")

        elif self.current_score >= 40:
            self.current_level = "WARNING"
            if self.yawn_count >= 1:
                self._check_alert("YAWN",
                    "Warning! Drowsiness detected. Please take a break soon.")
            elif head_distracted:
                self._check_alert("DISTRACTION",
                    "Warning! Keep your eyes on the road.")
            else:
                self._check_alert("FATIGUE",
                    "Warning! Signs of fatigue detected. Stay alert.")
        else:
            self.current_level = "SAFE"

        return self._result()

    # ─────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────
    def _update_score(self, raw_score):
        self.score_history.append(raw_score)
        if len(self.score_history) > self.history_size:
            self.score_history.pop(0)
        self.current_score = int(np.mean(self.score_history))

    def _check_alert(self, alert_type, message):
        now = time.time()
        if now - self.last_alert_time >= self.alert_cooldown:
            self.alert_needed    = True
            self.alert_message   = message
            self.last_alert_time = now

    def _result(self):
        session_secs = int(time.time() - self.session_start)
        mins = session_secs // 60
        secs = session_secs % 60
        return {
            "score":        self.current_score,
            "level":        self.current_level,
            "alert":        self.alert_needed,
            "message":      self.alert_message,
            "ear_frames":   self.ear_counter,
            "yawn_count":   self.yawn_count,
            "blink_count":  self.blink_count,
            "session_time": f"{mins:02d}:{secs:02d}"
        }

    def reset(self):
        self.__init__()


# ─────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    from detector import detect_frame

    print("Risk Engine Test — Press Q to quit")
    print("Try: close eyes 3s | yawn | cover camera | sit normal")
    cap    = cv2.VideoCapture(0)
    engine = RiskEngine()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        data   = detect_frame(frame)
        result = engine.update(data)

        colors = {
            "SAFE":    (0, 200, 0),
            "WARNING": (0, 165, 255),
            "DANGER":  (0, 0, 255)
        }
        col = colors[result["level"]]

        # Risk bar
        cv2.rectangle(frame, (20, 10), (320, 35), (50,50,50), -1)
        bar_w = int(3.0 * result["score"])
        cv2.rectangle(frame, (20, 10), (20 + bar_w, 35), col, -1)

        cv2.putText(frame, f"RISK: {result['score']}  [{result['level']}]",
                    (20,  65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.putText(frame, f"EAR: {data['ear']}  MAR: {data['mar']}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, f"Yawns: {result['yawn_count']}  Blinks: {result['blink_count']}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, f"Session: {result['session_time']}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2)

        if result["alert"]:
            cv2.rectangle(frame, (0, 190), (frame.shape[1], 240), (0,0,200), -1)
            cv2.putText(frame, f"!! {result['message'][:55]}",
                        (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            print(f"ALERT: {result['message']}")

        cv2.imshow("SixthSense AI - Risk Engine", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()