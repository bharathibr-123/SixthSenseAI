# SixthSenseAI
# 👁️ SixthSense AI — Intelligent Driver Safety System

> **"Every second matters on the road. SixthSense AI gives drivers the extra sense they need to stay safe."**

---

## 🏆 Overview

**SixthSense AI** is a real-time AI-powered driver monitoring system that uses just a laptop or phone camera to detect drowsiness, distraction, and fatigue — and prevents accidents before they happen.

- ✅ Works on **any laptop camera** — no special hardware
- ✅ Runs on **Intel i5 CPU** — no GPU required
- ✅ Speaks **Kannada, Hindi & English** — built for India
- ✅ **100% free** — no cost, no subscription
- ✅ **22 features** in a single web dashboard

---

## 🚨 The Problem

| Fact | Impact |
|------|--------|
| 1.5 lakh+ road deaths per year in India | National crisis |
| 40%+ highway accidents from drowsiness | Preventable deaths |
| Existing solutions cost ₹50,000–₹5 lakhs | Unaffordable |
| No solution speaks Kannada or Hindi | Not built for India |
| Truck & bus drivers most at risk | No monitoring exists |

**SixthSense AI solves all of this — for free.**

---

## 📺 Demo

**Run locally:**
```bash
python app.py
# Open http://127.0.0.1:5000
```

**What you will see:**
```
1. Camera opens → face detection starts instantly
2. Close eyes 3 seconds → Risk Score jumps to DANGER (70+)
3. Voice fires in your language: "DANGER! Stop the vehicle now!"
4. Red border flashes around camera feed
5. Snapshot saved automatically with timestamp
6. Rest Stop button appears → Google Maps opens nearby places
7. Emergency WhatsApp → opens with your live GPS location
```

---

## ✨ Features

### 🔍 Core AI Detection

| Signal | Method | Threshold | Result |
|--------|--------|-----------|--------|
| 👁️ Eye Closure | Eye Aspect Ratio (EAR) | EAR < 0.28 | Drowsiness detected |
| 🥱 Yawning | Mouth Aspect Ratio (MAR) | MAR > 0.50 | Fatigue detected |
| 📐 Head Nodding | Pitch angle (3D pose) | Pitch > 20° | Distraction detected |
| 😶 Head Turning | Yaw angle (3D pose) | Yaw > 25° | Distraction detected |
| 🚫 No Face | MediaPipe detection | Face missing | Instant DANGER |

---

### ⚡ Risk Score Engine

```
Score = Eye (0-60) + Yawn (0-20) + Head Pose (0-25)

  0 ──────── 40 ──────── 70 ──────── 100
  │  SAFE    │  WARNING  │  DANGER   │
  🟢          🟡          🔴
```

| Level | Score | Actions |
|-------|-------|---------|
| 🟢 SAFE | 0 – 40 | Silent monitoring |
| 🟡 WARNING | 40 – 70 | Voice alert + dashboard banner |
| 🔴 DANGER | 70 – 100 | Loud alert + snapshot + Maps + WhatsApp |

---

### 🔊 Multilingual Voice Alerts

| Situation | English | ಕನ್ನಡ | हिंदी |
|-----------|---------|-------|-------|
| Eyes closing | "DANGER! Stop vehicle!" | "ಅಪಾಯ! ವಾಹನ ನಿಲ್ಲಿಸಿ!" | "खतरा! गाड़ी रोकें!" |
| Fatigue | "Take a break immediately!" | "ತಕ್ಷಣ ವಿರಾಮ ತೆಗೆದುಕೊಳ್ಳಿ!" | "तुरंत आराम करें!" |
| Warning | "Signs of fatigue. Stay alert." | "ಆಯಾಸದ ಲಕ್ಷಣಗಳು." | "थकान के संकेत." |
| No face | "Eyes on the road!" | "ರಸ್ತೆ ನೋಡಿ!" | "सड़क देखें!" |
| Back safe | "Driver is alert." | "ಸುರಕ್ಷಿತ." | "सुरक्षित।" |

---

### 📊 Live Dashboard Features

| Feature | Description |
|---------|-------------|
| 📷 Live Camera Feed | Real-time face monitoring with colored risk overlay |
| 📈 Live Risk Graph | Line chart updating every 400ms |
| 🌡️ Alert Frequency Chart | Bar chart — SAFE / WARNING / DANGER counts |
| 👁️ EAR & MAR Values | Raw detection metrics live |
| 😮 Yawn & Blink Counter | Session fatigue tracking |
| 🔔 Alert Log | All alerts with timestamps |
| 🌙 Dark / Light Mode | Toggle anytime |
| 🎬 Demo Mode | Simulates full DANGER cycle — perfect for presentations |

---

### 🗺️ Smart Location Finder

Always visible at bottom of dashboard:

| Button | Opens Google Maps Showing |
|--------|--------------------------|
| ⛽ Rest Stop | Nearest petrol bunks, dhabas, rest areas |
| 🅿️ Parking | Nearest safe car parking |
| 🏥 Hospital | Nearest hospitals and clinics |
| 🚓 Police | Nearest police stations |

> Uses your real GPS location for accurate results

---

### 🛡️ Safety & Emergency

| Feature | Description |
|---------|-------------|
| 🚑 Emergency WhatsApp | Sends live GPS location to family with one click |
| 📸 Auto Snapshots | Saves driver photo on every DANGER — 160+ captured in testing |
| 📸 Snapshots Gallery | View all danger photos inside the app |
| 📱 Phone Notifications | Browser push notifications on DANGER |
| 🎤 Voice Command | Say "I'm okay" to dismiss alerts hands-free |

---

### 🌦️ Additional Monitoring

| Feature | Description |
|---------|-------------|
| 🌦️ Weather Alert | Warns about rain, fog, strong winds in real-time |
| 😤 Stress Detection | Detects aggressive head movements — road rage indicator |
| 📊 Session Summary | Peak score, total alerts, yawns, safety verdict |
| 🕓 Session History | Stores and compares last 5 driving sessions |

---

## 📊 How It Works

```
Camera Input
    │
    ▼
MediaPipe Face Mesh (468 landmarks)
    │
    ├──► Eye landmarks  ──► EAR calculation
    ├──► Mouth landmarks ──► MAR calculation  
    └──► Nose/chin      ──► 3D Head Pose (pitch, yaw)
                                │
                                ▼
                    Risk Score Engine (0–100)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
         🟢 SAFE          🟡 WARNING          🔴 DANGER
         Monitor          Voice alert +       Voice alert +
         silently         Dashboard banner    Snapshot +
                                              Maps +
                                              WhatsApp
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11 | Core application |
| Computer Vision | OpenCV 4.x | Camera capture & frame processing |
| Face Detection | MediaPipe Face Mesh | 468 3D facial landmarks |
| Web Backend | Flask + Waitress | API server (8 threads) |
| Voice | gTTS + Pygame | Multilingual text-to-speech |
| Frontend | HTML5 + CSS3 + JavaScript | Dashboard UI |
| Charts | Chart.js | Live risk graph & frequency chart |
| Maps | Google Maps API | Location-based features |
| Weather | Open-Meteo API | Free real-time weather (no API key) |
| Notifications | Web Notifications API | Browser push alerts |
| Voice Commands | Web Speech API | Hands-free control |
| Emergency | WhatsApp URL API | Family alert system |
| Storage | Browser localStorage | Session history |

---

## 🚀 Quick Start

### Prerequisites
```
• Python 3.11+
• Laptop or phone webcam  
• Windows 10 / 11
• Internet connection (for voice + weather)
```

### Step 1 — Clone Repository
```bash
git clone https://github.com/yourusername/SixthSenseAI.git
cd SixthSenseAI
```

### Step 2 — Install Dependencies
```bash
pip install opencv-python
pip install mediapipe==0.10.9
pip install flask
pip install waitress
pip install gtts
pip install pygame
pip install scikit-learn
pip install numpy
```

### Step 3 — Run
```bash
python app.py
```

### Step 4 — Open Dashboard
```
http://127.0.0.1:5000
```

> Done! Camera opens and monitoring starts automatically.

---

## 📁 Project Structure

```
SixthSenseAI/
│
├── app.py              ← Flask backend: camera, routes, streaming
├── detector.py         ← MediaPipe: EAR, MAR, head pose detection  
├── risk_score.py       ← Risk engine: SAFE / WARNING / DANGER logic
│
├── static/
│   ├── style.css       ← Dashboard styling (dark + light mode)
│   └── script.js       ← Charts, alerts, voice, history, gallery
│
├── templates/
│   └── index.html      ← Main dashboard HTML
│
├── snapshots/          ← Auto-saved DANGER photos
├── sounds/             ← Voice alert MP3 files  
└── model/              ← ML model storage
```

---

## 🆚 SixthSense AI vs Competition

| Feature | Mobileye | Seeing Machines | Drivewell | SixthSense AI |
|---------|:--------:|:---------------:|:---------:|:-------------:|
| Cost | ₹50,000+ | ₹2–5 Lakhs | Free | **Free** ✅ |
| Hardware | Special cam | Infrared cam | Phone only | **Any webcam** ✅ |
| GPU required | Yes | Yes | No | **No** ✅ |
| Monitors driver face | ❌ | ✅ | ❌ | **✅** |
| Kannada support | ❌ | ❌ | ❌ | **✅** |
| Hindi support | ❌ | ❌ | ❌ | **✅** |
| Rest stop finder | ❌ | ❌ | ❌ | **✅** |
| Emergency WhatsApp | ❌ | ❌ | ❌ | **✅** |
| Weather alert | ❌ | ❌ | ❌ | **✅** |
| Stress detection | ❌ | ❌ | ❌ | **✅** |
| Auto snapshots | ❌ | ❌ | ❌ | **✅** |
| Open Source | ❌ | ❌ | ❌ | **✅** |

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Detection speed | Real-time @ 15–25 FPS |
| Alert response time | < 0.3 seconds after eyes close |
| Minimum hardware | Intel i5 CPU, 8GB RAM, no GPU |
| Server threads | 8 (Waitress multi-threaded) |
| Dashboard refresh | Every 400ms |
| Snapshots captured | 160+ in real testing |

---

## 🗺️ Roadmap

**Phase 1 — Complete ✅**
- [x] Drowsiness detection (EAR + MAR + head pose)
- [x] Multilingual voice alerts (English / Kannada / Hindi)
- [x] Live dashboard with charts
- [x] Snapshots gallery (160+ photos)
- [x] Emergency WhatsApp alert with GPS
- [x] Weather + stress monitoring
- [x] Smart location finder (4 categories)

**Phase 2 — Coming Soon**
- [ ] Android mobile app
- [ ] Pedestrian & animal detection (YOLOv8)
- [ ] Blind spot detection (dual camera)
- [ ] Driver score card (Grade A/B/C/D)
- [ ] Break reminder every 2 hours

**Phase 3 — Future Vision**
- [ ] Cloud fleet management dashboard
- [ ] OBD2 vehicle speed integration
- [ ] Night driving AI enhancement
- [ ] Insurance integration — safe driving rewards

---

## 🤝 Contributing

Contributions are welcome!

```bash
# Fork the repository
# Create your feature branch
git checkout -b feature/YourFeature

# Commit your changes
git commit -m "feat: Add YourFeature"

# Push and open Pull Request
git push origin feature/YourFeature
```

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

---

## 🙋 About

**Bharathi B R**  
Computer Science Engineering Student  
ATME College of Engineering, Mysuru — 2024 Batch  

> *"I built SixthSense AI to solve a real problem — affordable driver safety for every Indian on the road, in their own language."*

**Presented at:** HackSprint 6.0 — 24 Hour National Level Hackathon  
**Organized by:** PES College of Engineering, Mandya, May 2026

---

## ⭐ Support

If this project helped or inspired you:

- ⭐ **Star this repo** — helps others discover it
- 🍴 **Fork it** — build your own version  
- 🐛 **Report issues** — help make it better
- 📢 **Share it** — spread road safety awareness

---

<div align="center">

**Built with ❤️ by Bharathi B R**

*SixthSense AI — Saving Lives, One Alert at a Time* 🚗💚

</div>
