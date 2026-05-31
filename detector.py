import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────
#  MediaPipe Setup
# ─────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────
#  Landmark Index Constants
# ─────────────────────────────────────────

# Left eye landmarks
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye landmarks
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
# Mouth landmarks
MOUTH     = [61,  291, 39,  181, 0,   17,  269, 405]
# Head pose landmarks (nose tip, chin, left eye corner, right eye corner, left mouth, right mouth)
HEAD_POSE_POINTS = [1, 152, 263, 33, 287, 57]

# ─────────────────────────────────────────
#  EAR — Eye Aspect Ratio
#  If EAR < threshold → eye is closing
# ─────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * img_w, lm.y * img_h))

    # Vertical distances
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (A + B) / (2.0 * C)
    return ear


# ─────────────────────────────────────────
#  MAR — Mouth Aspect Ratio
#  If MAR > threshold → mouth is open (yawn)
# ─────────────────────────────────────────
def mouth_aspect_ratio(landmarks, mouth_indices, img_w, img_h):
    points = []
    for idx in mouth_indices:
        lm = landmarks[idx]
        points.append((lm.x * img_w, lm.y * img_h))

    # Vertical distances
    A = np.linalg.norm(np.array(points[2]) - np.array(points[6]))
    B = np.linalg.norm(np.array(points[3]) - np.array(points[5]))
    # Horizontal distance
    C = np.linalg.norm(np.array(points[0]) - np.array(points[1]))

    mar = (A + B) / (2.0 * C)
    return mar


# ─────────────────────────────────────────
#  Head Pose — detect tilting/nodding
#  Returns (pitch, yaw, roll) in degrees
# ─────────────────────────────────────────
def head_pose_estimation(landmarks, img_w, img_h):
    # 3D model points (standard face model)
    model_points = np.array([
        (0.0,    0.0,    0.0),      # Nose tip
        (0.0,   -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),    # Left eye corner
        (225.0,  170.0, -135.0),    # Right eye corner
        (-150.0,-150.0, -125.0),    # Left mouth
        (150.0, -150.0, -125.0)     # Right mouth
    ], dtype=np.float64)

    # 2D image points from landmarks
    image_points = []
    for idx in HEAD_POSE_POINTS:
        lm = landmarks[idx]
        image_points.append((lm.x * img_w, lm.y * img_h))
    image_points = np.array(image_points, dtype=np.float64)

    # Camera internals (approximation)
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1         ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0, 0, 0

    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Get Euler angles
    proj_matrix = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    euler_angles = euler_angles.flatten()
    pitch = float(euler_angles[0])   # Up/down tilt
    yaw   = float(euler_angles[1])   # Left/right turn
    roll  = float(euler_angles[2])   # Head tilt

    return pitch, yaw, roll


# ─────────────────────────────────────────
#  MAIN DETECTOR — runs on each frame
#  Returns a dict of all readings
# ─────────────────────────────────────────
def detect_frame(frame):
    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    data = {
        "face_detected": False,
        "ear":   1.0,   # high = eyes open
        "mar":   0.0,   # low  = mouth closed
        "pitch": 0.0,
        "yaw":   0.0,
        "roll":  0.0,
    }

    if results.multi_face_landmarks:
        data["face_detected"] = True
        landmarks = results.multi_face_landmarks[0].landmark

        # Eye Aspect Ratio (average of both eyes)
        left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  img_w, img_h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, img_w, img_h)
        data["ear"] = round((left_ear + right_ear) / 2.0, 4)

        # Mouth Aspect Ratio
        data["mar"] = round(mouth_aspect_ratio(landmarks, MOUTH, img_w, img_h), 4)

        # Head Pose
        pitch, yaw, roll = head_pose_estimation(landmarks, img_w, img_h)
        data["pitch"] = round(pitch, 2)
        data["yaw"]   = round(yaw,   2)
        data["roll"]  = round(roll,  2)

    return data


# ─────────────────────────────────────────
#  QUICK TEST — run this file directly
#  to verify your camera works
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Starting camera test... Press Q to quit.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not found!")
            break

        data = detect_frame(frame)

        # Display readings on screen
        color = (0, 255, 0) if data["face_detected"] else (0, 0, 255)
        cv2.putText(frame, f"EAR: {data['ear']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"MAR: {data['mar']}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Pitch: {data['pitch']}  Yaw: {data['yaw']}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        status = "FACE DETECTED ✓" if data["face_detected"] else "NO FACE"
        cv2.putText(frame, status, (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("SixthSense AI - Detector Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test complete!")