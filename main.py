import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# --- Model setup ---
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- Skeleton ---
CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
    (0,11),(0,12)
]

def draw_skeleton(frame, landmarks, w, h):
    for a, b in CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
            x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 160, 70), 2)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 220, 100), -1)

# --- Swing sequence detector ---
class SwingDetector:
    def __init__(self):
        self.history = deque(maxlen=90)
        self.phase = "ADDRESS"
        self.phase_sequence = []
        self.cooldown = 0
        self.impact_landmarks = None

    def update(self, landmarks):
        if self.cooldown > 0:
            self.cooldown -= 1

        r_wrist    = landmarks[16]
        l_wrist    = landmarks[15]
        r_shoulder = landmarks[12]
        l_shoulder = landmarks[11]
        r_hip      = landmarks[24]
        l_hip      = landmarks[23]

        # --- Upgrade 1: Track both wrists ---
        avg_wrist_y = (r_wrist.y + l_wrist.y) / 2

        # --- Upgrade 3: Body rotation ---
        shoulder_rotation = abs(r_shoulder.x - l_shoulder.x)
        hip_rotation      = abs(r_hip.x - l_hip.x)

        # Store frame data
        frame_data = {
            "avg_wrist_y":        avg_wrist_y,
            "r_wrist_y":          r_wrist.y,
            "shoulder_rotation":  shoulder_rotation,
            "hip_rotation":       hip_rotation,
            "landmarks":          landmarks
        }
        self.history.append(frame_data)

        # --- Upgrade 2: Check sequence ---
        wrist_above_shoulder = avg_wrist_y < r_shoulder.y - 0.08
        wrist_above_hip      = avg_wrist_y < r_hip.y

        if wrist_above_shoulder:
            new_phase = "BACKSWING"
        elif wrist_above_hip:
            new_phase = "SETUP"
        else:
            new_phase = "ADDRESS"

        if new_phase != self.phase:
            self.phase_sequence.append(new_phase)
            if len(self.phase_sequence) > 6:
                self.phase_sequence.pop(0)
            self.phase = new_phase

        return self.phase

    def check_for_swing(self):
        if self.cooldown > 0 or len(self.history) < 45:
            return False

        wrist_ys   = [f["avg_wrist_y"] for f in self.history]
        swing_range = max(wrist_ys) - min(wrist_ys)

        # --- Upgrade 1: Both wrists traveled far enough ---
        went_high = min(wrist_ys) < 0.42

        # --- Upgrade 2: Correct phase sequence happened ---
        seq = self.phase_sequence
        had_backswing = "BACKSWING" in seq
        had_address   = "ADDRESS" in seq
        correct_sequence = had_backswing and had_address

        # --- Upgrade 3: Body actually rotated ---
        rotations = [f["shoulder_rotation"] for f in self.history]
        max_rotation = max(rotations)
        min_rotation = min(rotations)
        body_rotated = (max_rotation - min_rotation) > 0.05

        # --- Upgrade 4: Swing had acceleration (speed check) ---
        recent_wrists = wrist_ys[-30:]
        speeds = [abs(recent_wrists[i] - recent_wrists[i-1])
                  for i in range(1, len(recent_wrists))]
        max_speed = max(speeds) if speeds else 0
        fast_enough = max_speed > 0.008

        if swing_range > 0.50 and went_high and correct_sequence and body_rotated and fast_enough:
            # Find impact frame (wrist at lowest point = max Y in last 30 frames)
            recent = list(self.history)[-30:]
            impact_idx = max(range(len(recent)), key=lambda i: recent[i]["r_wrist_y"])
            self.impact_landmarks = recent[impact_idx]["landmarks"]
            self.cooldown = 120
            self.phase_sequence.clear()
            return True

        return False

# --- Shot analysis ---
def analyse_shot(wrist_history_data, landmarks_at_impact):
    lm = landmarks_at_impact

    r_shoulder = lm[12]
    l_shoulder = lm[11]
    r_hip      = lm[24]
    l_hip      = lm[23]
    r_wrist    = lm[16]
    l_wrist    = lm[15]
    r_elbow    = lm[14]

    # Swing path from shoulder tilt
    shoulder_tilt = (r_shoulder.y - l_shoulder.y) * 100
    hip_alignment = (r_hip.x - l_hip.x) * 100

    if shoulder_tilt > 4:
        swing_path = "outside_in"
    elif shoulder_tilt < -4:
        swing_path = "inside_out"
    else:
        swing_path = "neutral"

    # Clubface from wrist/elbow angle
    wrist_elbow_diff = (r_wrist.x - r_elbow.x) * 100
    if wrist_elbow_diff > 5:
        clubface = "open"
    elif wrist_elbow_diff < -5:
        clubface = "closed"
    else:
        clubface = "square"

    # Shot type
    shot_map = {
        ("outside_in", "open"):   ("SLICE",    (0, 60, 220)),
        ("outside_in", "square"): ("FADE",     (0, 140, 220)),
        ("outside_in", "closed"): ("PULL",     (220, 100, 0)),
        ("inside_out", "closed"): ("HOOK",     (0, 60, 220)),
        ("inside_out", "square"): ("DRAW",     (0, 200, 100)),
        ("inside_out", "open"):   ("PUSH",     (220, 180, 0)),
        ("neutral",    "open"):   ("FADE",     (0, 140, 220)),
        ("neutral",    "closed"): ("DRAW",     (0, 200, 100)),
        ("neutral",    "square"): ("STRAIGHT", (0, 220, 100)),
    }
    shot_type, shot_color = shot_map.get((swing_path, clubface), ("STRAIGHT", (0, 220, 100)))

    # Speed from wrist travel
    wrist_ys = [f["avg_wrist_y"] for f in wrist_history_data]
    if len(wrist_ys) >= 20:
        recent = wrist_ys[-20:]
        speed = sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent)))
        speed_score = min(100, int(speed * 800))
    else:
        speed_score = 50

    hip_rotation_score = min(100, max(0, int(abs(hip_alignment) * 3 + 50)))

    # Distance estimate for irons
    distance = int(150 * (0.5 + 0.3 * speed_score/100 + 0.2 * hip_rotation_score/100))

    tips = {
        "SLICE":    "Outside-in path detected. Drop the club inside on the downswing and rotate through impact.",
        "FADE":     "Slight outside-in path — a controlled fade. Strengthen grip slightly to straighten it out.",
        "PULL":     "Outside-in with closed face. Check your alignment and move the ball back in your stance.",
        "HOOK":     "Inside-out path with closed face. Keep the clubface square and hold off the release.",
        "DRAW":     "Great inside-out path — a controlled draw. Perfect shape for an iron!",
        "PUSH":     "Inside-out but face is open. Rotate your forearms through impact more aggressively.",
        "STRAIGHT": "Excellent — neutral path and square face. Focus on repeating this consistently.",
    }

    return {
        "shot_type":    shot_type,
        "shot_color":   shot_color,
        "distance":     distance,
        "speed_score":  speed_score,
        "hip_rotation": hip_rotation_score,
        "clubface":     clubface,
        "swing_path":   swing_path,
        "feedback":     tips.get(shot_type, "Keep practicing!")
    }

def draw_results(frame, results, w, h):
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 340, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (w - 340, 0), (w - 340, h), (0, 180, 70), 1)

    x = w - 320
    y = 50

    cv2.putText(frame, results["shot_type"], (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, results["shot_color"], 3)
    y += 50
    cv2.line(frame, (x, y), (w - 20, y), (0, 80, 30), 1)
    y += 25

    for label, value in [
        ("DISTANCE",    f"{results['distance']} yards"),
        ("SWING PATH",  results["swing_path"].replace("_"," ").upper()),
        ("CLUBFACE",    results["clubface"].upper()),
    ]:
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        y += 25
        cv2.putText(frame, value, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)
        y += 45

    for label, score in [("SWING SPEED", results["speed_score"]),
                          ("HIP ROTATION", results["hip_rotation"])]:
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        y += 22
        bar_w = int((w - 40 - x) * score / 100)
        cv2.rectangle(frame, (x, y), (w - 40, y + 12), (30, 30, 30), -1)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + 12), (0, 180, 70), -1)
        cv2.putText(frame, f"{score}%", (x, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y += 50

    cv2.line(frame, (x, y), (w - 20, y), (0, 80, 30), 1)
    y += 20
    cv2.putText(frame, "CADDIE TIP", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 70), 1)
    y += 22
    words = results["feedback"].split()
    line = ""
    for word in words:
        test = line + word + " "
        if len(test) > 28:
            cv2.putText(frame, line.strip(), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            y += 22
            line = word + " "
        else:
            line = test
    if line:
        cv2.putText(frame, line.strip(), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

# --- Main ---
swing_detector = SwingDetector()
last_results   = None

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Stand side-on to the camera and take a full swing! Q = quit, R = reset")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    phase = "STAND IN FRAME"

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        draw_skeleton(frame, landmarks, w, h)
        phase = swing_detector.update(landmarks)

        if swing_detector.check_for_swing():
            last_results = analyse_shot(
                list(swing_detector.history),
                swing_detector.impact_landmarks
            )

    if last_results:
        draw_results(frame, last_results, w, h)

    # Top bar
    bar_w = w - 340 if last_results else w
    cv2.rectangle(frame, (0, 0), (bar_w, 65), (0, 0, 0), -1)

    phase_colors = {
        "BACKSWING": (0, 220, 100),
        "SETUP":     (0, 180, 220),
        "ADDRESS":   (160, 160, 160)
    }
    cv2.putText(frame, phase, (20, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                phase_colors.get(phase, (160, 160, 160)), 2)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "Stand side-on | Full swing to analyse | R = reset | Q = quit",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.imshow("Golf Swing Analyzer", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        swing_detector = SwingDetector()
        last_results   = None

cap.release()
cv2.destroyAllWindows()
