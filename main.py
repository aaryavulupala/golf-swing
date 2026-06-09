import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Model setup ---
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

# Skeleton connections (pairs of landmark indices)
CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),  # arms
    (11,23),(12,24),(23,24),                   # torso
    (23,25),(25,27),(24,26),(26,28),           # legs
    (0,11),(0,12)                              # head to shoulders
]

def draw_skeleton(frame, landmarks, w, h):
    # Draw connections
    for a, b in CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            x1 = int(landmarks[a].x * w)
            y1 = int(landmarks[a].y * h)
            x2 = int(landmarks[b].x * w)
            y2 = int(landmarks[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 160, 70), 2)

    # Draw joints
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 220, 100), -1)

# --- Camera ---
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

print("Camera open! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.pose_landmarks:
        draw_skeleton(frame, result.pose_landmarks[0], w, h)
        cv2.putText(frame, "Person detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 100), 2)
    else:
        cv2.putText(frame, "Stand in frame", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

    cv2.imshow("Golf Swing Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
