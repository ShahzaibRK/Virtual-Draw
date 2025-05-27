import cv2
import mediapipe as mp
import numpy as np
import time
import os
from collections import deque

# Setup MediaPipe and drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Access the webcam
cap = cv2.VideoCapture(0)
canvas = None  # where we'll draw
drawing = False
prev_x, prev_y = 0, 0
smoothed_x, smoothed_y = 0, 0

# Save directory for screenshots
save_dir = "saved_drawings"
os.makedirs(save_dir, exist_ok=True)

# Pen colors – we'll switch between them with a double fist gesture
pen_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
current_color_idx = 0
pen_color = pen_colors[current_color_idx]

# Keep track of double fist timestamps
fist_times = deque(maxlen=5)

# Helper: checks if the pinch gesture is happening (thumb + index close together)
def is_pinch(thumb, index, threshold=40):
    distance = np.linalg.norm(np.array(thumb) - np.array(index))
    return distance < threshold

# Helper: converts landmark coordinates to pixel positions
def get_landmark_coords(landmarks, idx, shape):
    h, w = shape[:2]
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)

# Helper: simple logic to check if hand is in a closed fist pose
def detect_closed_fist(landmarks, shape):
    tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    mcp = mp_hands.HandLandmark.MIDDLE_FINGER_MCP
    center = get_landmark_coords(landmarks, mcp, shape)

    for tip in tips:
        tip_coord = get_landmark_coords(landmarks, tip, shape)
        if np.linalg.norm(np.array(tip_coord) - np.array(center)) > 80:
            return False
    return True

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror the webcam
        h, w, _ = frame.shape

        # Init canvas once
        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        multi_hand_landmarks = result.multi_hand_landmarks
        hand_labels = []

        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Pinch gesture
                index_finger_tip = get_landmark_coords(hand_landmarks.landmark, mp_hands.HandLandmark.INDEX_FINGER_TIP, frame.shape)
                thumb_tip = get_landmark_coords(hand_landmarks.landmark, mp_hands.HandLandmark.THUMB_TIP, frame.shape)

                # Double closed fist = switch color
                if detect_closed_fist(hand_landmarks.landmark, frame.shape):
                    fist_times.append(time.time())
                    if len(fist_times) >= 2 and (fist_times[-1] - fist_times[-2]) < 2.0:
                        current_color_idx = (current_color_idx + 1) % len(pen_colors)
                        pen_color = pen_colors[current_color_idx]
                        fist_times.clear()
                        cv2.putText(frame, "Color Changed!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, pen_color, 2)

                # Pinch to draw
                if is_pinch(thumb_tip, index_finger_tip):
                    drawing = True
                    x, y = index_finger_tip

                    # Smooth the drawing line a bit
                    smoothed_x = int(0.7 * smoothed_x + 0.3 * x) if smoothed_x else x
                    smoothed_y = int(0.7 * smoothed_y + 0.3 * y) if smoothed_y else y

                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = smoothed_x, smoothed_y

                    cv2.line(canvas, (prev_x, prev_y), (smoothed_x, smoothed_y), pen_color, 4)
                    prev_x, prev_y = smoothed_x, smoothed_y

                    cv2.putText(frame, "Drawing Mode", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, pen_color, 2)
                else:
                    drawing = False
                    prev_x, prev_y = 0, 0

                # Palm detection for canvas clear
                fingers_up = []
                for idx in [mp_hands.HandLandmark.THUMB_TIP,
                            mp_hands.HandLandmark.INDEX_FINGER_TIP,
                            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                            mp_hands.HandLandmark.RING_FINGER_TIP,
                            mp_hands.HandLandmark.PINKY_TIP]:
                    y_tip = get_landmark_coords(hand_landmarks.landmark, idx, frame.shape)[1]
                    y_dip = get_landmark_coords(hand_landmarks.landmark, idx - 2, frame.shape)[1]
                    fingers_up.append(y_tip < y_dip)

                hand_labels.append(all(fingers_up))

        # Clear canvas if both hands are open palms
        if hand_labels.count(True) >= 2:
            canvas = np.zeros_like(frame)
            cv2.putText(frame, "Canvas Cleared", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Blend drawing onto the live video
        combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
        cv2.imshow("Air Draw - Press 's' to Save | 'q' to Quit", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = os.path.join(save_dir, f"drawing_{int(time.time())}.png")
            cv2.imwrite(filename, canvas)
            print(f"Saved to {filename}")

cap.release()
cv2.destroyAllWindows()
