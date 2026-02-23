import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
from collections import deque

# ===============================
# Load Model
# ===============================

model = tf.keras.models.load_model("sign_model.h5")

# ===============================
# Text to Speech Setup
# ===============================

engine = pyttsx3.init()

# Change language voice (optional)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Change index if needed

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ===============================
# MediaPipe Setup
# ===============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

prediction_queue = deque(maxlen=10)
current_word = ""

# ===============================
# Webcam
# ===============================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    predicted_letter = ""
    confidence = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                data = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(data, verbose=0)

                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence > 0.80:
                    prediction_queue.append(class_index)

                    if len(prediction_queue) == 10:
                        final_index = max(set(prediction_queue), key=prediction_queue.count)
                        predicted_letter = labels[final_index]

    # Add letter when pressing ENTER
    key = cv2.waitKey(1)

    if key == 13 and predicted_letter:  # Enter key
        current_word += predicted_letter
        prediction_queue.clear()

    # Delete last letter
    if key == ord('d'):
        current_word = current_word[:-1]

    # Speak word
    if key == ord('s'):
        speak(current_word)

    # Clear word
    if key == ord('c'):
        current_word = ""

    # Display
    cv2.putText(frame, f"Letter: {predicted_letter}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Word: {current_word}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Language Word Generator", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
