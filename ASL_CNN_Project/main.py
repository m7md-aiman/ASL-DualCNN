import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance
import time
import pyttsx3

# === CONFIG ===
MODEL_PATH = "Dual_CNN.keras"
LABEL_PATH = "label_names2.txt"
STABILITY_SECONDS = 1.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONFIRM_COLOR = (0, 255, 0)
INACTIVE_COLOR = (200, 200, 200)

# === Text-to-speech ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# === Load model and labels ===
model = load_model(MODEL_PATH)
with open(LABEL_PATH, "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# === Webcam ===
cap = cv2.VideoCapture(0)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === State ===
last_prediction = None
stable_start_time = None
confirmed_text = ""
hand_present_last_frame = False
progress = 0
last_confirm_time = 0

print("🖐️ Hold gesture for 1.5s to confirm. Press 'R'=reset, 'B'=backspace, 'Q'=quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_detected = False
    prediction = None
    confidence = 0

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        hand_detected = True

        # Landmarks
        lmk = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        lmk -= np.mean(lmk, axis=0)
        lmk = lmk[..., np.newaxis]
        lmk = np.expand_dims(lmk, axis=0)

        # Hand image crop
        lm_xy = np.array([[lm.x * w, lm.y * h] for lm in hand.landmark])
        x_min, y_min = lm_xy.min(axis=0).astype(int)
        x_max, y_max = lm_xy.max(axis=0).astype(int)
        pad = 30
        x1, y1 = max(x_min - pad, 0), max(y_min - pad, 0)
        x2, y2 = min(x_max + pad, w), min(y_max + pad, h)
        hand_img = frame[y1:y2, x1:x2]

        if hand_img.size == 0:
            continue

        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        hand_img = cv2.resize(hand_img, (50, 50))
        pil_img = Image.fromarray(hand_img)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
        img = np.array(pil_img).astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Predict
        preds = model.predict([img, lmk], verbose=0)
        idx = np.argmax(preds[0])
        prediction = LABELS[idx]
        confidence = float(np.max(preds[0]))

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # === Prediction logic with stability
    if hand_detected and prediction and (prediction != "Blank" or confidence > 0.97):
        if prediction == last_prediction:
            if stable_start_time:
                progress = (time.time() - stable_start_time) / STABILITY_SECONDS
                if progress >= 1.0:
                    confirmed_text += prediction
                    stable_start_time = None
                    last_prediction = None
                    last_confirm_time = time.time()
                    progress = 0
            else:
                stable_start_time = time.time()
        else:
            last_prediction = prediction
            stable_start_time = time.time()
            progress = 0
    else:
        if hand_present_last_frame:
            # Speak last word
            if confirmed_text.strip():
                full_sentence = confirmed_text.strip()
                engine.say(full_sentence)
                engine.runAndWait()

            confirmed_text += " "
        last_prediction = None
        stable_start_time = None
        progress = 0

    hand_present_last_frame = hand_detected

    # === Show prediction
    if prediction:
        text_color = CONFIRM_COLOR if progress >= 1.0 else (0, 255, 255)
        cv2.putText(frame, f"{prediction} ({confidence:.2f})", (10, 80), FONT, 1, text_color, 2)

    # === Draw progress bar
    bar_x, bar_y, bar_w, bar_h = 10, 110, 300, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), INACTIVE_COLOR, 2)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), CONFIRM_COLOR, -1)

    # === Show confirmed sentence (bold black text)
    cv2.putText(frame, confirmed_text, (10, 40), FONT, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, confirmed_text, (10, 40), FONT, 1.2, (255, 255, 255), 2)

    # === Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        confirmed_text = ""
    elif key == ord('b') and len(confirmed_text) > 0:
        confirmed_text = confirmed_text[:-1]

    cv2.imshow("ASL Live Translator", frame)

cap.release()
cv2.destroyAllWindows()
