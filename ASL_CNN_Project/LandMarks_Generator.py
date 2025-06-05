import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

IMG_DIR = Path("/content/asl_dual_dataset/images")
LMK_DIR = Path("/content/asl_dual_dataset/landmarks")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

image_paths = list(IMG_DIR.rglob("*.jpg")) + list(IMG_DIR.rglob("*.png"))

for img_path in tqdm(image_paths, desc="Generating landmarks"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)  # (21, 3)
    else:
        coords = np.zeros((21, 3), dtype=np.float32)  # blank if no hand found

    save_path = str(img_path).replace("/images/", "/landmarks/").rsplit(".", 1)[0] + ".npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, coords)

print("✅ All landmarks regenerated with shape (21, 3)")
