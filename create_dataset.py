import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

DATA_DIR = './data'

data = []
labels = []
skipped = 0
one_hand_count = 0
two_hand_count = 0

for dir_ in sorted(os.listdir(DATA_DIR)):
    class_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_path):
        continue

    print(f"\n[INFO] Processing class: {dir_}")

    for img_path in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_path))
        if img is None:
            print(f"  Warning: Could not read {img_path}, skipping.")
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Skip if NO hands detected at all
        if not results.multi_hand_landmarks:
            skipped += 1
            continue

        data_aux = []
        hand_data = []

        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            hand_data.append((x_, y_))

        # Sort by x so left hand always comes first
        hand_data.sort(key=lambda h: min(h[0]))

        for x_, y_ in hand_data:
            x_min, y_min = min(x_), min(y_)
            for x, y in zip(x_, y_):
                data_aux.append(x - x_min)
                data_aux.append(y - y_min)

        # Pad with zeros if only one hand detected
        if len(data_aux) == 42:
            data_aux += [0.0] * 42
            one_hand_count += 1
        elif len(data_aux) == 84:
            two_hand_count += 1
        else:
            # Unexpected length — skip
            skipped += 1
            continue

        data.append(data_aux)
        labels.append(dir_)

print(f"\n✓ Collected {len(data)} samples across {len(set(labels))} classes")
print(f"  Two-hand samples : {two_hand_count}")
print(f"  One-hand samples : {one_hand_count}  (padded with zeros)")
print(f"  Skipped          : {skipped}")
print(f"  Feature size     : {len(data[0]) if data else 'N/A'}")

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✓ Saved to data.pickle")
