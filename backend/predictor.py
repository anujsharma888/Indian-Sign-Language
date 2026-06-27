import pickle
import cv2
import mediapipe as mp
import numpy as np

# ================= LOAD MODEL =================

model_dict = pickle.load(open('../model.p', 'rb'))

model = model_dict['model']

le = model_dict['label_encoder']

# ================= MEDIAPIPE =================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ================= PREDICT FUNCTION =================

def predict_frame(frame):

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None

    data_aux = []

    hand_data = []

    x_min = 1
    y_min = 1
    x_max = 0
    y_max = 0

    for hand_landmarks in results.multi_hand_landmarks:

        x_ = []
        y_ = []

        for lm in hand_landmarks.landmark:

            x_.append(lm.x)
            y_.append(lm.y)

            x_min = min(x_min, lm.x)
            y_min = min(y_min, lm.y)

            x_max = max(x_max, lm.x)
            y_max = max(y_max, lm.y)

        hand_data.append((x_, y_))

    # SORT LEFT → RIGHT

    hand_data.sort(
        key=lambda h: min(h[0])
    )

    for x_, y_ in hand_data:

        local_x_min = min(x_)
        local_y_min = min(y_)

        for x, y in zip(x_, y_):

            data_aux.append(x - local_x_min)
            data_aux.append(y - local_y_min)

    # ONE HAND PADDING

    if len(data_aux) == 42:

        data_aux += [0.0] * 42

    # INVALID SIZE

    if len(data_aux) != 84:

        return None

    prediction = model.predict(
        [np.asarray(data_aux)]
    )

    class_label = le.inverse_transform(
        prediction
    )[0]

    bbox = {
        "x": x_min,
        "y": y_min,
        "w": x_max - x_min,
        "h": y_max - y_min
    }

    return {
        "label": class_label,
        "bbox": bbox
    }