import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading

# ================= TEXT TO SPEECH =================

is_speaking = False


def speak_worker(word):

    global is_speaking

    try:

        # Create new engine every time
        engine = pyttsx3.init()

        engine.setProperty('rate', 150)

        engine.say(word)

        engine.runAndWait()

        engine.stop()

    except Exception as e:

        print("Speech Error:", e)

    is_speaking = False


def speak_word(word):

    global is_speaking

    if is_speaking:
        return

    is_speaking = True

    speech_thread = threading.Thread(
        target=speak_worker,
        args=(word,),
        daemon=True
    )

    speech_thread.start()


# ================= LOAD MODEL =================

model_dict = pickle.load(open('./model.p', 'rb'))

model = model_dict['model']

le = model_dict['label_encoder']

# ================= CAMERA =================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ================= MEDIAPIPE =================

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ================= LABEL MAP =================

label_map = {
    'A': 'A',
    'B': 'B',
    'C': 'C',
    'D': 'D',
    'E': 'E',
    'F': 'F',
    'G': 'G',
    'H': 'H',
    'I': 'I',
    'J': 'J',
    'K': 'K',
    'L': 'L',
    'M': 'M',
    'N': 'N',
    'O': 'O',
    'P': 'P',
    'Q': 'Q',
    'R': 'R',
    'S': 'S',
    'T': 'T',
    'U': 'U',
    'V': 'V',
    'W': 'W',
    'X': 'X',
    'Y': 'Y',
    'Z': 'Z'
}

# ================= VARIABLES =================

formed_word = ""

current_letter = ""

last_letter = ""

stable_start_time = None

LETTER_HOLD_TIME = 1.0

NO_HAND_TIMEOUT = 2.0

last_hand_seen_time = time.time()

final_word_spoken = False

last_added_letter = ""

last_capture_time = 0

LETTER_COOLDOWN = 1.2

# ================= FUNCTIONS =================


def draw_sign_zone(frame, zone, hand_in_zone, has_hand):

    x, y, w, h = zone

    CL = 32

    if not has_hand:

        color = (200, 200, 200)

        label = 'Place hand inside zone'

    elif hand_in_zone:

        color = (0, 255, 0)

        label = 'Hand detected'

    else:

        color = (0, 165, 255)

        label = 'Hand outside zone'

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        color,
        1
    )

    cv2.addWeighted(
        overlay,
        0.3,
        frame,
        0.7,
        0,
        frame
    )

    pts = [
        ((x, y), (1, 1)),
        ((x + w, y), (-1, 1)),
        ((x, y + h), (1, -1)),
        ((x + w, y + h), (-1, -1)),
    ]

    for (cx, cy), (dx, dy) in pts:

        cv2.line(
            frame,
            (cx, cy),
            (cx + dx * CL, cy),
            color,
            3,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (cx, cy),
            (cx, cy + dy * CL),
            color,
            3,
            cv2.LINE_AA
        )

    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA
    )


def is_hand_in_zone(hand_landmarks, W, H, zone, threshold=12):

    x, y, w, h = zone

    inside = sum(
        1 for lm in hand_landmarks.landmark
        if x <= int(lm.x * W) <= x + w and
           y <= int(lm.y * H) <= y + h
    )

    return inside >= threshold


# ================= MAIN LOOP =================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape

    # ================= LEFT PANEL =================

    cv2.rectangle(
        frame,
        (0, 0),
        (300, H),
        (30, 30, 30),
        -1
    )

    # ================= SIGN ZONE =================

    zone_w, zone_h = 850, 380

    zone_x = (W - zone_w) // 2 + 120

    zone_y = (H - zone_h) // 2

    ZONE = (zone_x, zone_y, zone_w, zone_h)

    # ================= RGB =================

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(frame_rgb)

    has_hand = False

    hand_in_zone = False

    current_letter = ""

    # ================= DETECTION =================

    if results.multi_hand_landmarks:

        has_hand = True

        last_hand_seen_time = time.time()

        zone_hands = []

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            if is_hand_in_zone(
                hand_landmarks,
                W,
                H,
                ZONE
            ):

                hand_in_zone = True

                zone_hands.append(hand_landmarks)

        # ================= PREDICTION =================

        if zone_hands:

            data_aux = []

            hand_data = []

            for hand_landmarks in zone_hands:

                x_ = [lm.x for lm in hand_landmarks.landmark]

                y_ = [lm.y for lm in hand_landmarks.landmark]

                hand_data.append(
                    (x_, y_, hand_landmarks)
                )

            hand_data.sort(
                key=lambda h: min(h[0])
            )

            for x_, y_, _ in hand_data:

                x_min = min(x_)

                y_min = min(y_)

                for x, y in zip(x_, y_):

                    data_aux.append(x - x_min)

                    data_aux.append(y - y_min)

            # One hand

            if len(data_aux) == 42:

                data_aux += [0.0] * 42

            # Two hands

            if len(data_aux) == 84:

                prediction = model.predict(
                    [np.array(data_aux)]
                )

                class_label = le.inverse_transform(
                    prediction
                )[0]

                current_letter = label_map.get(
                    class_label,
                    class_label
                )

    # ================= AUTO LETTER CAPTURE =================

    current_time = time.time()

    if current_letter != "":

        final_word_spoken = False

        # New letter

        if current_letter != last_letter:

            last_letter = current_letter

            stable_start_time = current_time

        else:

            if stable_start_time is not None:

                elapsed = current_time - stable_start_time

                if elapsed >= LETTER_HOLD_TIME:

                    # Allow duplicate letters

                    can_add = False

                    if current_letter != last_added_letter:

                        can_add = True

                    else:

                        cooldown_elapsed = (
                            current_time - last_capture_time
                        )

                        if cooldown_elapsed >= LETTER_COOLDOWN:

                            can_add = True

                    if can_add:

                        formed_word += current_letter

                        last_added_letter = current_letter

                        last_capture_time = current_time

                        print("Captured:", current_letter)

                        stable_start_time = (
                            current_time + 999
                        )

    else:

        last_letter = ""

        stable_start_time = None

    # ================= FINAL WORD =================

    no_hand_elapsed = (
        current_time - last_hand_seen_time
    )

    if (
        no_hand_elapsed >= NO_HAND_TIMEOUT and
        formed_word != ""
    ):

        if not final_word_spoken:

            final_text = formed_word.strip()

            if final_text != "":

                print("\nFINAL WORD:", final_text)

                print("Speaking:", final_text)

                speak_word(final_text)

                final_word_spoken = True

        # ================= DISPLAY FINAL WORD =================

        cv2.putText(
            frame,
            'FINAL WORD',
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            formed_word,
            (40, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (255, 255, 255),
            5
        )

        # ================= RESET =================

        if not is_speaking:

            formed_word = ""

            current_letter = ""

            last_letter = ""

            last_added_letter = ""

            stable_start_time = None

            final_word_spoken = False

    # ================= LEFT PANEL =================

    cv2.putText(
        frame,
        'TEXT OUTPUT',
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        'CURRENT',
        (40, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        current_letter,
        (40, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.5,
        (0, 255, 0),
        5
    )

    cv2.putText(
        frame,
        'WORD',
        (40, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        formed_word,
        (40, 580),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (255, 255, 255),
        4
    )

    # ================= STATUS =================

    if current_letter != "":

        cv2.putText(
            frame,
            'Capturing...',
            (40, H - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    # ================= HAND COUNT =================

    num_detected = (
        len(results.multi_hand_landmarks)
        if results.multi_hand_landmarks
        else 0
    )

    cv2.putText(
        frame,
        f'Hands: {num_detected}',
        (W - 180, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # ================= DRAW ZONE =================

    draw_sign_zone(
        frame,
        ZONE,
        hand_in_zone,
        has_hand
    )

    # ================= INSTRUCTIONS =================

    cv2.putText(
        frame,
        'Hold sign for 1 second',
        (340, H - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

    cv2.putText(
        frame,
        'Remove hand for 2 sec to speak word',
        (620, H - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

    # ================= SHOW FRAME =================

    cv2.imshow(
        'Sign Language Word Builder',
        frame
    )

    # ================= KEYS =================

    key = cv2.waitKey(1) & 0xFF

    # Clear word

    if key == ord('c'):

        formed_word = ""

        current_letter = ""

        last_letter = ""

        last_added_letter = ""

        stable_start_time = None

        final_word_spoken = False

        print("\nWord Cleared")

    # Quit

    if key == ord('q'):

        break

# ================= CLEANUP =================

cap.release()

cv2.destroyAllWindows()