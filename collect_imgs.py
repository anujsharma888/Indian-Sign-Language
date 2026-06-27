import os
import time
import cv2
import mediapipe as mp

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 11
dataset_size = 100
CAPTURE_DELAY = 0.05

# Choose mode here:
# 'one'    = only collect when 1 hand detected
# 'two'    = only collect when 2 hands detected
# 'hybrid' = collect whatever is visible (1 or 2 hands)
COLLECTION_MODE = 'hybrid'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def draw_guide_boxes(frame, h, w):
    # Left hand guide box
    cv2.rectangle(frame, (10, h//4), (w//2 - 10, 3*h//4), (255, 100, 0), 2)
    cv2.putText(frame, 'Left Hand', (15, h//4 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
    # Right hand guide box
    cv2.rectangle(frame, (w//2 + 10, h//4), (w - 10, 3*h//4), (0, 100, 255), 2)
    cv2.putText(frame, 'Right Hand', (w//2 + 15, h//4 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

def can_capture(num_hands):
    """Check if current hand count satisfies the collection mode."""
    if COLLECTION_MODE == 'one':
        return num_hands == 1
    elif COLLECTION_MODE == 'two':
        return num_hands == 2
    elif COLLECTION_MODE == 'hybrid':
        return num_hands >= 1
    return False

def get_mode_instruction():
    if COLLECTION_MODE == 'one':
        return 'Show 1 hand only'
    elif COLLECTION_MODE == 'two':
        return 'Show BOTH hands'
    elif COLLECTION_MODE == 'hybrid':
        return 'Show 1 or 2 hands'

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'\n[INFO] Class {j} — Mode: {COLLECTION_MODE.upper()} | {get_mode_instruction()}')

    # ── Wait loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        draw_guide_boxes(frame, h, w)

        ready = can_capture(num_hands)

        # Status text
        if ready:
            status_color = (0, 255, 0)
            status_text  = f'Class {j} | {num_hands} hand(s) ready | Press Q to start'
        else:
            status_color = (0, 0, 255)
            if COLLECTION_MODE == 'one':
                status_text = f'Class {j} | Show exactly 1 hand ({num_hands} detected)'
            elif COLLECTION_MODE == 'two':
                status_text = f'Class {j} | Need both hands ({num_hands}/2 detected)'
            else:
                status_text = f'Class {j} | Show at least 1 hand ({num_hands} detected)'

        cv2.putText(frame, status_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f'Mode: {COLLECTION_MODE.upper()} | {get_mode_instruction()}',
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2, cv2.LINE_AA)

        # Hand count indicator
        count_color = (0, 255, 0) if ready else (0, 0, 255)
        cv2.putText(frame, f'Hands: {num_hands}/{"1" if COLLECTION_MODE == "one" else "2"}',
                    (w - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, count_color, 2)

        cv2.imshow('Hybrid Data Collection', frame)

        key = cv2.waitKey(25)
        if key == ord('q') and ready:
            break
        elif key == ord('q') and not ready:
            print(f'  [!] Not ready — {get_mode_instruction()} before pressing Q')

    # ── Capture loop ──────────────────────────────────────────────────────────
    counter = 0
    one_hand_saved = 0
    two_hand_saved = 0
    consecutive_bad = 0

    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        display_frame = frame.copy()
        h, w, _ = display_frame.shape

        # Draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        draw_guide_boxes(display_frame, h, w)

        if can_capture(num_hands):
            consecutive_bad = 0

            # Progress bar
            progress = int((counter / dataset_size) * (w - 60))
            cv2.rectangle(display_frame, (30, h - 30), (w - 30, h - 10), (50, 50, 50), -1)
            cv2.rectangle(display_frame, (30, h - 30), (30 + progress, h - 10), (0, 255, 0), -1)

            # Status
            cv2.putText(display_frame, f'Saving class {j}: {counter + 1}/{dataset_size}',
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(display_frame,
                        f'1-hand: {one_hand_saved}  |  2-hand: {two_hand_saved}',
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

            # Hand count indicator
            cv2.putText(display_frame, f'Hands: {num_hands}/2',
                        (w - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save clean frame
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1

            if num_hands == 1:
                one_hand_saved += 1
            elif num_hands == 2:
                two_hand_saved += 1

        else:
            consecutive_bad += 1
            cv2.putText(display_frame,
                        f'Waiting... {get_mode_instruction()} ({num_hands} detected)',
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if consecutive_bad > 40:
                cv2.putText(display_frame, 'REPOSITION YOUR HANDS!',
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow('Hybrid Data Collection', display_frame)
        cv2.waitKey(25)
        time.sleep(CAPTURE_DELAY)

    print(f'✓ Class {j} done — {dataset_size} images saved '
          f'(1-hand: {one_hand_saved}, 2-hand: {two_hand_saved})')

cap.release()
cv2.destroyAllWindows()
print('\nDataset collection complete!')