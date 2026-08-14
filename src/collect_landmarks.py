import cv2
import csv
import os
import mediapipe as mp

# =========================
# SETTINGS
# =========================

MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_FILE = "data/raw/mudra_landmarks.csv"

SAMPLES_TO_COLLECT = 1000

# =========================
# ASK FOR MUDRA LABEL
# =========================

label = input("Enter mudra label: ").strip().lower()

if not label:
    print("Error: Mudra label cannot be empty.")
    exit()

print(f"\nCollecting samples for: {label}")
print(f"Target samples: {SAMPLES_TO_COLLECT}")
print("Press SPACE to start/pause collection.")
print("Press Q to quit.\n")

# =========================
# MEDIAPIPE SETUP
# =========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# =========================
# CSV SETUP
# =========================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

file_exists = os.path.exists(OUTPUT_FILE)

csv_file = open(OUTPUT_FILE, "a", newline="")

writer = csv.writer(csv_file)

if not file_exists:
    header = ["label"]

    for i in range(21):
        header.extend([
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ])

    writer.writerow(header)

# =========================
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    csv_file.close()
    exit()

collecting = False
count = 0

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        # =========================
        # HAND DETECTED
        # =========================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Draw the 21 landmarks
            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

            # =========================
            # COLLECT DATA
            # =========================

            if collecting and count < SAMPLES_TO_COLLECT:

                row = [label]

                for landmark in hand:

                    row.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                writer.writerow(row)
                csv_file.flush()

                count += 1

        # =========================
        # DISPLAY STATUS
        # =========================

        if collecting:
            status = f"COLLECTING: {count}/{SAMPLES_TO_COLLECT}"
            color = (0, 255, 0)
        else:
            status = "PAUSED - Press SPACE"
            color = (0, 255, 255)

        cv2.putText(
            frame,
            f"Mudra: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow(
            "Mudra-AI Landmark Collection",
            frame
        )

        # =========================
        # KEYBOARD CONTROLS
        # =========================

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            collecting = not collecting

        elif key == ord("q"):
            break

        # =========================
        # FINISHED
        # =========================

        if count >= SAMPLES_TO_COLLECT:

            collecting = False

            print(
                f"\nFinished collecting {count} "
                f"samples for '{label}'."
            )

            break

# =========================
# CLEANUP
# =========================

csv_file.close()
cap.release()
cv2.destroyAllWindows()

print(f"\nSaved {count} samples for '{label}'")
print(f"File: {OUTPUT_FILE}")