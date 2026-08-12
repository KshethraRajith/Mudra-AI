import cv2
import csv
import os
import time
import mediapipe as mp

# -----------------------------
# SETTINGS
# -----------------------------

LABEL = "pataaka"
SAMPLES = 500
SAVE_INTERVAL = 0.15  # seconds between samples

MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pataaka_landmarks.csv")

# -----------------------------
# MEDIAPIPE SETUP
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# CSV SETUP
# -----------------------------

header = ["label"]

for i in range(21):
    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

file_exists = os.path.exists(OUTPUT_FILE)

csv_file = open(
    OUTPUT_FILE,
    "a",
    newline=""
)

writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow(header)

# -----------------------------
# CAMERA
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open camera")
    exit()

print()
print("================================")
print("     PATAAKA DATA COLLECTION")
print("================================")
print()
print("Hold the Pataaka mudra in front")
print("of the camera.")
print()
print("Press SPACE to start collecting.")
print("Press Q to quit.")
print()

collecting = False
sample_count = 0
last_saved = 0

while True:

    ret, frame = camera.read()

    if not ret:
        print("❌ Could not read camera frame")
        break

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)

    # -----------------------------
    # DRAW HAND
    # -----------------------------

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        # Draw points
        for landmark in hand_landmarks:

            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Draw connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        for start, end in connections:

            x1 = int(
                hand_landmarks[start].x
                * frame.shape[1]
            )
            y1 = int(
                hand_landmarks[start].y
                * frame.shape[0]
            )

            x2 = int(
                hand_landmarks[end].x
                * frame.shape[1]
            )
            y2 = int(
                hand_landmarks[end].y
                * frame.shape[0]
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        # -----------------------------
        # COLLECT DATA
        # -----------------------------

        if collecting:

            current_time = time.time()

            if (
                current_time - last_saved
                >= SAVE_INTERVAL
                and sample_count < SAMPLES
            ):

                row = [LABEL]

                for landmark in hand_landmarks:

                    row.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                writer.writerow(row)
                csv_file.flush()

                sample_count += 1
                last_saved = current_time

    # -----------------------------
    # DISPLAY STATUS
    # -----------------------------

    if not collecting:

        cv2.putText(
            frame,
            "Press SPACE to start",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            f"Pataaka samples: {sample_count}/{SAMPLES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Mudra-AI Pataaka Dataset Collector",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # SPACE → start
    if key == ord(" "):

        if not collecting:
            collecting = True
            print("🟢 Collection started!")

    # Q → quit
    elif key == ord("q"):
        break

    # Stop automatically
    if sample_count >= SAMPLES:

        print()
        print("✅ 500 Pataaka samples collected!")
        break


camera.release()
csv_file.close()
landmarker.close()

cv2.destroyAllWindows()

print()
print("================================")
print("      COLLECTION COMPLETE")
print("================================")
print(f"Dataset saved to:")
print(OUTPUT_FILE)
print()