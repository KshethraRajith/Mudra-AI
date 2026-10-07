import os
import cv2
import joblib
import mediapipe as mp
import numpy as np
from collections import deque
from flask import Flask, render_template, Response

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "mudra_classifier.pkl"
)

HAND_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "hand_landmarker.task"
)

CONFIDENCE_THRESHOLD = 50.0
SMOOTHING_WINDOW = 7


# =========================
# MUDRA INFORMATION
# =========================

MUDRA_INFO = {
    "pataaka": {
        "name": "Pataaka",
        "number": "Asamyuta Hasta #1",
        "meaning": "Flag / banner"
    },
    "tripataka": {
        "name": "Tripataka",
        "number": "Asamyuta Hasta #2",
        "meaning": "Three-part flag"
    },
    "ardhapataka": {
        "name": "Ardhapataaka",
        "number": "Asamyuta Hasta #3",
        "meaning": "Half flag"
    },
    "kartari_mukham": {
        "name": "Kartari Mukham",
        "number": "Asamyuta Hasta #4",
        "meaning": "Scissors / opening"
    },
    "mayuram": {
        "name": "Mayuram",
        "number": "Asamyuta Hasta #5",
        "meaning": "Peacock"
    }
}


# =========================
# LOAD MODEL
# =========================

model = joblib.load(MODEL_PATH)


# =========================
# MEDIAPIPE
# =========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


# =========================
# NORMALIZE LANDMARKS
# =========================

def normalize_landmarks(hand):

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=float
    )

    landmarks = landmarks - landmarks[0]

    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    scale = np.max(distances)

    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten()


# =========================
# DRAW LANDMARKS
# =========================

def draw_landmarks(frame, hand):

    h, w = frame.shape[:2]

    connections = [
        (0,1), (1,2), (2,3), (3,4),
        (0,5), (5,6), (6,7), (7,8),
        (0,9), (9,10), (10,11), (11,12),
        (0,13), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20),
        (5,9), (9,13), (13,17)
    ]

    points = []

    for landmark in hand:

        x = int(landmark.x * w)
        y = int(landmark.y * h)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    for a, b in connections:

        cv2.line(
            frame,
            points[a],
            points[b],
            (0, 255, 0),
            2
        )


# =========================
# VIDEO STREAM
# =========================

def generate_frames():

    cap = cv2.VideoCapture(0)

    prediction_history = deque(
        maxlen=SMOOTHING_WINDOW
    )

    stable_prediction = None
    stable_confidence = 0

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            # BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)


            # =========================
            # HAND DETECTED
            # =========================

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                draw_landmarks(
                    frame,
                    hand
                )

                # Normalize
                features = normalize_landmarks(
                    hand
                )

                X = features.reshape(
                    1, -1
                )

                # Prediction
                prediction = model.predict(X)[0]

                probabilities = model.predict_proba(X)[0]

                confidence = (
                    np.max(probabilities) * 100
                )

                # Confidence filtering
                if confidence >= CONFIDENCE_THRESHOLD:

                    prediction_history.append(
                        prediction
                    )

                    counts = {}

                    for p in prediction_history:
                        counts[p] = counts.get(p, 0) + 1

                    stable_prediction = max(
                        counts,
                        key=counts.get
                    )

                    stable_confidence = confidence


                # =========================
                # DISPLAY
                # =========================

                if stable_prediction in MUDRA_INFO:

                    info = MUDRA_INFO[
                        stable_prediction
                    ]

                    # Background panel
                    cv2.rectangle(
                        frame,
                        (15, 15),
                        (450, 155),
                        (20, 20, 20),
                        -1
                    )

                    # Mudra name
                    cv2.putText(
                        frame,
                        info["name"],
                        (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 255, 0),
                        3
                    )

                    # Number
                    cv2.putText(
                        frame,
                        info["number"],
                        (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                    # Confidence
                    cv2.putText(
                        frame,
                        f"Confidence: {stable_confidence:.1f}%",
                        (30, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                    # Meaning
                    cv2.putText(
                        frame,
                        info["meaning"],
                        (30, 145),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (220, 220, 220),
                        1
                    )

                else:

                    cv2.putText(
                        frame,
                        "Analyzing...",
                        (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )


            # =========================
            # NO HAND
            # =========================

            else:

                prediction_history.clear()

                stable_prediction = None
                stable_confidence = 0

                cv2.putText(
                    frame,
                    "Show your hand",
                    (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )


            # =========================
            # ENCODE FRAME
            # =========================

            success, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    cap.release()


# =========================
# FLASK APP
# =========================

app = Flask(
    __name__,
    template_folder=os.path.join(
        BASE_DIR,
        "templates"
    )
)


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )