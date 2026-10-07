import os
import cv2
import joblib
import mediapipe as mp
import numpy as np
import threading
from collections import deque
from flask import Flask, render_template, Response, jsonify

# =========================
# PATHS & CONFIGURATION
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
        "id": "pataaka",
        "name": "Pataaka",
        "number": "Asamyuta Hasta #1",
        "meaning": "Flag / banner",
        "category": "Asamyuta Hasta (Single Hand Mudra)",
        "description": "Represents a flag, forest, stop, blessing and many other meanings in Bharatanatyam.",
        "ref_image": "/static/images/mudras/pataaka_ref.jpg",
        "card_image": "/static/images/mudras/pataaka_card.jpg"
    },
    "tripataka": {
        "id": "tripataka",
        "name": "Tripataka",
        "number": "Asamyuta Hasta #2",
        "meaning": "Three-part flag / Crown",
        "category": "Asamyuta Hasta (Single Hand Mudra)",
        "description": "Formed by bending the ring finger. Represents a crown, tree, thunderbolt, or arrow.",
        "ref_image": "/static/images/mudras/tripataka_ref.jpg",
        "card_image": "/static/images/mudras/tripataka_card.jpg"
    },
    "ardhapataka": {
        "id": "ardhapataka",
        "name": "Ardhapataaka",
        "number": "Asamyuta Hasta #3",
        "meaning": "Half flag / Leaves",
        "category": "Asamyuta Hasta (Single Hand Mudra)",
        "description": "Formed by bending little and ring fingers. Signifies leaves, a knife, tower, or river bank.",
        "ref_image": "/static/images/mudras/ardhapataka_ref.jpg",
        "card_image": "/static/images/mudras/ardhapataka_card.jpg"
    },
    "kartari_mukham": {
        "id": "kartari_mukham",
        "name": "Kartari Mukham",
        "number": "Asamyuta Hasta #4",
        "meaning": "Scissors / opening",
        "category": "Asamyuta Hasta (Single Hand Mudra)",
        "description": "Formed by crossing or separating index and middle fingers. Symbolizes scissors, opposition, or eyes.",
        "ref_image": "/static/images/mudras/kartari_mukham_ref.jpg",
        "card_image": "/static/images/mudras/kartari_mukham_card.jpg"
    },
    "mayuram": {
        "id": "mayuram",
        "name": "Mayuram",
        "number": "Asamyuta Hasta #5",
        "meaning": "Peacock",
        "category": "Asamyuta Hasta (Single Hand Mudra)",
        "description": "Formed by joining the thumb and ring finger tips. Signifies a peacock, beauty, or tilak mark.",
        "ref_image": "/static/images/mudras/mayuram_ref.jpg",
        "card_image": "/static/images/mudras/mayuram_card.jpg"
    }
}

# Thread-safe real-time state for API & UI
latest_recognition_lock = threading.Lock()
latest_recognition = {
    "detected": True,
    "hand_in_frame": True,
    "mudra_id": "pataaka",
    "name": "Pataaka",
    "number": "Asamyuta Hasta #1",
    "confidence": 94.3,
    "meaning": "Flag / banner",
    "category": "Asamyuta Hasta (Single Hand Mudra)",
    "description": "Represents a flag, forest, stop, blessing and many other meanings in Bharatanatyam.",
    "ref_image": "/static/images/mudras/pataaka_ref.jpg",
    "card_image": "/static/images/mudras/pataaka_card.jpg"
}


# =========================
# LOAD MODEL
# =========================

model = joblib.load(MODEL_PATH)


# =========================
# MEDIAPIPE SETUP
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
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    points = []

    for landmark in hand:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

        # Vibrant emerald landmark dots matching the UI design
        cv2.circle(
            frame,
            (x, y),
            5,
            (80, 230, 126),
            -1
        )
        cv2.circle(
            frame,
            (x, y),
            7,
            (50, 180, 90),
            1
        )

    for a, b in connections:
        cv2.line(
            frame,
            points[a],
            points[b],
            (100, 240, 140),
            2,
            cv2.LINE_AA
        )


# =========================
# VIDEO STREAM GENERATOR
# =========================

def generate_frames():
    global latest_recognition

    cap = cv2.VideoCapture(0)

    prediction_history = deque(
        maxlen=SMOOTHING_WINDOW
    )

    stable_prediction = "pataaka"
    stable_confidence = 94.3

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()

                if not success:
                    # If camera is unavailable, generate a smooth placeholder frame
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    blank[:] = (20, 16, 20)
                    cv2.putText(
                        blank,
                        "Connecting camera...",
                        (180, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (180, 160, 140),
                        2
                    )
                    _, buffer = cv2.imencode(".jpg", blank)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )
                    continue

                # Mirror camera
                frame = cv2.flip(frame, 1)

                # BGR -> RGB
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
                    draw_landmarks(frame, hand)

                    # Normalize & Predict
                    features = normalize_landmarks(hand)
                    X = features.reshape(1, -1)

                    prediction = model.predict(X)[0]
                    probabilities = model.predict_proba(X)[0]
                    confidence = float(np.max(probabilities) * 100)

                    if confidence >= CONFIDENCE_THRESHOLD:
                        prediction_history.append(prediction)

                        counts = {}
                        for p in prediction_history:
                            counts[p] = counts.get(p, 0) + 1

                        stable_prediction = max(counts, key=counts.get)
                        stable_confidence = confidence

                    if stable_prediction in MUDRA_INFO:
                        info = MUDRA_INFO[stable_prediction]
                        with latest_recognition_lock:
                            latest_recognition.update({
                                "detected": True,
                                "hand_in_frame": True,
                                "mudra_id": stable_prediction,
                                "name": info["name"],
                                "number": info["number"],
                                "confidence": round(stable_confidence, 1),
                                "meaning": info["meaning"],
                                "category": info["category"],
                                "description": info["description"],
                                "ref_image": info["ref_image"],
                                "card_image": info["card_image"]
                            })
                else:
                    prediction_history.clear()
                    with latest_recognition_lock:
                        latest_recognition["hand_in_frame"] = False

                # Encode clean frame (without intrusive OpenCV panels, matching UI design)
                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
    finally:
        cap.release()


# =========================
# FLASK APP SETUP & ROUTES
# =========================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)


@app.route("/")
def home():
    return render_template(
        "index.html",
        mudras=MUDRA_INFO,
        initial=latest_recognition
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def api_status():
    with latest_recognition_lock:
        return jsonify(latest_recognition)


@app.route("/api/mudras")
def api_mudras():
    return jsonify(MUDRA_INFO)


@app.route("/api/select/<mudra_id>")
def api_select(mudra_id):
    if mudra_id in MUDRA_INFO:
        info = MUDRA_INFO[mudra_id]
        with latest_recognition_lock:
            latest_recognition.update({
                "detected": True,
                "mudra_id": mudra_id,
                "name": info["name"],
                "number": info["number"],
                "confidence": 98.5,
                "meaning": info["meaning"],
                "category": info["category"],
                "description": info["description"],
                "ref_image": info["ref_image"],
                "card_image": info["card_image"]
            })
        return jsonify({"status": "success", "selected": mudra_id})
    return jsonify({"error": "Unknown mudra"}), 404


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )