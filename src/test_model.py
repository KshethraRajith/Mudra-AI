import cv2
import joblib
import mediapipe as mp
import numpy as np


# =========================
# SETTINGS
# =========================

MODEL_PATH = "models/mudra_classifier.pkl"
HAND_MODEL_PATH = "models/hand_landmarker.task"


# =========================
# NORMALIZE LANDMARKS
# =========================

def normalize_landmarks(hand):
    """
    Normalize MediaPipe hand landmarks
    using the wrist as the origin.
    """

    landmarks = np.array([
        [landmark.x, landmark.y, landmark.z]
        for landmark in hand
    ])

    # Use wrist as origin
    wrist = landmarks[0]

    landmarks = landmarks - wrist

    # Calculate hand size
    distances = np.linalg.norm(landmarks, axis=1)

    scale = np.max(distances)

    # Avoid division by zero
    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten()


# =========================
# LOAD TRAINED MODEL
# =========================

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Classes:", model.classes_)


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
# CAMERA
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started.")
print("Press Q to quit.")


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

        # Create MediaPipe image
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


            # =========================
            # DRAW LANDMARKS
            # =========================

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
            # CREATE NORMALIZED FEATURES
            # =========================

            features = normalize_landmarks(hand)

            X = features.reshape(1, -1)


            # =========================
            # PREDICTION
            # =========================

            prediction = model.predict(X)[0]

            probabilities = model.predict_proba(X)[0]

            confidence = np.max(probabilities) * 100


            # =========================
            # DISPLAY
            # =========================

            cv2.putText(
                frame,
                f"Mudra: {prediction}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        else:

            cv2.putText(
                frame,
                "No hand detected",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        # =========================
        # SHOW CAMERA
        # =========================

        cv2.imshow(
            "Mudra-AI Live Prediction",
            frame
        )


        # =========================
        # QUIT
        # =========================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()

print("Camera closed.")
print("Test completed.")