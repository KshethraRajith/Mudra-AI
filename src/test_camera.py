import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open camera")
    exit()

print("✅ Camera opened successfully!")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("❌ Could not read frame")
        break

    cv2.imshow("Mudra-AI Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()