"""
Description:
    This script captures images from a webcam and stores them into
    two separate folders:
        1. Hand images
        2. No-hand images

    The collected dataset can be used for training machine learning
    or computer vision models related to hand detection.

Controls:
    H  -> Capture and save a HAND image
    N  -> Capture and save a NO-HAND image
    Q  -> Exit the application
"""
import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAND_DIR = os.path.join(BASE_DIR, "dataset", "hand")
NO_HAND_DIR = os.path.join(BASE_DIR, "dataset", "no-hand")
os.makedirs(HAND_DIR, exist_ok=True)
os.makedirs(NO_HAND_DIR, exist_ok=True)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera index 0 unavailable. Trying index 1.")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Trying default camera mode.")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened. Please check device availability.")
    exit()
else:
    print("Camera initialized successfully.")
hand_count = len(os.listdir(HAND_DIR))
nohand_count = len(os.listdir(NO_HAND_DIR))

print("\nApplication Controls:")
print("H -> Save HAND image")
print("N -> Save NO-HAND image")
print("Q -> Quit application\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from camera.")
        break
    frame = cv2.flip(frame, 1)
    cv2.putText(
        frame,
        "H=Hand | N=No-Hand | Q=Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    cv2.imshow("Dataset Capture", frame)
    key = cv2.waitKey(1)
    
    if key in [ord('h'), ord('H')]:
        img_path = os.path.join(HAND_DIR, f"hand_{hand_count}.jpg")

        if cv2.imwrite(img_path, frame):
            print(f"Hand image saved: {img_path}")
            hand_count += 1
        else:
            print(f"Failed to save hand image: {img_path}")
            
    elif key in [ord('n'), ord('N')]:
        img_path = os.path.join(NO_HAND_DIR, f"no_hand_{nohand_count}.jpg")

        if cv2.imwrite(img_path, frame):
            print(f"No-hand image saved: {img_path}")
            nohand_count += 1
        else:
            print(f"Failed to save no-hand image: {img_path}")

    elif key in [ord('q'), ord('Q')]:
        print("Exiting dataset collection.")
        break

cap.release()
cv2.destroyAllWindows()
