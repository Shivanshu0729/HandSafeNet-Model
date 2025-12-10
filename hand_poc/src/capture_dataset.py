import cv2
import os

# Base directory of the project (hand_poc folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset folders
HAND_DIR = os.path.join(BASE_DIR, "dataset", "hand")
NO_HAND_DIR = os.path.join(BASE_DIR, "dataset", "no-hand")

# Create folders if they don't exist
os.makedirs(HAND_DIR, exist_ok=True)
os.makedirs(NO_HAND_DIR, exist_ok=True)

# Open webcam (try different indexes in case camera 0 is busy)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera did not open with index 0. Trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Trying default camera mode...")
    cap = cv2.VideoCapture(0)

# If still not opened, exit program
if not cap.isOpened():
    print("Camera FAILED to open. Make sure it is connected or free.")
    exit()
else:
    print("Camera opened successfully!")

# Counters for naming saved images
hand_count = len(os.listdir(HAND_DIR))
nohand_count = len(os.listdir(NO_HAND_DIR))

print("\nInstructions:")
print("Press H → Save HAND image")
print("Press N → Save NO-HAND image")
print("Press Q → Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read camera frame.")
        break

    # Mirror effect so movement feels natural
    frame = cv2.flip(frame, 1)

    # Display instructions on screen
    cv2.putText(frame, "H=Hand  N=NoHand  Q=Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    cv2.imshow("Dataset Capture", frame)
    key = cv2.waitKey(1)

    # Save hand images
    if key in [ord('h'), ord('H')]:
        img_path = os.path.join(HAND_DIR, f"hand_{hand_count}.jpg")
        if cv2.imwrite(img_path, frame):
            print(f"[SAVED HAND] {img_path}")
        else:
            print(f"[ERROR SAVING] {img_path}")
        hand_count += 1

    # Save no-hand images
    elif key in [ord('n'), ord('N')]:
        img_path = os.path.join(NO_HAND_DIR, f"no_hand_{nohand_count}.jpg")
        if cv2.imwrite(img_path, frame):
            print(f"[SAVED NO-HAND] {img_path}")
        else:
            print(f"[ERROR SAVING] {img_path}")
        nohand_count += 1

    # Quit program
    elif key in [ord('q'), ord('Q')]:
        print("Exiting...")
        break

# Clean up camera and windows
cap.release()
cv2.destroyAllWindows()
