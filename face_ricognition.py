import face_recognition
import cv2
import os
import time
import platform
from datetime import datetime


# ============================================================
# SECTION 1: VOICE — Speak names out loud
# ============================================================

def speak(text):
    system = platform.system()
    if system == "Darwin":         # ✅ MAC — uses built-in say command
        os.system(f"say '{text}'")

    
    elif system == "Windows":
        os.system(f'PowerShell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')


    elif system == "Linux":      # 🐧 LINUX — uses espeak
        os.system(f"espeak '{text}'")


def speak_all(names):
    if not names:
        speak("No faces detected")
        return
    for name in names:
        print(f"🔊 Speaking: {name}")
        speak(f"{name} present")


# ============================================================
# SECTION 2: LOADER — Load face images from folder
# ============================================================

def load_known_faces(picture_folder="picture"):
    known_encodings = []
    known_names = []

    for filename in os.listdir(picture_folder):
        if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):

            # "roni_sarkar.jpeg" → "Roni Sarkar"
            name = os.path.splitext(filename)[0]
            name = name.replace("_", " ")
            name = name.title()

            image_path = os.path.join(picture_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"✅ Loaded: {name}")
            else:
                print(f"❌ No face found in: {filename}")

    print(f"\n✅ Total {len(known_names)} faces loaded!\n")
    return known_encodings, known_names


# ============================================================
# SECTION 3: RECOGNIZER — Detect faces in camera frame
# ============================================================

def recognize_faces(frame, known_encodings, known_names):
    detected_names = []

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_index = distances.argmin()

        top *= 4; right *= 4; bottom *= 4; left *= 4

        if matches[best_index]:
            label = known_names[best_index]
            color = (0, 255, 0)   # Green
            detected_names.append(label)
        else:
            label = "Unknown"
            color = (0, 0, 255)   # Red

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return frame, detected_names


# ============================================================
# SECTION 4: MAIN — Run everything
# ============================================================

PICTURE_FOLDER = "picture"
DETECTION_TIME = 5  # seconds to collect names before speaking

# Load faces
known_encodings, known_names = load_known_faces(PICTURE_FOLDER)

# Announce before camera opens
speak("Attention please")

# Open camera
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 30)
print("📷 Camera started! Press Q to quit.")
print(f"⏱  Collecting faces for {DETECTION_TIME} seconds...\n")

# FPS tracking variables
fps = 0
prev_frame_time = time.time()

detected_names = set()
cycle_start = time.time()

while True:
    ret, video = video_capture.read()
    frame = cv2.flip(video, 1)

    # Calculate accurate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time

    # Recognize faces
    frame, names_in_frame = recognize_faces(frame, known_encodings, known_names)

    # Store new names
    now = time.time()
    elapsed = now - cycle_start

    for name in names_in_frame:
        if name not in detected_names:
            detected_names.add(name)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] 👤 Detected: {name}")

    # Countdown on screen
    remaining = int(DETECTION_TIME - elapsed)
    cv2.putText(frame, f"Speaking in: {remaining}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # FPS on screen
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Time is up — speak all names then exit
    if elapsed >= DETECTION_TIME:
        os.system("clear")  # 🪟 WINDOWS USERS: change "clear" to "cls"
        print("=== ⏰ Time is up! Speaking names... ===\n")


        # First stop camera and close window
        video_capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)        # forces window to close immediately on Mac
        time.sleep(0.5)       # small wait to make sure window is fully gone


        # Then speak names
        print(f"Detected names: {detected_names}")
        speak_all(detected_names)

        print("\n=== ✅ Done! Terminating program... ===\n")

        time.sleep(1)

        # Exit the program
        exit()

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
