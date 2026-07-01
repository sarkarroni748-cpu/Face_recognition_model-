import cv2
import os
import time
import platform
import threading
import numpy as np
import db_manager as db


from datetime import datetime
from collections import defaultdict
from insightface.app import FaceAnalysis


# ============================================================
# SECTION 1: VOICE
# ============================================================

def speak(text):
    system = platform.system()

    if system == "Darwin":
        os.system(f"say '{text}'")

    elif system == "Linux":
        os.system(f"espeak '{text}'")

    elif system == "Windows":
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


def speak_all(names):
    if not names:
        speak("No faces detected")
        return

    for name in sorted(names):
        print(f"🔊 Speaking: {name}")
        speak(f"{name} present")


# ============================================================
# SECTION 2: INSIGHTFACE
# ============================================================

print("\n⏳ Loading InsightFace...\n")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))

print("✅ InsightFace Ready!\n")


# ============================================================
# SECTION 3: GLOBALS
# ============================================================

SIMILARITY_THRESHOLD = 0.55

label_map = {}
label_lock = threading.Lock()

thread_running = False
thread_lock = threading.Lock()


# ============================================================
# SECTION 4: COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ============================================================
# SECTION 5: LOAD STUDENTS
# ============================================================

def load_known_faces(picture_folder="picture"):
    if not os.path.exists(picture_folder):
        print(f"❌ ERROR: Folder '{picture_folder}' not found!")
        exit()

    students = defaultdict(list)
    total_images = 0

    for filename in os.listdir(picture_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(picture_folder, filename)
        base_name = os.path.splitext(filename)[0]

        # Example: roni_1.jpg -> "Roni"
        if "_" in base_name:
            student_name = "_".join(base_name.split("_")[:-1])
        else:
            student_name = base_name

        student_name = student_name.replace("_", " ").title()

        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Cannot read: {filename}")
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"⚠️ No face found: {filename}")
            continue

        embedding = faces[0].embedding
        students[student_name].append(embedding)
        total_images += 1

        print(f"✅ Loaded: {filename} -> {student_name}")

    print("\n====================")
    print(f"Total Images: {total_images}")
    print(f"Total Students: {len(students)}")
    print("====================\n")

    if len(students) == 0:
        print("❌ No valid student images found!")
        exit()

    return students


# ============================================================
# SECTION 6: RECOGNITION ENGINE
# ============================================================

def recognize_faces(frame, students):
    results = []
    faces = app.get(frame)

    if len(faces) == 0:
        return results

    for face in faces:
        embedding = face.embedding
        best_name = "Unknown"
        best_score = -1

        for student_name, known_embeddings in students.items():
            scores = [
                cosine_similarity(embedding, known_emb)
                for known_emb in known_embeddings
            ]
            avg_score = np.mean(scores)

            if avg_score > best_score:
                best_score = avg_score
                best_name = student_name

        if best_score < SIMILARITY_THRESHOLD:
            best_name = "Unknown"

        x1, y1, x2, y2 = face.bbox.astype(int)

        results.append({
            "name": best_name,
            "score": float(best_score),
            "bbox": (x1, y1, x2, y2)
        })

    return results


# ============================================================
# SECTION 7: THREAD WORKER
# ============================================================

def run_recognition(frame, students):
    global thread_running

    try:
        results = recognize_faces(frame, students)

        new_map = {
            result["bbox"]: {
                "name": result["name"],
                "score": result["score"]
            }
            for result in results
        }

        with label_lock:
            label_map.clear()
            label_map.update(new_map)

    except Exception as e:
        print(f"\n❌ Recognition Error: {e}")

    finally:
        with thread_lock:
            thread_running = False


# ============================================================
# SECTION 8: MAIN
# ============================================================

PICTURE_FOLDER = "picture"
DETECTION_TIME = 10
THREAD_INTERVAL = 1.0

students = load_known_faces(PICTURE_FOLDER)
db.initialize_database()
speak("Attention please")

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FPS, 30)

if not video_capture.isOpened():
    print("❌ ERROR: Cannot open camera!")
    exit()

print("📷 Camera started! Press Q to quit.")
print(f"⏱ Collecting faces for {DETECTION_TIME} seconds...\n")

fps = 0
prev_frame_time = time.time()
detected_names = set()
cycle_start = time.time()
last_thread_time = 0

while True:
    ret, video = video_capture.read()

    if not ret:
        print("❌ Camera read failed.")
        break

    frame = cv2.flip(video, 1)
    current_time = time.time()

    fps = 1 / (current_time - prev_frame_time + 1e-9)
    prev_frame_time = current_time

    elapsed = current_time - cycle_start

    # ========================================================
    # Launch recognition thread
    # ========================================================

    with thread_lock:
        is_running = thread_running

    if not is_running and (current_time - last_thread_time) >= THREAD_INTERVAL:
        frame_copy = frame.copy()

        with thread_lock:
            thread_running = True

        t = threading.Thread(
            target=run_recognition,
            args=(frame_copy, students),
            daemon=True
        )
        t.start()
        last_thread_time = current_time

    # ========================================================
    # Draw Results
    # ========================================================

    with label_lock:
        current_results = dict(label_map)

    for bbox, data in current_results.items():
        x1, y1, x2, y2 = bbox
        name = data["name"]
        score = data["score"]

        if name == "Unknown":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

            if name not in detected_names:
                detected_names.add(name)
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] 👤 Detected: {name} ({score:.2f})")

        label = f"{name} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 12),
            (x1 + text_w + 10, y1),
            color,
            -1
        )

        cv2.putText(
            frame, label,
            (x1 + 3, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 0, 0), 2
        )

    # ========================================================
    # UI OVERLAYS
    # ========================================================

    remaining = max(0, int(DETECTION_TIME - elapsed))

    cv2.putText(frame, f"Speaking in: {remaining}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Detected: {len(detected_names)}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    # ========================================================
    # Time finished
    # ========================================================

    if elapsed >= DETECTION_TIME:
        print("\n=== ⏰ Time is up! ===\n")

        video_capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Flush window events (prevents freeze on Mac)
        time.sleep(0.5)

        print("Detected names:\n")
        for name in sorted(detected_names):
            print(f"✅ {name}")

        print()
        speak_all(detected_names)
        for name in detected_names:
            db.mark_attendance(name)
        print("\n=== DONE ===\n")
        exit()

    cv2.imshow("InsightFace Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
cv2.waitKey(1)  # Flush window events on q-quit
# finish