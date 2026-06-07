import face_recognition
import cv2
import os

# ---- STEP 1: Automatically load all images from folder ----
known_encodings = []
known_names = []

picture_folder = "picture" # chagr that picture with your picture stored folder or directory 

for filename in os.listdir(picture_folder):
    if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
        
        # Get name from filename
        # "roni_sarkar.jpeg" → "Roni Sarkar"
        name = os.path.splitext(filename)[0]  # Remove extension
        name = name.replace("_", " ")          # Replace _ with space
        name = name.title()                    # Capitalize each word

        # Load and encode image
        image_path = os.path.join(picture_folder, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Make sure face is found in image
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"✅ Loaded: {name}")
        else:
            print(f" No face found in: {filename}")

print(f"\n✅ Total {len(known_names)} faces loaded!")

# ---- STEP 2: Open camera ----
video_capture = cv2.VideoCapture(0)
print("Camera started! Press Q to quit.")

while True:
    ret, video = video_capture.read()
    frame=cv2.flip(video,1)

    # Resize and convert
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # Compare with all known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        # Find best match
        best_index = distances.argmin()

        # Scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Show name if matched
        if matches[best_index]:
            label = known_names[best_index]
            color = (0, 255, 0)   # Green
        else:
            label = "Unknown"
            color = (0, 0, 255)   # Red

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Show name above box
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
