import zipfile
import os
import face_recognition
import cv2
import numpy as np

# Step 1: Unzip 'input.zip' to 'input/' directory
zip_path = "input.zip"
extract_to = "input"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Extracted '{zip_path}' to '{extract_to}/'")

# Step 2: Load known faces
known_face_encodings = []
known_face_names = []

known_dir = "input/known_faces"

for filename in os.listdir(known_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(known_dir, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            print(f"Warning: No face found in {filename}")

# Step 3: Open mobile camera or webcam
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with URL (IP Webcam)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name = "Unknown"
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale back face locations to original size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show video
    cv2.imshow('Video - Press Q to Exit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
