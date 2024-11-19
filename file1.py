import cv2
import os
import json
import numpy as np
import face_recognition

def extract_faces(video_path, frames_dir, face_dir, known_faces_dir, every=5, alert_threshold=0.6):
    """
    Extract faces from video, save them, and compare with known faces.
    Enhanced with name labels, confidence scores, and metadata logging.
    """
    video_path = os.path.normpath(video_path)
    frames_dir = os.path.normpath(frames_dir)
    face_dir = os.path.normpath(face_dir)
    known_faces_dir = os.path.normpath(known_faces_dir)
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)

    # Open video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video FPS for timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    known_faces = load_known_faces(known_faces_dir)
    metadata = []
    detected_faces_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frames at intervals (every nth frame)
        if frame_count % every == 0:
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
            timestamp = frame_count / fps  # Calculate timestamp in seconds
            formatted_time = f"{int(timestamp // 60)}:{int(timestamp % 60):02d}"
            
            # Add forensic overlays
            overlay_frame = frame.copy()
            cv2.putText(overlay_frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(overlay_frame, f"Time: {formatted_time}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Detect faces in the frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for face_location, face_encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_path = os.path.join(face_dir, f"face_{detected_faces_count:05d}.jpg")
                cv2.imwrite(face_path, face_image)
                detected_faces_count += 1

                # Compare with known faces and get name/confidence
                match_name, confidence = compare_faces(known_faces, face_encoding, alert_threshold)

                # Draw bounding box and name
                cv2.rectangle(overlay_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                label = f"{match_name} ({confidence:.2f})" if match_name else "Unknown"
                cv2.putText(overlay_frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Save metadata
                metadata.append({
                    "frame_id": frame_count,
                    "timestamp": formatted_time,
                    "face_id": detected_faces_count - 1,
                    "name": match_name if match_name else "Unknown",
                    "confidence": confidence,
                    "face_path": face_path
                })

                # Pop-up alert
                if match_name:
                    cv2.imshow("ALERT: Matching Face Detected", overlay_frame)
                    cv2.waitKey(5000)  # Display alert for 5 seconds

            # Save the frame with overlays
            cv2.imwrite(frame_path, overlay_frame)

        frame_count += 1

        # Show the video frame while processing
        cv2.imshow("Video", overlay_frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write metadata to a JSON file
    metadata_path = os.path.join(frames_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {metadata_path}")

    cap.release()
    cv2.destroyAllWindows()

def load_known_faces(known_faces_dir):
    """
    Load and encode known faces from the specified directory.
    Returns a dictionary with names and encodings.
    """
    known_faces = []
    for file_name in os.listdir(known_faces_dir):
        file_path = os.path.join(known_faces_dir, file_name)
        if os.path.isfile(file_path):
            name, _ = os.path.splitext(file_name)
            image = face_recognition.load_image_file(file_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces.append((name, encodings[0]))
    return known_faces

def compare_faces(known_faces, face_encoding, threshold):
    """
    Compare a face encoding against a list of known faces.
    :return: Name and confidence score if a match is found within the threshold.
    """
    for name, known_encoding in known_faces:
        distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
        if distance <= threshold:
            return name, 1 - distance
    return None, None

if __name__ == '__main__':
    video_path = "test.mp4"  # Path to the video file
    frames_dir = "frames"  # Directory to save extracted frames
    face_dir = "faces"  # Directory to save detected faces
    known_faces_dir = "known_faces"  # Directory of known faces

    extract_faces(video_path, frames_dir, face_dir, known_faces_dir, every=5)
