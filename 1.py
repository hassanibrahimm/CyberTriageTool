import cv2
import os
from rich.console import Console
from rich.table import Table
from datetime import datetime

def get_file_metadata(file_path):
    """Retrieve file metadata such as creation, modification, and access times."""
    file_stats = os.stat(file_path)
    created_time = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
    modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    accessed_time = datetime.fromtimestamp(file_stats.st_atime).strftime('%Y-%m-%d %H:%M:%S')
    return created_time, modified_time, accessed_time

def is_unique_face(new_face, saved_faces, threshold=0.3):
    """Check if a detected face is unique based on position and size."""
    x, y, w, h = new_face
    for (sx, sy, sw, sh) in saved_faces:
        if abs(x - sx) < w * threshold and abs(y - sy) < h * threshold and abs(w - sw) < w * threshold and abs(h - sh) < h * threshold:
            return False
    return True

def non_max_suppression(faces, overlap_thresh=0.5):
    """Eliminate overlapping bounding boxes."""
    if len(faces) == 0:
        return []

    # Sort boxes by area (largest to smallest)
    boxes = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    picked = []

    while boxes:
        current = boxes.pop(0)
        picked.append(current)
        boxes = [box for box in boxes if not is_overlapping(current, box, overlap_thresh)]

    return picked

def is_overlapping(box1, box2, threshold):
    """Check if two boxes overlap."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the area of the intersection
    overlap_x1 = max(x1, x2)
    overlap_y1 = max(y1, y2)
    overlap_x2 = min(x1 + w1, x2 + w2)
    overlap_y2 = min(y1 + h1, y2 + h2)
    intersection_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)

    area1 = w1 * h1
    area2 = w2 * h2
    overlap_ratio = intersection_area / min(area1, area2)
    return overlap_ratio > threshold

def extract_video_metadata_and_detect_faces(video_path, output_folder):
    console = Console()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        console.print(f"[bold red]Error:[/bold red] Could not open video file: {video_path}")
        return

    # Extract video metadata
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Extract file metadata
    created_time, modified_time, accessed_time = get_file_metadata(video_path)

    # Display metadata
    metadata_table = Table(title="Video Metadata", style="bold cyan")
    metadata_table.add_column("Property", style="bold magenta")
    metadata_table.add_column("Value", style="bold green")
    metadata_table.add_row("Resolution", f"{frame_width}x{frame_height}")
    metadata_table.add_row("Frame Rate", f"{fps:.2f} fps")
    metadata_table.add_row("Total Frames", str(total_frames))
    metadata_table.add_row("Duration", f"{duration:.2f} seconds")
    metadata_table.add_row("Date Created", created_time)
    metadata_table.add_row("Date Modified", modified_time)
    metadata_table.add_row("Date Accessed", accessed_time)
    console.print(metadata_table)

    # Prepare output folder
    video_name = os.path.basename(video_path).split('.')[0]
    metadata_file = os.path.join(output_folder, f"{video_name}_metadata.txt")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Write metadata to file
    with open(metadata_file, "w") as file:
        file.write("Video Metadata:\n")
        file.write(f"Resolution: {frame_width}x{frame_height}\n")
        file.write(f"Frame Rate: {fps:.2f} fps\n")
        file.write(f"Total Frames: {total_frames}\n")
        file.write(f"Duration: {duration:.2f} seconds\n")
        file.write(f"Date Created: {created_time}\n")
        file.write(f"Date Modified: {modified_time}\n")
        file.write(f"Date Accessed: {accessed_time}\n")
        file.write("\nDetected Faces:\n")

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize trackers for unique faces
    frame_count = 0
    saved_count = 0
    saved_faces = []  # Store unique detected faces
    face_table = Table(title="Detected Faces", style="bold cyan")
    face_table.add_column("Face ID", style="bold magenta")
    face_table.add_column("Position (x, y, w, h)", style="bold green")

    # Process each frame for face detection
    cv2.namedWindow(f'Face Detection - {video_name}', cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))
        faces = non_max_suppression(faces)

        # Save only unique faces
        for (x, y, w, h) in faces:
            if is_unique_face((x, y, w, h), saved_faces):
                saved_faces.append((x, y, w, h))
                face = frame[y:y+h, x:x+w]
                face_path = os.path.join(output_folder, f"{video_name}_face_{saved_count:04d}.jpg")
                cv2.imwrite(face_path, face)
                saved_count += 1

                # Save face details in metadata file
                with open(metadata_file, "a") as file:
                    file.write(f"Face {saved_count}: Position (x={x}, y={y}, w={w}, h={h})\n")

                face_table.add_row(str(saved_count), f"({x}, {y}, {w}, {h})")

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow(f'Face Detection - {video_name}', frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            console.print("[bold green]Exiting video display.[/bold green]")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Display face detection results
    console.print(face_table)
    console.print(f"[bold green]Total frames processed:[/bold green] {frame_count}")
    console.print(f"[bold green]Total unique faces saved:[/bold green] {saved_count}")

# Example usage for a single video
video_paths = [
    "C:\\Users\\Administrator\\Desktop\\Forensics\\1.mp4",
]
output_folder = "output_data"

# Process each video
for video_path in video_paths:
    extract_video_metadata_and_detect_faces(video_path, output_folder)
