import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from datetime import datetime


class NUCESVideoTriage:
    def __init__(self, root):
        self.root = root
        self.root.title("NUCES Video Triage")
        self.root.geometry("1200x800")
        self.root.configure(bg="black")  # Hacker-style black background
        self.video_path = None
        self.output_folder = "video_analysis_output"
        os.makedirs(self.output_folder, exist_ok=True)

        self.video_running = False
        self.cap = None
        self.metadata = None
        self.face_images = []
        self.face_index = 0

        self.init_menu()
        self.init_welcome_screen()

    def init_menu(self):
        menu_bar = tk.Menu(self.root, bg="black", fg="red")
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0, bg="black", fg="red")
        file_menu.add_command(label="New Video Analysis", command=self.init_upload_screen)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

    def init_welcome_screen(self):
        self.clear_screen()
        self.welcome_frame = tk.Frame(self.root, bg="black")
        self.welcome_frame.pack(fill="both", expand=True)

        # Add background images
        image_paths = ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
        images = [Image.open(path).resize((600, 400), Image.Resampling.LANCZOS) for path in image_paths]
        image_photos = [ImageTk.PhotoImage(img) for img in images]

        # Top Background
        top_label = tk.Label(self.welcome_frame, image=image_photos[0], bg="black")
        top_label.image = image_photos[0]
        top_label.place(x=0, y=0, relwidth=1, height=400)

        # Bottom Background
        bottom_label = tk.Label(self.welcome_frame, image=image_photos[1], bg="black")
        bottom_label.image = image_photos[1]
        bottom_label.place(x=0, y=400, relwidth=1, height=400)

        # Add Title Text
        welcome_label = tk.Label(
            self.welcome_frame,
            text="Welcome to NUCES Video Triage",
            font=("Courier", 36, "bold"),
            fg="red",
            bg="black",
        )
        welcome_label.place(relx=0.5, rely=0.3, anchor="center")

        # Add Start Button
        start_button = tk.Button(
            self.welcome_frame,
            text="Start",
            command=self.init_upload_screen,
            font=("Courier", 20, "bold"),
            bg="red",
            fg="black",
            activebackground="#8B0000",  # Darker red
            relief="raised",
            bd=5,
            padx=30,
            pady=10,
        )
        start_button.place(relx=0.5, rely=0.5, anchor="center")

    def init_upload_screen(self):
        self.clear_screen()
        self.upload_frame = tk.Frame(self.root, bg="black")
        self.upload_frame.pack(fill="both", expand=True)

        # Left Image
        left_image = Image.open("3.jpg").resize((400, 800), Image.Resampling.LANCZOS)
        left_photo = ImageTk.PhotoImage(left_image)
        left_label = tk.Label(self.upload_frame, image=left_photo, bg="black")
        left_label.image = left_photo
        left_label.place(x=0, y=0, width=400, height=800)

        # Right Image
        right_image = Image.open("4.jpg").resize((400, 800), Image.Resampling.LANCZOS)
        right_photo = ImageTk.PhotoImage(right_image)
        right_label = tk.Label(self.upload_frame, image=right_photo, bg="black")
        right_label.image = right_photo
        right_label.place(x=800, y=0, width=400, height=800)

        # Center Label and Button
        upload_label = tk.Label(
            self.upload_frame,
            text="Upload a Video",
            font=("Courier", 28, "bold"),
            fg="red",
            bg="black",
        )
        upload_label.place(relx=0.5, rely=0.3, anchor="center")

        upload_button = tk.Button(
            self.upload_frame,
            text="Upload Video",
            command=self.upload_video,
            font=("Courier", 16, "bold"),
            bg="red",
            fg="black",
            activebackground="#8B0000",
            relief="raised",
            bd=5,
            padx=30,
            pady=10,
        )
        upload_button.place(relx=0.5, rely=0.5, anchor="center")

    def init_player_screen(self):
        self.clear_screen()
        self.player_frame = tk.Frame(self.root, bg="black")
        self.player_frame.pack(fill="both", expand=True)

        # Video Section
        video_section = tk.Frame(self.player_frame, bg="red", relief="raised", bd=2)
        video_section.place(relx=0.1, rely=0.1, relwidth=0.5, relheight=0.6)

        video_label = tk.Label(video_section, text="Video", font=("Courier", 18, "bold"), bg="red", fg="black")
        video_label.pack(anchor="n", pady=10)

        self.video_panel = tk.Label(video_section, bg="black")
        self.video_panel.pack(fill="both", expand=True, padx=10, pady=10)

        # Control Section
        self.control_section = tk.Frame(self.player_frame, bg="red", relief="ridge", bd=2)
        self.control_section.place(relx=0.65, rely=0.1, relwidth=0.3, relheight=0.8)

        control_label = tk.Label(
            self.control_section, text="Controls", font=("Courier", 18, "bold"), bg="red", fg="black"
        )
        control_label.pack(anchor="n", pady=10)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.video_path = file_path
            self.extract_metadata_and_faces()
            self.init_player_screen()
            self.play_video()

    def play_video(self):
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.video_running = True
            self.process_video()

    def process_video(self):
        if self.cap and self.video_running:
            ret, frame = self.cap.read()
            if ret:
                # Convert to grayscale for face detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces using Haar Cascade
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

                # Highlight detected faces with rectangles
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle for face detection

                # Convert the frame to RGB and resize for GUI display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (int(self.video_panel.winfo_width()), int(self.video_panel.winfo_height())))
                frame_image = ImageTk.PhotoImage(Image.fromarray(frame))

                # Display the frame in the video panel
                self.video_panel.configure(image=frame_image)
                self.video_panel.image = frame_image

                # Continue processing the next frame
                self.root.after(10, self.process_video)
            else:
                self.cap.release()
                self.display_metadata_button()

    def display_metadata_button(self):
        metadata_button = tk.Button(
            self.control_section,
            text="View Metadata",
            command=self.display_metadata,
            font=("Courier", 16, "bold"),
            bg="red",
            fg="black",
            activebackground="#8B0000",
            relief="raised",
            bd=5,
            padx=20,
            pady=10,
        )
        metadata_button.pack(pady=10)

    def display_metadata(self):
        self.clear_panel()
        metadata_text = tk.Text(
            self.control_section, height=10, wrap="word", bg="black", fg="red", font=("Courier", 12)
        )
        metadata_text.pack(fill="x", padx=10, pady=5)

        if self.metadata:
            metadata_text.insert(tk.END, "Video Metadata:\n")
            for key, value in self.metadata.items():
                metadata_text.insert(tk.END, f"{key}: {value}\n")


        faces_button = tk.Button(
            self.control_section,
            text="View Faces",
            command=self.display_faces_buttons,
            font=("Courier", 16, "bold"),
            bg="red",
            fg="black",
            activebackground="#8B0000",
            relief="raised",
            bd=5,
            padx=20,
            pady=10,
        )
        faces_button.pack(pady=10)

    def display_faces_buttons(self):
        self.clear_panel()

        prev_face_button = tk.Button(
            self.control_section,
            text="Previous Face",
            command=self.prev_face,
            font=("Courier", 16, "bold"),
            bg="red",
            fg="black",
            activebackground="#8B0000",
            relief="raised",
            bd=5,
            padx=20,
            pady=10,
        )
        prev_face_button.pack(side="left", padx=10)

        next_face_button = tk.Button(
            self.control_section,
            text="Next Face",
            command=self.next_face,
            font=("Courier", 16, "bold"),
            bg="red",
            fg="black",
            activebackground="#8B0000",
            relief="raised",
            bd=5,
            padx=20,
            pady=10,
        )
        next_face_button.pack(side="left", padx=10)

    def extract_metadata_and_faces(self):
        self.metadata = self.extract_metadata(self.video_path)
        self.face_images = self.extract_faces(self.video_path)

    def extract_metadata(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return {}

        file_stats = os.stat(video_path)
        creation_time = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        modification_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        access_time = datetime.fromtimestamp(file_stats.st_atime).strftime('%Y-%m-%d %H:%M:%S')

        metadata = {
            "Resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            "Frame Rate": f"{cap.get(cv2.CAP_PROP_FPS):.2f} fps",
            "Total Frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "Duration": f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)):.2f} seconds",
            "Date Created": creation_time,
            "Date Modified": modification_time,
            "Date Accessed": access_time,
            "Video Path": video_path,
        }
        cap.release()
        return metadata

    def extract_faces(self, video_path):
        cap = cv2.VideoCapture(video_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_images = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 5 == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                for i, (x, y, w, h) in enumerate(faces):
                    face_image = frame[y:y + h, x:x + w]
                    face_path = os.path.join(self.output_folder, f"face_{frame_count}_{i}.jpg")
                    cv2.imwrite(face_path, face_image)
                    face_images.append(face_path)
            frame_count += 1

        cap.release()
        return face_images

    def prev_face(self):
        if self.face_images and self.face_index > 0:
            self.face_index -= 1
            self.display_face_image()

    def next_face(self):
        if self.face_images and self.face_index < len(self.face_images) - 1:
            self.face_index += 1
            self.display_face_image()

    def display_face_image(self):
        if self.face_images:
            face_image_path = self.face_images[self.face_index]
            face_image = Image.open(face_image_path)
            face_image = face_image.resize((300, 300), Image.Resampling.LANCZOS)
            face_image = ImageTk.PhotoImage(face_image)
            self.video_panel.config(image=face_image)
            self.video_panel.image = face_image

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def clear_panel(self):
        for widget in self.control_section.winfo_children():
            widget.destroy()

    def animate_button(self, button):
        def hover_in(event):
            button.configure(bg="#8B0000", fg="white")

        def hover_out(event):
            button.configure(bg="red", fg="black")

        button.bind("<Enter>", hover_in)
        button.bind("<Leave>", hover_out)


if __name__ == "__main__":
    root = tk.Tk()
    app = NUCESVideoTriage(root)
    root.mainloop()
