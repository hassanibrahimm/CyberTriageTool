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
        self.root.configure(bg="black")
        self.video_path = None
        self.output_folder = "video_analysis_output"
        os.makedirs(self.output_folder, exist_ok=True)

        self.video_running = False
        self.cap = None
        self.metadata = None
        self.face_images = []
        self.face_index = 0

        self.init_menu()
        self.init_welcome_animation()

    def init_menu(self):
        menu_bar = tk.Menu(self.root, bg="black", fg="red")
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0, bg="black", fg="red")
        file_menu.add_command(label="New Video Analysis", command=self.init_upload_screen)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

    def init_welcome_animation(self):
        self.clear_screen()
        self.welcome_frame = tk.Frame(self.root, bg="black")
        self.welcome_frame.pack(fill="both", expand=True)

        # Background Images
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

        # Animated Title
        self.title_label = tk.Label(
            self.welcome_frame,
            text="",
            font=("Courier", 36, "bold"),
            fg="red",
            bg="black",
        )
        self.title_label.place(relx=0.5, rely=0.3, anchor="center")
        self.animate_text("Welcome to NUCES Video Triage")

        # Start Button
        start_button = tk.Button(
            self.welcome_frame,
            text="Start",
            command=self.init_upload_screen,
            font=("Courier", 20, "bold"),
            bg="red",
            fg="black",
            activebackground="#8B0000",
            relief="raised",
            bd=5,
            padx=30,
            pady=10,
        )
        start_button.place(relx=0.5, rely=0.5, anchor="center")

    def animate_text(self, text, index=0):
        if index < len(text):
            self.title_label.config(text=self.title_label.cget("text") + text[index])
            self.root.after(100, self.animate_text, text, index + 1)

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

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.video_path = file_path
            self.extract_metadata_and_faces()
            self.init_video_screen()
            self.play_video()

    def init_video_screen(self):
        self.clear_screen()
        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.pack(fill="both", expand=True)

        video_border = tk.Frame(self.video_frame, bg="red", bd=5)
        video_border.place(relx=0.5, rely=0.4, anchor="center", width=820, height=470)

        self.video_panel = tk.Label(video_border, bg="black")
        self.video_panel.place(x=10, y=10, width=800, height=450)

    def play_video(self):
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.video_running = True
            self.process_video()

    def process_video(self):
        if self.cap and self.video_running:
            ret, frame = self.cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (800, 450))
                frame_image = ImageTk.PhotoImage(Image.fromarray(frame))

                self.video_panel.configure(image=frame_image)
                self.video_panel.image = frame_image

                self.root.after(10, self.process_video)
            else:
                self.cap.release()
                self.prompt_buttons()

    def prompt_buttons(self):
        self.clear_screen()
        prompt_label = tk.Label(
            self.root,
            text="What would you like to analyze?",
            font=("Courier", 28, "bold"),
            fg="red",
            bg="black",
        )
        prompt_label.place(relx=0.5, rely=0.3, anchor="center")

        metadata_button = tk.Button(
            self.root,
            text="View Metadata",
            command=self.display_metadata,
            font=("Courier", 16, "bold"),
            bg="red",
            fg="black",
            activebackground="#8B0000",
            relief="raised",
            bd=5,
            padx=30,
            pady=10,
        )
        metadata_button.place(relx=0.4, rely=0.5, anchor="center")

        faces_button = tk.Button(
            self.root,
            text="View Detected Faces",
            command=self.display_faces,
            font=("Courier", 16, "bold"),
            bg="red",
            fg="black",
            activebackground="#8B0000",
            relief="raised",
            bd=5,
            padx=30,
            pady=10,
        )
        faces_button.place(relx=0.6, rely=0.5, anchor="center")

    def display_metadata(self):
        self.clear_screen()
        metadata_text = tk.Text(self.root, font=("Courier", 16), bg="black", fg="red", wrap="word")
        metadata_text.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

        if self.metadata:
            metadata_text.insert(tk.END, "Video Metadata:\n\n")
            for key, value in self.metadata.items():
                metadata_text.insert(tk.END, f"{key}: {value}\n")

    def display_faces(self):
        self.clear_screen()
        if self.face_images:
            face_label = tk.Label(
                self.root,
                text="Detected Faces",
                font=("Courier", 28, "bold"),
                fg="red",
                bg="black",
            )
            face_label.place(relx=0.5, rely=0.1, anchor="center")

            image_canvas = tk.Canvas(self.root, bg="black")
            image_canvas.place(relx=0.1, rely=0.2, relwidth=0.8, relheight=0.7)

            for idx, face_path in enumerate(self.face_images[:5]):  # Display first 5 faces
                face_image = Image.open(face_path).resize((150, 150), Image.Resampling.LANCZOS)
                face_photo = ImageTk.PhotoImage(face_image)
                x_pos = 100 + idx * 200
                y_pos = 100
                image_canvas.create_image(x_pos, y_pos, image=face_photo)

    def extract_metadata_and_faces(self):
        self.metadata = self.extract_metadata(self.video_path)
        self.face_images = self.extract_faces(self.video_path)

    def extract_metadata(self, video_path):
        cap = cv2.VideoCapture(video_path)
        file_stats = os.stat(video_path)
        metadata = {
            "Resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            "Frame Rate": f"{cap.get(cv2.CAP_PROP_FPS):.2f} fps",
            "Total Frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "Duration": f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)):.2f} seconds",
            "Date Created": datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
            "Date Modified": datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            "Date Accessed": datetime.fromtimestamp(file_stats.st_atime).strftime('%Y-%m-%d %H:%M:%S'),
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
                for (x, y, w, h) in faces:
                    face_image = frame[y:y + h, x:x + w]
                    face_path = os.path.join(self.output_folder, f"face_{frame_count}.jpg")
                    cv2.imwrite(face_path, face_image)
                    face_images.append(face_path)
            frame_count += 1
        cap.release()
        return face_images

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = NUCESVideoTriage(root)
    root.mainloop()
