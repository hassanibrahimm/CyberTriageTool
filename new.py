import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
from fpdf import FPDF
from datetime import datetime
import os

class VideoTriagePro:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Triage Pro")
        self.video_path = None
        self.output_folder = "video_analysis_output"
        os.makedirs(self.output_folder, exist_ok=True)

        self.video_running = False
        self.cap = None
        self.current_frame = None
        self.frame_images = []  # Stores frame file paths
        self.face_images = []  # Stores face file paths
        self.frame_index = 0
        self.face_index = 0

        self.init_menu()
        self.init_welcome_screen()

    def init_menu(self):
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New Video Analysis", command=self.new_video_analysis)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

    def init_welcome_screen(self):
        self.clear_screen()
        self.welcome_frame = tk.Frame(self.root)
        self.welcome_frame.pack(fill="both", expand=True)
        bg_image = Image.open("background.jpeg")
        bg_image = bg_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()), Image.ANTIALIAS)
        self.bg_image = ImageTk.PhotoImage(bg_image)

        bg_label = tk.Label(self.welcome_frame, image=self.bg_image)
        bg_label.place(relwidth=1, relheight=1)

        welcome_label = tk.Label(
            self.welcome_frame, text="Welcome to Video Triage Pro", font=("Arial", 24), bg="black", fg="white"
        )
        welcome_label.pack(pady=50)

        start_button = tk.Button(
            self.welcome_frame, text="Start", command=self.init_upload_screen, font=("Arial", 16), bg="blue", fg="white"
        )
        start_button.pack()

    def init_upload_screen(self):
        self.clear_screen()
        self.upload_frame = tk.Frame(self.root)
        self.upload_frame.pack(fill="both", expand=True)
        bg_image = Image.open("background.jpeg")
        bg_image = bg_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()), Image.ANTIALIAS)
        self.bg_image = ImageTk.PhotoImage(bg_image)

        bg_label = tk.Label(self.upload_frame, image=self.bg_image)
        bg_label.place(relwidth=1, relheight=1)

        upload_label = tk.Label(
            self.upload_frame, text="Upload a Video", font=("Arial", 24), bg="black", fg="white"
        )
        upload_label.pack(pady=50)

        upload_button = tk.Button(
            self.upload_frame,
            text="Upload Video",
            command=self.upload_video,
            font=("Arial", 16),
            bg="green",
            fg="white",
        )
        upload_button.pack()

    def init_player_screen(self):
        self.clear_screen()
        self.player_frame = tk.Frame(self.root)
        self.player_frame.pack(fill="both", expand=True)

        # Left Panel - Video
        self.video_panel = tk.Label(self.player_frame, bg="black")
        self.video_panel.place(relwidth=0.66, relheight=1)

        # Control Buttons
        control_frame = tk.Frame(self.player_frame, bg="gray")
        control_frame.place(relx=0.0, rely=0.9, relwidth=0.66, relheight=0.1)

        start_stop_button = tk.Button(control_frame, text="Start/Stop", command=self.start_stop_video, bg="blue", fg="white")
        start_stop_button.pack(side="left", padx=10, pady=10)

        prev_button = tk.Button(control_frame, text="Previous (-3s)", command=lambda: self.skip_video(-3), bg="green", fg="white")
        prev_button.pack(side="left", padx=10, pady=10)

        next_button = tk.Button(control_frame, text="Next (+3s)", command=lambda: self.skip_video(3), bg="green", fg="white")
        next_button.pack(side="left", padx=10, pady=10)

        # Right Panel - Metadata and Captures
        self.control_panel = tk.Frame(self.player_frame, bg="lightgray")
        self.control_panel.place(relx=0.66, relwidth=0.34, relheight=1)

        metadata_label = tk.Label(self.control_panel, text="Metadata", font=("Arial", 16), bg="lightgray")
        metadata_label.pack(pady=10)
        self.metadata_text = tk.Text(self.control_panel, height=10, bg="white", wrap="word")
        self.metadata_text.pack(fill="x", padx=10)

        self.alert_label = tk.Label(self.control_panel, text="", font=("Arial", 14), bg="lightgray")
        self.alert_label.pack(pady=20)

        # Frame and Face Navigation
        frame_nav = tk.Frame(self.control_panel, bg="lightgray")
        frame_nav.pack(pady=10)
        prev_frame_button = tk.Button(frame_nav, text="Previous Frame", command=self.prev_frame, bg="blue", fg="white")
        prev_frame_button.pack(side="left", padx=5)
        next_frame_button = tk.Button(frame_nav, text="Next Frame", command=self.next_frame, bg="blue", fg="white")
        next_frame_button.pack(side="left", padx=5)

        face_nav = tk.Frame(self.control_panel, bg="lightgray")
        face_nav.pack(pady=10)
        prev_face_button = tk.Button(face_nav, text="Previous Face", command=self.prev_face, bg="blue", fg="white")
        prev_face_button.pack(side="left", padx=5)
        next_face_button = tk.Button(face_nav, text="Next Face", command=self.next_face, bg="blue", fg="white")
        next_face_button.pack(side="left", padx=5)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.video_path = file_path
            self.extract_frames_and_faces()
            self.init_player_screen()

    def start_stop_video(self):
        if self.video_running:
            self.video_running = False
        else:
            self.video_running = True
            threading.Thread(target=self.play_video).start()

    def play_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        while self.cap.isOpened() and self.video_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (int(self.video_panel.winfo_width()), int(self.video_panel.winfo_height())))
            frame_image = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_panel.configure(image=frame_image)
            self.video_panel.image = frame_image
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)

    def skip_video(self, seconds):
        if self.cap:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + int(fps * seconds))

    def prev_frame(self):
        # Navigate through captured frames
        pass

    def next_frame(self):
        # Navigate through captured frames
        pass

    def prev_face(self):
        # Navigate through captured faces
        pass

    def next_face(self):
        # Navigate through captured faces
        pass

    def extract_frames_and_faces(self):
        # Extract frames and faces
        pass

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTriagePro(root)
    root.mainloop()
