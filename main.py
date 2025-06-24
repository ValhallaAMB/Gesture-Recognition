import sys
import cv2
# import numpy as np
import time
import pyttsx3
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QGraphicsScene
from PyQt5 import uic

# Mediapipe Imports
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

from spellchecker import SpellChecker
from textblob import TextBlob as tb


class WelcomeScreen(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/welcome_screen.ui", self)
        self.startButton.clicked.connect(self.open_main_window)

    def open_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/main_screen.ui", self)
        
        # Text-to-speech engine initialization
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 100)
        self.engine.setProperty('volume', 0.75)
        
        self.last_detected_gesture = None
        self.last_detected_time = 0
        self.last_gesture_time = time.time()

        # Connect UI buttons to functions
        self.cameraOnButton.clicked.connect(self.start_camera)
        self.cameraOffButton.clicked.connect(self.stop_camera)
        self.clearButton.clicked.connect(self.clear_transcription)
        self.exitButton.clicked.connect(self.exit_app)
        self.toggleLandmarksButton.clicked.connect(self.toggle_landmarks)

        # Setup QGraphicsScene for video
        self.scene = QGraphicsScene()
        self.videoView.setScene(self.scene)

        # Initialize MediaPipe gesture recognizer
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path="./models/gesture_recognizer22.task"),
            num_hands=1,
        )
        self.recognizer = GestureRecognizer.create_from_options(options)

        # Setup spell checker and TextBlob variables
        self.spell = SpellChecker()
        self.blob = tb("")
        self.text = ""

        # Video capture (will be opened in start_camera)
        self.cap = None

        # Timer for grabbing frames and processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Toggle for displaying landmarks
        self.display_landmarks = False

    def toggle_landmarks(self):
        self.display_landmarks = not self.display_landmarks
        print(f"Landmark display: {'ON' if self.display_landmarks else 'OFF'}")

    def draw_landmarks(self, image, results):
        if not self.display_landmarks:
            return

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        for hand_landmarks in results.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks]
            )
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
            )

    def display_sentence(self, image, results):
        current_time = time.time()
        gesture_detected = False

        for gestures in results.gestures:
            for gesture in gestures:
                print(f"Detecting: {gesture.category_name} (Confidence: {gesture.score:.5f})")
                if gesture.score > 0.995 or gesture.score > 0.95 and gesture.category_name == "Z":
                    gesture_detected = True
                    self.last_gesture_time = current_time

                    if (self.last_detected_gesture != gesture.category_name or 
                        current_time - self.last_detected_time > 3):
                        self.text += gesture.category_name
                        self.last_detected_gesture = gesture.category_name
                        self.last_detected_time = current_time
                        print(f"Detected: {gesture.category_name} (Confidence: {gesture.score:.5f})")

        if not gesture_detected and current_time - self.last_gesture_time > 2:
            if self.text:
                corrected_word = self.spell.correction(self.text).lower()
                self.blob += corrected_word + " "
                self.text = ""
                # Speak the corrected word and wait for it to finish
                self.engine.say(corrected_word)
                self.engine.runAndWait()
            self.last_gesture_time = current_time

        if len(self.blob) > 100:
            self.blob = tb(self.blob.string[-100:])
            self.blob.correct()
        elif self.blob.string.count(" ") >= 2:
            self.blob.correct()
        elif self.blob.string.count(" ") >= 1 and current_time - self.last_gesture_time > 2:
            self.blob.correct()

        if current_time - self.last_gesture_time >= 5:
            self.blob = tb("")
            self.text = ""

        if hasattr(self, "transcriptionTextBox"):
            self.transcriptionTextBox.setText(self.blob.string + self.text if self.text else self.blob.string)


    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Recognize gestures
        results = self.recognizer.recognize(mp_image)

        # Draw landmarks if toggle is ON
        self.draw_landmarks(frame, results)

        # Display recognized sentence and update transcription box
        self.display_sentence(frame, results)

        # Convert frame for PyQt display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.videoView.width(), self.videoView.height(), Qt.KeepAspectRatio
        )
        self.scene.clear()
        self.scene.addPixmap(pixmap)

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open webcam")
            return
        if not self.timer.isActive():
            self.timer.start(30)

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.scene.clear()

    def clear_transcription(self):
        self.text = ""
        self.blob = tb("")
        if hasattr(self, "transcriptionTextBox"):
            self.transcriptionTextBox.clear()

    def exit_app(self):
        self.close()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    with open("./stylesheets/dark_theme.qss", "r") as f:
        app.setStyleSheet(f.read())

    welcome = WelcomeScreen()
    welcome.show()

    sys.exit(app.exec_())
