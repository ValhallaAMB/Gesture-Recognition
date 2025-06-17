import sys
import os
import cv2
import numpy as np
import time
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
         # Construct the absolute path to the assets directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_dir, "assets", "logo.jpg")
        if os.path.exists(logo_path):
            self.logoLabel.setPixmap(QPixmap(logo_path))
        else:
            print("Logo not found at:", logo_path)

    def open_main_window(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("./UI/main_screen.ui", self)
        
        self.last_detected_gesture = None
        self.last_detected_time = 0
        self.last_gesture_time = time.time()
        
        # Timers and Threshold
        self.confidence_threshold = 0.99
        self.confirm_word_timer = 2.0
        self.double_letter_timer = 3.0
        self.display_letter_delay_timer = 1.5  
        
        # Set UI elements to match initial values
        self.confidenceSlider.setValue(int(self.confidence_threshold * 100))
        self.confidenceValueLabel.setText(f"{self.confidence_threshold:.2f}")
        self.wordTimerEdit.setText(str(int(self.confirm_word_timer)))
        self.letterTimerEdit.setText(str(int(self.double_letter_timer)))
        
        self.confidenceSlider.valueChanged.connect(self.update_confidence_threshold)
        self.wordTimerEdit.editingFinished.connect(self.update_word_timer)
        self.letterTimerEdit.editingFinished.connect(self.update_letter_timer)
        
        self.confidence_threshold = float(self.confidenceValueLabel.text())
        self.confirm_word_timer = float(self.wordTimerEdit.text())
        self.double_letter_timer = float(self.letterTimerEdit.text())

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
            base_options=BaseOptions(model_asset_path="./models/gesture_recognizer.task"),
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
    
    
    # For confidence threshold slider        
    def update_confidence_threshold(self):
        value = self.confidenceSlider.value() / 100
        self.confidence_threshold = value
        self.confidenceValueLabel.setText(f"{value:.2f}")

    # For word confirmation timer
    def update_word_timer(self):
        try:
            value = float(self.wordTimerEdit.text())
            if value >= 0.5:  
                self.confirm_word_timer = value
            else:
                self.wordTimerEdit.setText(str(self.confirm_word_timer))
        except ValueError:
            self.wordTimerEdit.setText(str(self.confirm_word_timer))

    # For double letter detection timer
    def update_letter_timer(self):
        try:
            value = float(self.letterTimerEdit.text())
            # Must be a positive value
            if value > 0:  
                self.double_letter_timer = value
            else:
                self.letterTimerEdit.setText(str(self.double_letter_timer))
        except ValueError:
            self.letterTimerEdit.setText(str(self.double_letter_timer))

    def display_sentence(self, image, results):
        current_time = time.time()
        gesture_detected = False

        for gestures in results.gestures:
            for gesture in gestures:
                print(f"Detecting: {gesture.category_name} (Confidence: {gesture.score:.5f})")
                if gesture.score > self.confidence_threshold:
                    gesture_detected = True
                    self.last_gesture_time = current_time

                    # Add gesture to text if it's new or after a cooldown
                    if (self.last_detected_gesture != gesture.category_name and 
                        current_time - self.last_detected_time > self.display_letter_delay_timer) or (current_time - self.last_detected_time > self.double_letter_timer):
                        self.text += gesture.category_name
                        self.last_detected_gesture = gesture.category_name
                        self.last_detected_time = current_time
                        print(f"Detected: {gesture.category_name} (Confidence: {gesture.score:.5f})")

        # If no gesture detected and timer expired, process current word
        if not gesture_detected and current_time - self.last_gesture_time > self.confirm_word_timer:
            if self.text:
                corrected_word = self.spell.correction(self.text)
                if corrected_word:
                    self.blob += corrected_word.lower() + " "
                else:
                    # If correction fails, just use lowercase on self.text instead
                    self.blob += self.text.lower() + " "
                self.text = ""
            self.last_gesture_time = current_time


        # Update transcription box UI
        if hasattr(self, "transcriptionTextBox"):
            combined_text = self.blob.string + self.text if self.text else self.blob.string
            self.transcriptionTextBox.setPlainText(combined_text)

            # Scroll to end
            cursor = self.transcriptionTextBox.textCursor()
            cursor.movePosition(cursor.End)
            self.transcriptionTextBox.setTextCursor(cursor)
            self.transcriptionTextBox.ensureCursorVisible()


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
