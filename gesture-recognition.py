from pprint import pprint
import cv2 # used for video capture and image processing
import numpy as np
import mediapipe as mp # used for hand tracking and gesture recognition
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
) # used for gesture recognition
from mediapipe.tasks.python import BaseOptions # used for base options
from mediapipe.framework.formats import landmark_pb2 # used for landmark data
from textblob import TextBlob as tb # used for text processing
from spellchecker import SpellChecker # used for spell checking
import time # used for time and time extraction

mp_hands = mp.solutions.hands # mediapipe hands module
mp_drawing = mp.solutions.drawing_utils # mediapipe drawing module
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe drawing styles module

spell = SpellChecker() # spell checker object

text = "" # string to store text
blob = tb("") # TextBlob object to store text
last_gesture_time = time.time() #this will help create spaces using timing of the last detected gesture

last_detected_gesture = None  # To store the last detected gesture
last_detected_time = 0  # To store the time of the last detected gesture

# Function to draw hand landmarks on the image
def draw_landmarks(image, results) -> None:
    for hand_landmarks in results.hand_landmarks:
        # Create a NormalizedLandmarkList to hold the hand landmarks in MediaPipe format
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        # Converts the hand landmarks to a list of NormalizedLandmark protobuf objects with x, y, z coordinates.
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )

        # Draw the hand landmarks on the image
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )


# Function to display the recognized sentence on the image
def display_sentence(image, results) -> None:
    global blob, text, last_gesture_time, last_detected_gesture, last_detected_time
    
    current_time = time.time()
    gesture_detected = False
    
    # Extracting gesture names and scores
    for gestures in results.gestures:
        for gesture in gestures:
            # Ensure the gesture score is above a threshold (e.g., 0.99)
            if gesture.score > 0.99:
                gesture_detected = True
                last_gesture_time = current_time
                
                # Add gesture to text only if cooldown period has elapsed or if it's a different gesture
                if (
                    last_detected_gesture != gesture.category_name
                    or current_time - last_detected_time > 3    # wait 3 seconds before displaying duplicate letters, increase this if you are slow in fingerspelling
                ):
                    text += gesture.category_name
                    last_detected_gesture = gesture.category_name
                    last_detected_time = current_time
                print(gesture.category_name, gesture.score)

    # To remove sentences from the display and make way for new sentences, the first if statement has to run first otherwise it wont work
    # if you make a joined if statement with the keyword "and" it doesnt work due to clashes in logic
    if not gesture_detected:
        time_since_last_activity = current_time - last_gesture_time

        # check for the LONGEST timeout first.
        # Has it been over 5 seconds? ok, clear the screen. Increase this timer if you are slow at fingerspelling
        if time_since_last_activity > 5:
            # We only clear if there is actually text on the screen.
            if len(text) > 0 or len(blob.string) > 0:
                print(f"CLEARING TEXT........")
                blob = tb("")
                text = ""
                last_gesture_time = current_time # Reset the timer. This prevents the screen from being cleared on every single frame after the 5-second mark.
        elif time_since_last_activity > 2 and len(text) > 0:
                # Correct the spelling of the word and add it to the main sentence blob.
                corrected_word = spell.correction(text)
                if corrected_word: 
                    blob += corrected_word.lower() + " "
                text = "" 
                # We reset the timer. This marks the end of the word as a new "last activity" point.
                last_gesture_time = current_time
    

    # Display the recognized sentence on the image
    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(
        image,
        blob.string + text if len(text) > 0 else blob.string, # if you dont include ".string" the application will crash because it cannot convert the text into a string by itself
        (3, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    # Load GestureRecognizer model
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path="models/gesture_recognizer8.task"),
        num_hands=1,  # Keep it 1 for now
    )

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Main function to run the GestureRecognizer model with specified options
    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Process frame
            detection_result = recognizer.recognize(mp_image)

            # Draw hand landmarks
            draw_landmarks(frame, detection_result)

            # Display sentence
            display_sentence(frame, detection_result)

            # Show the frame
            cv2.imshow("Hand Gesture Recognition", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
