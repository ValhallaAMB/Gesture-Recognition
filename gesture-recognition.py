# To do list: add some more comments on the variables you used so you dont fuck up your brain letter when explaining things
#             add more comments on the functions and if statements you added
#             clean your code  




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

# last_hand_position = None # to track the previous hand posiiton

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
    # global last_hand_position
    current_time = time.time()
    gesture_detected = False
    
    # Extracting gesture names and scores
    for gestures in results.gestures:
        for gesture in gestures:
            # Ensure the gesture score is above a threshold (e.g., 0.99)
            if gesture.score > 0.99:
                gesture_detected = True
                last_gesture_time = current_time
                
                # Add gesture to text only if cooldown period has elapsed or it's a different gesture
                if (
                    last_detected_gesture != gesture.category_name
                    or current_time - last_detected_time > 3    # wait 3 seconds befor displaying duplicate letters, increase this if you are slow in fingerspelling
                ):
                    text += gesture.category_name
                    last_detected_gesture = gesture.category_name
                    last_detected_time = current_time
                print(gesture.category_name, gesture.score)

    # If 2 seconds pass and no gesture is being detected, add a space to spearate words
    if not gesture_detected and current_time - last_gesture_time > 2:
        if text:
            corrected_word = spell.correction(text).lower()
            blob += corrected_word + " "
            text = ""
        last_gesture_time = current_time     
    
    #Sentence spelling and correction
    if len(blob) > 100:
        blob = blob[-100:]
        blob.correct()
    elif blob.string.count(" ") >= 2:
        blob.correct()
    elif blob.string.count(" ") >= 1 and current_time - last_gesture_time> 2: # this is for singlular letters such as "I" and "a" in sentences, increase this if you are slow in fingerspelling
        blob.correct()
    
    if current_time - last_gesture_time >= 5:
        blob = tb("")
        text = ""
    

    # Display the recognized sentence on the image
    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(
        image,
        blob.string + text if len(text) > 0 else blob.string, # if you dont include .string the application will crash because it cant convert the text into a string by itself
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
        base_options=BaseOptions(model_asset_path="models/new_HP4_gesture_recognizer.task"),
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
