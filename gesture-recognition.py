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

mp_hands = mp.solutions.hands # mediapipe hands module
mp_drawing = mp.solutions.drawing_utils # mediapipe drawing module
mp_drawing_styles = mp.solutions.drawing_styles # mediapipe drawing styles module

spell = SpellChecker() # spell checker object

sentence = [] # list to store sentences
text = "" # string to store text
blob = tb("") # TextBlob object to store text

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
    global sentence
    global blob
    global text
    # for gestures in results.gestures:
    #     for gesture in gestures:
    #         if gesture.score > 0.99:
    #             if len(text) > 0:
    #                 if not text.endswith(gesture.category_name):
    #                     text += gesture.category_name
    #             else:
    #                 text += gesture.category_name
    #         # print(gesture.category_name, gesture.score)

    # if not results.gestures and len(text) > 0 and not text.endswith(" "):
    #     sentence.append(spell.correction(text))
    #     text = ""
    #     sentence.append(" ")

    # if len(text) > 25:
    #     text = text[-25:]

    # Extracting gesture names and scores
    for gestures in results.gestures:
        for gesture in gestures:
            # Ensure the gesture score is above a threshold (e.g., 0.99)
            if gesture.score > 0.99:
                if len(text) > 0:
                    if not text.endswith(gesture.category_name):
                        text += gesture.category_name
                else:
                    text += gesture.category_name
            print(gesture.category_name, gesture.score)
    
    # Spelling and sentence correction
    if not results.gestures and len(text) > 0 and not text.endswith(" "):
        text = text.lower()
        blob += spell.correction(text)
        blob += " "
        text = ""

    if len(blob) > 25:
        blob = blob[-25:]
        blob.correct()
    elif blob.string.count(" ") >= 2:
        blob.correct()

    # for gestures in results.gestures:
    #     for gesture in gestures:
    #         if gesture.score > 0.99:
    #             if len(sentence) > 0:
    #                 if sentence[-1] != gesture.category_name:
    #                     sentence.append(gesture.category_name)
    #             else:
    #                 sentence.append(gesture.category_name)
    #         print(gesture.category_name, gesture.score)

    # if not results.gestures and len(sentence) > 0 and sentence[-1] != " ":
    #     sentence.append(" ")

    # if len(sentence) > 25:
    #     sentence = sentence[-25:]

    # Display the recognized sentence on the image
    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
    cv2.putText(
        image,
        # text,
        # blob.string,
        # "".join(sentence),
        # "".join(sentence) + text if len(text) > 0 else "".join(sentence),
        blob.string + text if len(text) > 0 else blob.string,
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
        base_options=BaseOptions(model_asset_path="models/gesture_recognizer22.task"),
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
