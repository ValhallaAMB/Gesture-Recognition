import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# NOTES 
# DATA FOR F MUST BE RECOLLECTED AND RETRAINED

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
        
        imgCropShape = imgCrop.shape
        
        aspectRatio = h/w
        
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            
            wGap = math.ceil((imgSize - wCal)/2)
            
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)
            
            if prediction[index] >= 0.90:  # Confidence metric to be adjusted after each training session to avoid flicker
                pred_confidence_float = prediction[index]
                pred_confidence_string = f"{pred_confidence_float * 100:.2f}%"
                
                cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                cv2.putText(imgOutput, pred_confidence_string, (x + w + 10, y - 40), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)
        
        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            
            hGap = math.ceil((imgSize - hCal)/2)
            
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)
            
            if prediction[index] >= 0.90:  # Confidence metric to be adjusted after each training session to avoid flicker
                pred_confidence_float = prediction[index]
                pred_confidence_string = f"{pred_confidence_float * 100:.2f}%"
                
                cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                cv2.putText(imgOutput, pred_confidence_string, (x + w + 10, y - 40), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)
        
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)
        
    cv2.imshow("Image", imgOutput)  
    key = cv2.waitKey(1)
