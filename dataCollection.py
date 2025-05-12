import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # initialize webcam capture
detector = HandDetector(maxHands=1)  # initialize hand detector with max 1 hand

offset = 20  # padding around the hand bounding box
imgSize = 300  # standardized image size

folder = "Data/I"  # folder to save the processed images
counter = 0  # counter for saved images


while True:
    success, img = cap.read()  # read frame from webcam
    hands, img = detector.findHands(img)  # detect hands in the frame
    
    if hands:  # if hands are detected
        hand = hands[0]  # get the first hand
        x, y, w, h = hand['bbox']  # extract bounding box coordinates
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255  # create white canvas
        
        # crop the image around the hand with offset
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
        
        imgCropShape = imgCrop.shape
        
        aspectRatio = h/w  # calculate aspect ratio
        
        # resize while maintaining aspect ratio
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            
            wGap = math.ceil((imgSize - wCal)/2)
            
            imgWhite[:, wGap: wCal + wGap] = imgResize
        
        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            
            hGap = math.ceil((imgSize - hCal)/2)
            
            imgWhite[hGap: hCal + hGap, :] = imgResize    
        
        cv2.imshow('ImageCrop', imgCrop)  # display cropped image
        cv2.imshow('ImageWhite', imgWhite)  # display processed image
        
    cv2.imshow("Image", img)  # display original image with hand tracking
    key = cv2.waitKey(1)
    if key == ord("s"):  # save image when 's' key is pressed
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
