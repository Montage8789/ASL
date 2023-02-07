import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

#Use webcam
capture = cv2.VideoCapture(0)

#Create hand detector
detector = HandDetector(maxHands=1)

#Help with classifier
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
labels = ["A", "B", "C", "D", "E"]

#Help with constant numbers for photo calculations
offset = 20
imageSize = 300

#Help with saving images
folder = "Images/E"
count = 0

while True:
    success, img = capture.read()
    imageOutput = img.copy()
    hands, img = detector.findHands(img)

    #If there is a hand, show cropped image
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        #Create image windows
        imageWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imageCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        #Put cropped image on white background to keep image size constant
        imageCropShape = imageCrop.shape

        #Used to help with centering cropped image onto white
        aspectRatio = h / w
        if aspectRatio > 1:
            constant = imageSize / h
            calcWidth = math.ceil(constant * w)
            imageResize = cv2.resize(imageCrop, (calcWidth, imageSize))
            imageResizeShape = imageResize.shape
            wGap = math.ceil((imageSize - calcWidth) / 2)
            imageWhite[:, wGap:calcWidth + wGap] = imageResize
            prediction, index = classifier.getPrediction(imageWhite)
        else:
            constant = imageSize / w
            calcHeight = math.ceil(constant * h)
            imageResize = cv2.resize(imageCrop, (imageSize, calcHeight))
            imageResizeShape = imageResize.shape
            hGap = math.ceil((imageSize - calcHeight) / 2)
            imageWhite[hGap:calcHeight + hGap, :] = imageResize
            prediction, index = classifier.getPrediction(imageWhite)

        cv2.putText(imageOutput, labels[index], (x, y - offset), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("Cropped Image", imageCrop)
        cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("Image", imageOutput)
    key = cv2.waitKey(1)

    #press 's' key to save image
    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imageWhite)
        print(count)
