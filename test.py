import cv2
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

# Webcam & Models
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    r"C:\Users\hima1\Downloads\converted_keras (2)\keras_model.h5",
    r"C:\Users\hima1\Downloads\converted_keras (2)\labels.txt"
)

offset = 20
imgSize = 300
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# --- Chatbot Structure ---
messages = []      # list of (sender, text) → e.g. ("You", "HELLO")
sentence = ""      # building typed text
last_letter = ""
last_time = 0
delay = 1.0  # sec between accepted letters

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        current_letter = labels[index]

        # Debounce logic
        if current_letter != last_letter and (time.time() - last_time) > delay:
            sentence += current_letter
            last_letter = current_letter
            last_time = time.time()

        # Show letter on bounding box
        cv2.putText(imgOutput, current_letter, (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 4)

    # --- Chat UI ---
    chat_area_height = 200
    h, w, _ = imgOutput.shape
    cv2.rectangle(imgOutput, (0, h - chat_area_height), (w, h), (240, 240, 240), cv2.FILLED)

    # Show past messages
    y_offset = h - chat_area_height + 30
    for sender, msg in messages[-5:]:  # show last 5 messages
        text = f"{sender}: {msg}"
        cv2.putText(imgOutput, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y_offset += 30

    # Show current typing sentence
    cv2.putText(imgOutput, "You: " + sentence, (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)

    cv2.imshow("Sign Chatbot Writer", imgOutput)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # ENTER key → send message
        if sentence.strip():
            messages.append(("You", sentence))
            # Placeholder for chatbot response (you will add API later)
            messages.append(("Bot", "<<< chatbot reply here >>>"))
            sentence = ""  # reset typed text
    elif key == 27:  # ESC → exit
        break

cap.release()
cv2.destroyAllWindows()
