import numpy as np
import os
import cv2
from keras.models import model_from_json

os.chdir("C:/Users/soham/OneDrive/Desktop/face_detector")

emotions = {0:'Angry', 1:'Disgusted', 2:'Fearful', 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"}

json_file = open('emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

colors = [(0,0,255),(0,255,0),(255,0,143),(0,255,255),(128,128,128),(255,0,0),(0,165,255)]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in num_faces:
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), colors[maxindex], 4)

        cv2.putText(frame, emotions[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[maxindex], 2, cv2.LINE_AA)
        
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()