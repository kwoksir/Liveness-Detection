import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model graph
json_file = open('models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load antispoofing model weights
model.load_weights('models/antispoofing_model.h5')
cap = cv2.VideoCapture(0)

def fancyDraw(img, bbox, l=20, t=3, rt= 3):
    x, y, w, h = bbox
    x1, y1 = x+w, y+h
    #cv2.rectangle(img, (x + 20, y + 20 , w - 40, h - 40), (0,255,0), 2)

    #Top Left x,y
    cv2.line(img, (x,y), (x+l, y), (255,0,255), t)
    cv2.line(img, (x,y), (x, y+l), (255,0,255), t)

    #Top Right x,y
    cv2.line(img, (x1,y), (x1-l, y), (255,0,255), t)
    cv2.line(img, (x1,y), (x1, y+l), (255,0,255), t)

    #Bottom Left x,y
    cv2.line(img, (x,y1), (x+l, y1), (255,0,255), t)
    cv2.line(img, (x,y1), (x, y1-l), (255,0,255), t)

    #Bottom Right x,y
    cv2.line(img, (x1,y1), (x1-l, y1), (255,0,255), t)
    cv2.line(img, (x1,y1), (x1, y1-l), (255,0,255), t)

    return img

while True:
    try:
        ret,img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)
        for (x,y,w,h) in faces:
            bbox = (x,y,w,h)
            face = img[y-5:y+h+5,x-5:x+w+5]
            resized_face = cv2.resize(face,(160,160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)

            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model.predict(resized_face,verbose=0)[0]
            #print(preds)
            if preds > 0.005:
                label = 'Fake'
                cv2.putText(img,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                img = fancyDraw(img, bbox)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            else:
                label = 'Real'
                cv2.putText(img,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('Liveness Detection', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()
