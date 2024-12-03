import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = load_model('T:/CodeBase/FullStackMLWeb/WebAIProject/ObjectDetection/models/trainedModel.h5', compile=False)

class_names = ['Touhid', 'Tishad', 'Shimla', 'Abir']

cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_roi = frame[y:y+h, x:x+w]
                face_img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_img = cv2.resize(face_img, (64, 64))
                face_img = image.img_to_array(face_img)
                face_img = np.expand_dims(face_img, axis=0)
                face_img = face_img / 255.0
                prediction = model.predict(face_img)
                max_index = np.argmax(prediction[0])
                confidence = prediction[0][max_index] * 100
                
                # Check confidence level
                if confidence >= 60:
                    predicted_class = class_names[max_index]
                else:
                    predicted_class = "Unknown"
                
                label = f"{predicted_class} ({confidence:.2f}%)"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret,buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
