import os
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, render_template
import cv2
from feat import Detector
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import time
import io
import matplotlib.pyplot as plt
from datetime import datetime
import random

app = Flask(__name__)



detector = Detector(
    face_model = "retinaface",
    landmark_model = "mobilefacenet",
    au_model = 'xgb',
    emotion_model = "resmasknet",
    facepose_model = "img2pose",
)



latest_emotions = {}

def generate_frames():
    camera = cv2.VideoCapture(0)
    time.sleep(2) 
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > 30:
            break
        
        success, frame = camera.read()
        if not success:
            break
        
        print(frame)
        print(frame.shape)
        print(frame.dtype)
        # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"captured_frame_{current_time}.jpg"
        
        filename = "static/example.jpg"
        cv2.imwrite(filename, frame)
        
        _, buffer = cv2.imencode('.jpg', frame)

        
        bytes_io = io.BytesIO(buffer)

        
        predictions = detector.detect_image(filename)
        if predictions is not None:
            
            global latest_emotions
            latest_emotions = predictions.emotions.iloc[0].to_dict()
            print("Emotional Analysis Results:", latest_emotions)
        else:
            print("Failure")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
        #cv2.putText(frame, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    
    camera.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotions')
def emotions():
    return jsonify(latest_emotions)

@app.route('/')
def index():
    return render_template('index.html', latest_emotions = latest_emotions)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)