from flask import Flask, render_template, Response, request
import cv2
from expression_recognition import detect_emotion
from sentiment_analysis import analyze_sentiment

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            emotion = detect_emotion(frame)
            feedback = analyze_sentiment(emotion)
            cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/feedback', methods=['POST'])
def feedback():
    emotion = request.form['emotion']
    feedback = analyze_sentiment(emotion)
    return feedback

if __name__ == '__main__':
    app.run(debug=True)
