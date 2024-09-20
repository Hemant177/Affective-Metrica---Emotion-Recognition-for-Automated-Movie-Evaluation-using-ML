# # expression_recognition.py
# import cv2
# import numpy as np
# from keras.models import load_model

# # Load the pre-trained model (Replace 'model_path' with your actual model path)
# model = load_model('model_path.h5')
# emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# def detect_emotion(frame):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         roi_gray = gray_frame[y:y+w, x:x+h]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#         prediction = model.predict(cropped_img)
#         maxindex = int(np.argmax(prediction))
#         return emotion_dict[maxindex]
#     return "No Face Detected"


import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from transformers import pipeline
from sklearn.linear_model import LinearRegression

# Load models
expression_model = load_model('emotion_model.h5')  # Replace with your actual model file name
sentiment_pipeline = pipeline('sentiment-analysis')

# Initialize video capture
video_capture = cv2.VideoCapture(0)  # 0 for default camera

# Initialize sentiment and rating models
def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label']

def infer_rating(expression_scores):
    # Example: Simple linear regression for rating inference
    X_train = np.array([[0, 0], [1, 1], [2, 2]])  # Dummy training data
    y_train = np.array([1, 2, 3])  # Dummy ratings
    model = LinearRegression().fit(X_train, y_train)
    return model.predict([expression_scores])[0]

def process_frame(frame):
    # Preprocess frame for expression analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        
        # Predict expression
        expression_prediction = expression_model.predict(face)
        expression_scores = expression_prediction[0]
        
        # Infer rating from expression scores
        rating = infer_rating(expression_scores)
        return rating

def main():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Process frame for emotion analysis
        rating = process_frame(frame)
        print(f'Predicted Rating: {rating}')
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
