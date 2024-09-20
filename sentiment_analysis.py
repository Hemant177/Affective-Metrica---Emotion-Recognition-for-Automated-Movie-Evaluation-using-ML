def analyze_sentiment(emotion):
    positive_emotions = ["Happy", "Surprise"]
    negative_emotions = ["Angry", "Disgust", "Fear", "Sad"]
    neutral_emotions = ["Neutral"]

    if emotion in positive_emotions:
        return "Positive Feedback: You seem to have enjoyed the movie!"
    elif emotion in negative_emotions:
        return "Negative Feedback: The movie might not have been to your liking."
    elif emotion in neutral_emotions:
        return "Neutral Feedback: The movie was okay for you."
    else:
        return "Unable to analyze sentiment."
