from flask import Flask
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS
import re
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

# loaded_model = joblib.load(r'C:\Users\Nkanabo\Desktopfinalized_model.sav')
with open('fine_tuned_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)


# Define a function to preprocess the text input
def preprocess_text(text):
    # Replace any non-alphanumeric characters with spaces
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)

    # Convert the text to lowercase
    text = text.lower()

    # Return the preprocessed text
    return text


# Define a function that takes a text input and returns the predicted sentiment and confidence percentages
def predict_sentiment(text):
    # Preprocess the text input to match the format expected by the model
    preprocessed_text = preprocess_text(text)

    # Use the model to predict the sentiment probabilities for the preprocessed text
    sentiment_probs = model.predict_proba([preprocessed_text])[0]

    # Determine the predicted sentiment class based on the sentiment probabilities
    sentiment_class = np.argmax(sentiment_probs)

    # Return the predicted sentiment class and confidence percentages for each sentiment class
    if sentiment_class == 0:
        return 'Negative', sentiment_probs[0] * 100, sentiment_probs[1] * 100, sentiment_probs[2] * 100
    elif sentiment_class == 1:
        return 'Neutral', sentiment_probs[0] * 100, sentiment_probs[1] * 100, sentiment_probs[2] * 100
    else:
        return 'Positive', sentiment_probs[0] * 100, sentiment_probs[1] * 100, sentiment_probs[2] * 100


class prediction(Resource):
    def get(self, comment):
        predict = predict_sentiment(comment)
        sentiment, negative_perc, neutral_perc, positive_perc = predict_sentiment(comment)
        result = [sentiment, negative_perc, neutral_perc, positive_perc]
        print(f'Sentiment: {sentiment}')
        print(f'Negative: {negative_perc:.2f}%')
        print(f'Neutral: {neutral_perc:.2f}%')
        print(f'Positive: {positive_perc:.2f}%')
        sult = {'Negative': negative_perc, 'Neutral': neutral_perc,
                'Positive': positive_perc}
        return sult


api.add_resource(prediction, '/prediction/<string:comment>')

if __name__ == '__main__':
    app.run(debug=True)
