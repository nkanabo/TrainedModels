from flask import Flask
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)


class prediction(Resource):
    def get(self, comment):
        print(comment)
        df = pd.DataFrame([comment], columns=['Comment'])
        model = pickle.load(open('fine_tuned_logistic_regression.pkl', 'rb'))
        prediction = model.predict_proba(df)
        result = {'Negative': round(prediction[0][0] * 100, 2), 'Neutral': round(prediction[0][1] * 100, 2),
                  'Positive': round(prediction[0][2] * 100, 2)}
        return result


api.add_resource(prediction, '/prediction/<string:comment>')

if __name__ == '__main__':
    app.run(debug=True)
