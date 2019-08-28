from .util import preprocessor
from .util import predictions
from flask import Flask, jsonify, request
import pickle

def create_app():
    app = Flask(__name__)

    @app.route('/')
    @app.route('/index')
    @app.route('/api', methods=['POST'])
    def make_predict():
        #get data
        data = request.get_json(force=True)
        #parse
        predict_request = data['book_desc']
        #preds
        preds = predictions(predict_request)
        #send back to browser
        output = preds.to_json(orient='index')
        return output


    return app

