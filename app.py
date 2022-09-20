from copyreg import pickle
from flask import Flask, request, render_template,session, app, jsonify
import datetime
import json
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pickle


app = Flask(__name__)

model = pickle.load(open('finalized_model1.pkl', 'rb'))
vec = pickle.load(open('feature1.pkl', 'rb'))

# transformer = TfidfTransformer()
# loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature1.pkl", "rb")))
# loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
saved_vocabulary = pickle.load(open("feature1.pkl", 'rb'))

@app.route('/prediction', methods=['POST'])
# def predict():
#     text = request.form
#     vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
#     vectoriser.fit(text)
#     text = vectoriser.transform(text)
#     return model.predict(text)[0]

@app.route('/')
def home_page():
    
    return render_template('index.html')

@app.route('/form/submit', methods=['POST'])
def my_form_post():
    response = request.form
    user_data = json.loads(json.dumps(dict(response)))
    tweet = user_data['tweet']
    # t2 = tweet.reshape(1, -1)
    
    # vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000, vocabulary= )
    tweet = [tweet]
    y1 = vec.transform(tweet)
    prediction = model.predict(y1)
    json_str = json.dumps({'Prediction': prediction.tolist(), 'Tweet': tweet})
    
    return (json_str)
    












if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0")