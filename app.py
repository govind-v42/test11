from copyreg import pickle
from flask import Flask, request, render_template,session, app
import datetime
import json

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


app = Flask(__name__)

model = pickle.load(open('finalized_model.pkl', 'rb'))

@app.route('/prediction', methods=['POST'])
def predict():
    text = request.form
    vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
    vectoriser.fit(text)
    text = vectoriser.transform(text)
    return model.predict(text)[0]

@app.route('/')
def home_page():
    return "Welcome to Prediction home"
    












if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0")