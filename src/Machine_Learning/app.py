import os, sys
import pandas as pd 
import joblib
import flask 
import time
from flask import Flask, request
from sklearn.feature_extraction.text import TfidfVectorizer

memory = joblib.Memory("../input/", verbose=0)

app = Flask(__name__)

@memory.cache
def sentence_predict(sentence, model): 
  sentiment = [pre.preprocess(sentence)]
  tfv = joblib.load(os.path.join('models', 'tfv.pkl'))
  xtest_tfv =  tfv.transform(sentiment) 
  for FOLD in range(5): 
    clf = joblib.load(os.path.join('models', f'{model}_FOLD_{FOLD}.pkl'))
    preds = clf.predict_proba(xtest_tfv)

    if FOLD == 0: 
      predictions = preds
    else: 
      predictions += preds 
    
  predictions /=5
  return predictions

@app.route("/predict")
def predict():
    model = request.args.get('model', 'NB')  
    sentence = request.args.get("sentence")
    start_time = time.time()
    prediction = sentence_predict(str(sentence), model)
    positive_prediction = prediction[0][1]
    negative_prediction = prediction[0][0]
    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        'model': str(model),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == '__main__':
  app.run()








