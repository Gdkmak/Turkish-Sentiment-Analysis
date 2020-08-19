import os, sys
import numpy as np 
import pandas as pd
import joblib
import time
import flask
from flask import Flask, request
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from keras.models import load_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocessing as pre


TRAINING_DATA = os.path.join('input', 'train_fold.csv')
max_features = 20000
sequence_length = 100

# instantiate a Text Vectorizer object
vectorize_layer = TextVectorization(
      max_tokens=max_features,
      output_mode="int",
      output_sequence_length=sequence_length
  )
  
def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)

app = Flask(__name__)

memory = joblib.Memory('../input/', verbose=0)

@memory.cache
def sentence_predict(sentence):
  df = pd.read_csv(TRAINING_DATA, sep=',')
  X = np.asarray(df['sentiment'], dtype=np.str)
  sentence = np.asarray([pre.preprocess(sentence)])
  vectorize_layer.adapt(np.concatenate((X, sentence)))

  X_test = tf.expand_dims(sentence, -1)
  X_test = vectorize_layer(X_test)

  model = load_model(os.path.join('models','deeplearning_model.h5'))
  pred = model.predict(X_test)

  return pred[0][0]


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    prediction = sentence_predict(str(sentence))
    positive_prediction = prediction
    negative_prediction = 1. - prediction
    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == '__main__': 
  app.run()










