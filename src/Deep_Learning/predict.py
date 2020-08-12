import os, sys
import numpy as np 
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from keras.models import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocessing as pre

TEST_DATA = os.environ.get('TEST_DATA')
TRAINING_DATA = os.environ.get('TRAINING_DATA')
MODEL = os.environ.get('MODEL')

max_features = 20000
sequence_length = 100

def vectorize_text(text):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text)

def predict():
  df = pd.read_csv(TRAINING_DATA, sep=',')
  X = np.asarray(df['sentiment'], dtype=np.str)
  sentence = np.asarray([pre.preprocess(TEST_DATA)])
  
  vectorize_layer = TextVectorization(
      max_tokens=max_features,
      output_mode="int",
      output_sequence_length=sequence_length,
  )
  vectorize_layer.adapt(np.concatenate((X, sentence)))
  X_test = tf.expand_dims(sentence, -1)
  X_test = vectorize_layer(X_test)

  model = load_model(os.path.join('models','deeplearning_model.h5'))
  pred = model.predict(X_test)

  return pred[0][0]


if __name__ == '__main__': 
  result = predict()
  print(f'This sentence is {result:.0%} likely to be positive\n\t\t {1.-result:.0%} likely to be negative')











