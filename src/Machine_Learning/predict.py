import os, sys
import pandas as pd 
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from . import dispatcher

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import preprocessing as pre


TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get('MODEL')

def predict(): 
  sentiment = [pre.preprocess(TEST_DATA)]
  tfv = joblib.load(os.path.join('models', 'tfv.pkl'))
  xtest_tfv =  tfv.transform(sentiment) 
  for FOLD in range(5): 
    clf = joblib.load(os.path.join('models', f'{MODEL}_FOLD_{FOLD}.pkl'))
    preds = clf.predict_proba(xtest_tfv)

    if FOLD == 0: 
      predictions = preds
    else: 
      predictions += preds 
    
  predictions /=5
  return predictions


if __name__ == '__main__': 
  result = predict()
  print(f'This sentence is {result[0][1]:.0%} likely to be positive\n\t\t {result[0][0]:.0%} likely to be negative')











