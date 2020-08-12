import os 
import pandas as pd 
import matplotlib.pyplot as plt
import joblib
import os,sys,inspect

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from . import dispatcher


TRAINING_DATA = os.environ.get('TRAINING_DATA')
MODEL = os.environ.get('MODEL')

FOLD_MAPPING = {
  0: [1, 2, 3, 4],
  1: [0, 2, 3, 4],
  2: [0, 1, 3, 4],
  3: [0, 1, 2, 4],
  4: [0, 1, 2, 3],
}

if __name__ == '__main__': 
  df = pd.read_csv(TRAINING_DATA)
  df = df.replace(['positive','negative'], [1,0])
  df = df.dropna()

  tfv = TfidfVectorizer(min_df=3,  max_features=None, 
              strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
              ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1)

  average_precision = 0

  for FOLD in range(5):   
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold== FOLD]

    xtrain = train_df.sentiment.values
    xvalid = valid_df.sentiment.values

    ytrain = train_df.target.values.tolist()
    yvalid = valid_df.target.values.tolist()

  # fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit(list(xtrain) + list(xvalid))
    xtrain_tfv =  tfv.transform(xtrain) 
    xvalid_tfv = tfv.transform(xvalid)

  # train the model
    clf = dispatcher.MODELS[MODEL]
    clf.fit(xtrain_tfv, ytrain)
    preds = clf.predict(xvalid_tfv)

  # evaluate the results 
    precision = average_precision_score(yvalid, preds)
    average_precision += precision 

    print(f'Average precision-recall score for the {MODEL} model in FOLD {FOLD} is {precision:0.2f}')
    # save the model
    joblib.dump(clf, f'models/{MODEL}_FOLD_{FOLD}.pkl')
    joblib.dump(tfv, f'models/tfv.pkl')

  average_precision /= 5 
  print(f'Average precision-recall score for all FOLDS of {MODEL} model is {average_precision:0.2f}')










