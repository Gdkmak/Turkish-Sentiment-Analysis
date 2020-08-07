import os 
import pandas as pd 
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import dispatcher


TRAINING_DATA = 'input/train_fold.csv' # os.environ.get("TRAINING_DATA")
FOLD = 0 # int(os.environ.get("FLOD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
  0: [1, 2, 3, 4],
  1: [0, 2, 3, 4],
  2: [0, 1, 3, 4],
  3: [0, 1, 2, 4],
  4: [0, 1, 2, 3],
}

if __name__ == '__main__': 
  df = pd.read_csv(TRAINING_DATA)
  
###### Just for now
  to_drop = []
  for i in range(df.shape[0]): 
    if str(df.loc[i, 'sentiment']) == 'nan':
      to_drop.append(i)
  df = df.drop(to_drop)
######

  train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
  valid_df = df[df.kfold== FOLD]

  xtrain = train_df.sentiment.values
  xvalid = valid_df.sentiment.values

  ytrain = train_df.target.values.tolist()
  yvalid = valid_df.target.values.tolist()

  tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1)

# fitting TF-IDF to both training and test sets (semi-supervised learning)
  tfv.fit(list(xtrain) + list(xvalid))
  xtrain_tfv =  tfv.transform(xtrain) 
  xvalid_tfv = tfv.transform(xvalid)

# train the model
  #clf = dispatcher.MODELS[MODEL]
  clf  = MultinomialNB()
  clf.fit(xtrain_tfv, ytrain)
  preds = clf.predict(xvalid_tfv)

# evaluate the results 
  print(confusion_matrix(yvalid, preds))


  # save the model
  joblib.dump(clf, f'models/{MODEL}.pkl')












