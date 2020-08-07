import sys 
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
from . import dispatcher


TRAINING_DATA = sys.argv[0]
FOLD = int(sys.argv[1])
MODEL = sys.argv[2]

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
  clf = dispatcher.MODELS[MODEL]
  #clf  = MultinomialNB()
  clf.fit(xtrain_tfv, ytrain)
  preds = clf.predict(xvalid_tfv)

# evaluate the results 
  average_precision = average_precision_score(yvalid, preds)
  print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
  disp = plot_precision_recall_curve(clf, xvalid_tfv, yvalid)
  disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))

  # save the model
  joblib.dump(clf, f'models/{MODEL}.pkl')












