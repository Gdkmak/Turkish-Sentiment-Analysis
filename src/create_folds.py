import pandas as pd
import preprocessing as pre
from sklearn import model_selection


if __name__ == '__main__':
  
  sentiment = [] 
  target = [] 
  with open('/content/Movie-review-sentiment/input/tr_polarity.pos', 'r', encoding = "ISO-8859-1") as file_pos:
    for row in file_pos:   
        sentence = row
        sentiment.append(pre.preprocess(sentence))
        target.append('positive')
      
  with open('/content/Movie-review-sentiment/input/tr_polarity.neg', 'r', encoding = 'ISO-8859-1') as file_neg: 
    for row in file_neg: 
        sentence = row
        sentiment.append(pre.preprocess(sentence))
        target.append('negative')

  df = pd.DataFrame({'sentiment': sentiment, 'target': target, 'kfold': -1 })
  df = df.sample(frac = 1).reset_index(drop=True)

  kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=10)

  for fold, (train_idx, val_idx) in enumerate(kf.split(X= df, y= df.target.values)): 
    print(len(train_idx), len(val_idx))
    df.loc[val_idx, 'kfold']  = fold

  df.to_csv('/content/Movie-review-sentiment/input/train_fold.csv')