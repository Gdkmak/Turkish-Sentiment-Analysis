import zipfile
import csv
import pandas as pd 


if __name__ == '__main__':
    
    sentiment = [] 
    target = [] 

    with open('input/tr_polarity.pos', 'r', encoding = "ISO-8859-1") as file_pos:
        for row in file_pos:
    
            sentence = str(row).strip()
            sentiment.append(sentence)
            target.append('positive')
        
    with open('input/tr_polarity.neg', 'r', encoding = 'ISO-8859-1') as file_neg: 
        for row in file_neg: 
            sentence = str(row).strip()
            sentiment.append(sentence)
            target.append('negative')

    ds = pd.DataFrame({'sentiment': sentiment, 'target': target})
    ds = ds.sample(frac= 1)
    ds.to_csv('input/non_preprocessing_train_fold.csv', index= False)