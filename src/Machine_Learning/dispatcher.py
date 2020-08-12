from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

MODELS = {
    'NB': MultinomialNB(), 
    'LR': LogisticRegression(C=1.)
}