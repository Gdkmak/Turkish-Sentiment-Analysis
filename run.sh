TRAINING_DATA='input/train_fold.csv'
MODEL=$1

export TRAINING_DATA 
export MODEL 

# For either Logistic Regression or Naive Bayes
# Pass either LR or NB When you run the command sh run.sh 
# run one command at a time
# first train and then app 

python3 -m src.Machine_Learning.train
#python3 -m src.Machine_Learning.app

# For Deep Learning
# run one command at a time
# first train and then app 

#python3 -m src.Deep_Learning.train
python3 -m src.Deep_Learning.app

# For BERT
# run one command at a time
# first train and then app 

#python3 -m src.BERT.train
#python3 -m src.BERT.app
