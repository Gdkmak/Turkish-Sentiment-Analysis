TRAINING_DATA='input/train_fold.csv'
MODEL=$1

export TRAINING_DATA 
export MODEL 

#python3 -m src.Machine_Learning.train
#python3 -m src.Machine_Learning.app

#python3 -m src.Deep_Learning.train
python3 -m src.Deep_Learning.app

#python3 -m src.BERT.train
#python3 -m src.BERT.app
