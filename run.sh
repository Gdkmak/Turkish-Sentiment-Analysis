TRAINING_DATA='input/train_fold.csv'
MODEL=$1
TEST_DATA=$2 

export TRAINING_DATA 
export MODEL 
export TEST_DATA

# python -m src.train
python -m src.predict
