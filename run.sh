TRAINING_DATA = 'input/train_fold.csv'
FOLD = 0
MODEL = $1
# export TRAINING_DATA 
# export FOLD 
# export MODEL 

python -m src.train $TRAINING_DATA $FOLD