import transformers
import os 

MAX_LEN = 50
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
BERT_PATH = "savasy/bert-base-turkish-sentiment-cased"
MODEL_PATH = "models/preprocessed_model.bin"
TRAINING_FILE = os.environ.get('TRAINING_DATA')
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True, cache_dir='PYTORCH_PRETRAINED_BERT_CACHE')