from . import config
import torch
import flask
import time
from . import dataset
from flask import Flask
from flask import request
from .  import model 
import functools
import torch.nn as nn
import joblib


app = Flask(__name__)

MODEL = None
DEVICE = "cuda"

memory = joblib.Memory("../input/", verbose=0)


@memory.cache
def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    review = str(sentence)
    review = " ".join(review.split())

    ids, mask, token_type_ids = dataset.tokenization(tokenizer, review)

    ids = ids.unsqueeze(0).to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.unsqueeze(0).to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long).unsqueeze(0)

    outputs = MODEL(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    start_time = time.time()
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1 - positive_prediction
    response = {}
    response["response"] = {
        "positive": str(positive_prediction),
        "negative": str(negative_prediction),
        "sentence": str(sentence),
        "time_taken": str(time.time() - start_time),
    }
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = model.BERTBaseUncased()
    MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()
