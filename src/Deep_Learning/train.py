import numpy as np 
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers

TRAINING_DATA=os.environ.get('TRAINING_DATA')
EPOCH=3

# Model constants
max_features=15000
embedding_dim=128
sequence_length=50

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


if __name__== '__main__':
  # read the dataset 
  df = pd.read_csv(TRAINING_DATA, sep=',')
  df = df.replace(['positive', 'negative'],[1,0])
  df = df.dropna()

  X = np.asarray(df['sentiment'], dtype=np.str)
  y = np.asarray(df['target'], dtype=np.int)

  # since the data is not big enough, 10% will be left for validation purposes
  split = int(df.shape[0] * .10)
  X_test = (X[0:split])
  y_test = (y[0:split])

  X_train = (X[split::])
  y_train = (y[split::])

  # instantiate our text vectorization layer. 
  # I am using this layer to normalize, split, and map
  # strings to integers, so we set our 'output_mode' to 'int'.
  # also set an explicit maximum sequence length, since the CNNs later in
  # the model won't support ragged sequences.
  vectorize_layer = TextVectorization(
      max_tokens=max_features,
      output_mode="int",
      output_sequence_length=sequence_length,
  )
  vectorize_layer.adapt(np.concatenate((X_test,X_train)))

  dataset_tr = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  dataset_tr = dataset_tr.batch(32)

  dataset_val = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  dataset_val = dataset_val.batch(32)


  # Vectorize the data.
  train_ds = dataset_tr.map(vectorize_text)
  val_ds = dataset_val.map(vectorize_text)

  # Do async prefetching / buffering of the data for best performance on GPU.
  train_ds = train_ds.cache().prefetch(buffer_size=10)
  val_ds = val_ds.cache().prefetch(buffer_size=10)

  # A integer input for vocab indices.
  inputs = tf.keras.Input(shape=(None,), dtype="int64")

  # add a layer to map those vocab indices into a space of dimensionality
  x = layers.Embedding(max_features, embedding_dim)(inputs)
  x = layers.Dropout(0.5)(x)

  # Conv1D + global max pooling
  x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
  x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
  x = layers.GlobalMaxPooling1D()(x)

  # vanilla hidden layer:
  x = layers.Dense(128, activation="relu")(x)
  x = layers.Dropout(0.5)(x)

  # a single unit output layer, and squash it with a sigmoid:
  predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
  model = tf.keras.Model(inputs, predictions)

  # Compile the model with binary crossentropy loss and an adam optimizer.
  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
  model.fit(train_ds, validation_data=val_ds, epochs=EPOCH)
  
  # evaluate the model
  scores = model.evaluate(val_ds, verbose=0)
  print("The %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  
  # save model and architecture to single file
  model.save(os.path.join('models','deeplearning_model.h5'))