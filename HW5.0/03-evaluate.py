import json
import numpy as np
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D

from keras.utils.np_utils import to_categorical
from keras.metrics import AUC

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

all_titles = ['Musical Memories', 'Poirot Investigates', 'Children of the Frost']

data = json.load(open('data.json', 'r'))
texts = data['x']
titles = data['y']

# model type
model_type = 'RNN'

# Cuts off reviews after 100 words
maxlen = 100
# Considers only the top 10,000 words in the dataset
max_words = 10000

model = load_model(model_type+'/model.h5')
model.summary()


tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = [all_titles.index(x) for x in titles]
num_labels = len(all_titles)
labels = to_categorical(labels)

# labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Splits the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.seed(0)
np.random.shuffle(indices)
data = data[indices]

labels = labels[indices]

# Do 70-30 split of the “training” data into (train/validation)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

train_loss, train_acc, train_auc = model.evaluate(x_train, y_train)
val_loss, val_acc, val_auc = model.evaluate(x_val, y_val)
test_loss, test_acc, test_auc = model.evaluate(x_test, y_test)

print('train_acc:', train_acc)
print('val_acc:', val_acc)
print('test_acc:', test_acc)
print('train_AUC:', train_auc)
print('val_AUC:', val_auc)
print('test_AUC:', test_auc)