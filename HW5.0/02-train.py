import json
import numpy as np
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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

# Cuts off reviews after 100 words
maxlen = 100
# Considers only the top 10,000 words in the dataset
max_words = 10000
# model type
model_type = 'RNN'


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


if model_type == 'RNN':

	model = Sequential()

	model.add(Embedding(max_words, 32))
	model.add(LSTM(32))
	model.add(Dropout(0.2))
	model.add(Dense(3, activation='sigmoid'))

elif model_type == 'CNN':

	layer1=tf.keras.layers.Conv1D(
				10, 3, activation='relu',
				kernel_initializer="ones",
				bias_initializer="zeros",
				# input_shape=data.shape[1:]
			)

	model = Sequential()

	model.add(Embedding(max_words, 32))
	model.add(layer1)
	model.add(GlobalMaxPooling1D())
	model.add(Dense(3, activation='sigmoid'))




model.compile(optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['acc', AUC()])

history = model.fit(x_train, y_train,
						epochs=20,
						batch_size=128,
						validation_data=(x_val, y_val))


acc = history.history['acc']
val_acc = history.history['val_acc']
auc = history.history['auc']
val_auc = history.history['val_auc']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(acc) + 1)

plt.plot(epochs, auc, 'bo', label='Training AUC')
plt.plot(epochs, val_auc, 'b', label='Validation AUC')
plt.title('Training and validation AUC')
plt.legend()
plt.savefig(model_type + '/auc.png')
plt.show()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(model_type + '/acc.png')
plt.show()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(model_type + '/loss.png')
plt.show()



with open(model_type + '/log.txt', 'w') as fw:

	fw.write('========== TRAINING INFO ==========\n')

	for i in range(len(epochs)):
		fw.write('Training epoch {}\n'.format(epochs[i]))
		fw.write('acc: {}\tval_acc: {}\n'.format(acc[i], val_acc[i]))
		fw.write('auc: {}\tval_auc: {}\n'.format(auc[i], val_auc[i]))
		fw.write('loss: {}\tval_loss: {}\n'.format(loss[i], val_loss[i]))

	fw.write('\n========== FINAL METRICS ==========\n')

	train_loss, train_acc, train_auc = model.evaluate(x_train, y_train)
	val_loss, val_acc, val_auc = model.evaluate(x_val, y_val)
	test_loss, test_acc, test_auc = model.evaluate(x_test, y_test)
	print('train_acc:', train_acc)
	print('val_acc:', val_acc)
	print('test_acc:', test_acc)
	print('train_AUC:', train_auc)
	print('val_AUC:', val_auc)
	print('test_AUC:', test_auc)

	fw.write('train_acc: {}\n'.format(train_acc))
	fw.write('val_acc: {}\n'.format(val_acc))
	fw.write('test_acc {}\n:'.format(test_acc))
	fw.write('train_AUC: {}\n'.format(train_auc))
	fw.write('val_AUC: {}\n'.format(val_auc))
	fw.write('test_AUC: {}\n'.format(test_auc))

fw.close()

model.save(model_type + '/model.h5')