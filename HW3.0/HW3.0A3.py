from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models, layers
from keras.regularizers import l1_l2
import numpy as np
import matplotlib.pyplot as plt


GAMMA_L1 = 0
GAMMA_L2 = 0
activation = 'tanh'
layer_list = [64, 64]


# Loading the Boston housing dataset
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences, dimension=10000):

	results = np.zeros((len(sequences), dimension))
	
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# Model definition
def build_model():

	model = models.Sequential()

	model.add(layers.Dense(layer_list[0], 
							activation=activation, 
							kernel_regularizer=l1_l2(l1=GAMMA_L1, l2=GAMMA_L2), 
							input_shape=(10000,)))

	for i in range(1, len(layer_list)):
		model.add(layers.Dense(layer_list[i], 
							activation=activation, 
							kernel_regularizer=l1_l2(l1=GAMMA_L1, l2=GAMMA_L2)))
	
	model.add(layers.Dense(46, activation='softmax', kernel_regularizer=l1_l2(l1=GAMMA_L1, l2=GAMMA_L2)))
	
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

	return model



k = 4
num_val_samples = len(train_data) // k
num_epochs = 20

all_train_loss = []
all_train_acc = []
all_val_loss = []
all_val_acc = []
all_test_loss = []
all_test_acc = []


# K-fold validation
for i in range(k):

	print('processing fold #', i)

	val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

	partial_train_data = np.concatenate(
				[x_train[:i * num_val_samples], x_train[(i + 1) * num_val_samples:]], axis=0)
	partial_train_targets = np.concatenate(
				[y_train[:i * num_val_samples], y_train[(i + 1) * num_val_samples:]], axis=0)
	
	model = build_model()
	history = model.fit(partial_train_data, partial_train_targets,
						 validation_data=(val_data, val_targets),
						 epochs=num_epochs, batch_size=512, verbose=0)

	# val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
	# all_scores.append(val_mae)
	train_loss = history.history['loss']
	train_acc = history.history['accuracy']
	val_loss = history.history['val_loss']
	val_acc = history.history['val_accuracy']

	all_train_loss.append(train_loss)
	all_train_acc.append(train_acc)
	all_val_loss.append(val_loss)
	all_val_acc.append(val_acc)



average_train_loss = [np.mean([x[i] for x in all_train_loss]) for i in range(num_epochs)]
average_train_acc = [np.mean([x[i] for x in all_train_acc]) for i in range(num_epochs)]
average_val_loss = [np.mean([x[i] for x in all_val_loss]) for i in range(num_epochs)]
average_val_acc = [np.mean([x[i] for x in all_val_acc]) for i in range(num_epochs)]

plt.plot(range(1, len(average_train_loss) + 1), average_train_loss, label = 'avg train loss')
plt.plot(range(1, len(average_val_loss) + 1), average_val_loss, label = 'avg val loss')
plt.xlabel('Epochs')
plt.ylabel('avg loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


plt.plot(range(1, len(average_train_acc) + 1), average_train_acc, label = 'avg train accuracy')
plt.plot(range(1, len(average_val_acc) + 1), average_val_acc, label = 'avg val accuracy')
plt.xlabel('Epochs')
plt.ylabel('avg accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

best_epoch = np.argmax(average_val_acc)

print("\n#--------K-FOLD VALIDATION--------")
print('Epoch:', best_epoch)
print('average train loss:', average_train_loss[best_epoch])
print('average train accuracy:', average_train_acc[best_epoch])
print('average validation loss:', average_val_loss[best_epoch])
print('average validation accuracy:', average_val_acc[best_epoch])


print("\n#--------FINAL MODEL--------")
print('GAMMA_L1:', GAMMA_L1)
print('GAMMA_L2:', GAMMA_L2)
print('activation:', activation)
print('layers:', layer_list)

model = build_model()
model.fit(x_train, y_train, epochs=best_epoch+1, batch_size=512, verbose=0)
_, test_acc = model.evaluate(x_test, y_test)
print('test accuracy:', test_acc)