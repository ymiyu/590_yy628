from keras.datasets import boston_housing
from keras import models, layers
from keras.regularizers import l1_l2
import numpy as np
import matplotlib.pyplot as plt


GAMMA_L1=0.0001
GAMMA_L2=0.0001
activation = 'tanh'
layer_list = [128, 64]

# Loading the Boston housing dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Normalizing the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# Model definition
def build_model():

	model = models.Sequential()

	model.add(layers.Dense(layer_list[0], 
							activation=activation, 
							kernel_regularizer=l1_l2(l1=GAMMA_L1, l2=GAMMA_L2), 
							input_shape=(train_data.shape[1],)))

	for i in range(1, len(layer_list)):
		model.add(layers.Dense(layer_list[i], 
							activation=activation, 
							kernel_regularizer=l1_l2(l1=GAMMA_L1, l2=GAMMA_L2)))

	model.add(layers.Dense(1, kernel_regularizer=l1_l2(l1=GAMMA_L1, l2=GAMMA_L2)))
	model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

	return model



k = 4
num_val_samples = len(train_data) // k
num_epochs = 100

all_train_loss = []
all_train_mae = []
all_val_loss = []
all_val_mae = []

train_mse = []
train_mae = []
val_mse = []
val_mae = []


# K-fold validation
for i in range(k):

	print('processing fold #', i)

	val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

	partial_train_data = np.concatenate(
				[train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
	partial_train_targets = np.concatenate(
				[train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
	
	model = build_model()
	history = model.fit(partial_train_data, partial_train_targets,
						 validation_data=(val_data, val_targets),
						 epochs=num_epochs, batch_size=32, verbose=0)

	# val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
	# all_scores.append(val_mae)
	train_loss = history.history['loss']
	train_mae = history.history['mae']
	val_loss = history.history['val_loss']
	val_mae = history.history['val_mae']

	all_train_loss.append(train_loss)
	all_train_mae.append(train_mae)
	all_val_loss.append(val_loss)
	all_val_mae.append(val_mae)


average_train_loss = [np.mean([x[i] for x in all_train_loss]) for i in range(num_epochs)]
average_train_mae = [np.mean([x[i] for x in all_train_mae]) for i in range(num_epochs)]
average_val_loss = [np.mean([x[i] for x in all_val_loss]) for i in range(num_epochs)]
average_val_mae = [np.mean([x[i] for x in all_val_mae]) for i in range(num_epochs)]

best_epoch = np.argmin(average_val_mae)

print("\n#--------K-FOLD VALIDATION--------")
print('Epoch:', best_epoch)
print('average train MAE:', average_train_mae[best_epoch])
print('average train MSE:', average_train_loss[best_epoch])
print('average validation MAE:', average_val_mae[best_epoch])
print('average validation MSE:', average_val_loss[best_epoch])


plt.plot(range(1, len(average_train_loss) + 1), average_train_loss, label = 'avg train loss')
plt.plot(range(1, len(average_val_loss) + 1), average_val_loss, label = 'avg val loss')
plt.xlabel('Epochs')
plt.ylabel('avg loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


plt.plot(range(1, len(average_train_mae) + 1), average_train_mae, label = 'avg train mae')
plt.plot(range(1, len(average_val_mae) + 1), average_val_mae, label = 'avg val mae')
plt.xlabel('Epochs')
plt.ylabel('avg MAE')
plt.title('Training and validation MAE')
plt.legend()
plt.show()


print("\n#--------FINAL MODEL--------")
print('GAMMA_L1:', GAMMA_L1)
print('GAMMA_L2:', GAMMA_L2)
print('activation:', activation)
print('layers:', layer_list)
# test different parameters
model = build_model()
model.fit(train_data, train_targets, epochs=best_epoch+1, batch_size=32, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print('test MAE:', test_mae_score)
print('test MSE:', test_mse_score)