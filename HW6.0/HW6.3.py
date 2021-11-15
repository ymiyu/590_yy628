import numpy as np
import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras.callbacks import TensorBoard
from keras.losses import MeanSquaredError

#GET DATASET
from keras.datasets import cifar10
(X, Y), (test_images, test_labels) = cifar10.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(50000,32,32,3)

#MODEL
n_bottleneck=3

# #DEEPER
model = models.Sequential()

model.add(layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(32, 32, 3)))
# model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(2))
model.add(layers.Conv2D(16, 3, activation='relu', padding='same'))
# model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPool2D(2))

model.add(layers.Conv2D(16, 3, activation='relu', padding='same'))
# model.add(layers.BatchNormalization())
model.add(layers.UpSampling2D(2))
model.add(layers.Conv2D(32, 3, activation='relu', padding='same'))
# model.add(layers.BatchNormalization())
model.add(layers.UpSampling2D(2))
model.add(layers.Conv2D(3, 3, activation='softmax', padding='same'))
# model.add(layers.BatchNormalization())


# NH=200
# model.add(layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.Conv2D(32, 3, padding='same', activation='relu', strides=(2,2)))
# model.add(layers.Conv2D(32, 3, padding='same', activation='relu', strides=(2,2)))
# # model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(n_bottleneck,  activation='linear'))

# model.add(layers.Dense(2048, activation='relu'))
# model.add(layers.Reshape((8, 8, 32)))
# model.add(layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2,2)))
# model.add(layers.Conv2DTranspose(16, 3, padding='same', activation='relu', strides=(2,2)))
# model.add(layers.Conv2D(3, 3, padding='same', activation='relu'))



#COMPILE AND FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')
model.summary()
history = model.fit(X, X, 
						epochs=20, 
						batch_size=1000,
						validation_split=0.2)


# plot and save training/validation history plots
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)


plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('HW6.3-loss.png')
plt.show()

print('\n===== TEST MSE =====\n')

mse = MeanSquaredError()

#NORMALIZE AND RESHAPE
test_images = test_images/np.max(test_images) 
test_images = test_images.reshape(10000,32,32,3)
recosntructed_images = model.predict(test_images)
test_mse = mse(test_images, recosntructed_images).numpy()
print('test mse:', test_mse)


# #EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
# from keras import Model 
# extract = Model(model.inputs, model.layers[4].output) # Dense(128,...)
# X1 = extract.predict(X)
# # print(X1.shape)

# #2D PLOT
# plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
# plt.show()

# #3D PLOT
# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=X1[:,0], 
#     ys=X1[:,1], 
#     zs=X1[:,2], 
#     c=Y, 
#     cmap='tab10'
# )
# plt.show()

#PLOT ORIGINAL AND RECONSTRUCTED 
X1=model.predict(X) 

#RESHAPE
X=X.reshape(50000,32,32,3); #print(X[0])
X1=X1.reshape(50000,32,32,3); #print(X[0])

#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=0; I2=1
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X[I2])
ax[3].imshow(X1[I2])
plt.savefig('HW6.3-reconstructed.png')
plt.show()

# save model
model.save('HW6.3-model.h5')


print('\n===== Anomaly Detection =====\n')

# compute mse
train_true = []
train_pred = []

for i in range(len(X)):
	trained_mse = mse(X[i], X1[i]).numpy()
	if trained_mse < 0.08:
		train_pred.append(1)
	else:
		train_pred.append(0)
	train_true.append(1)

acc = sum(x==y for x, y in zip(train_true, train_pred)) / len(train_true)
print('accuracy (trained):', acc)


from keras.datasets import cifar100 
(X, Y), (test_images, test_labels) = cifar100.load_data()

Y = Y.reshape(Y.shape[0])

X = X[Y!=58, :, :, :]

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(49500,32,32,3)

X1=model.predict(X) 
anomaly_mse = mse(X, X1)

anomaly_true = []
anomaly_pred = []

for i in range(len(X)):
	trained_mse = mse(X[i], X1[i]).numpy()
	if trained_mse < 0.08:
		anomaly_pred.append(1)
	else:
		anomaly_pred.append(0)
	anomaly_true.append(0)

acc = sum(x==y for x, y in zip(anomaly_true, anomaly_pred)) / len(anomaly_true)
print('accuracy (anomalies):', acc)

y_true = train_true + anomaly_true
y_pred = train_pred + anomaly_pred

acc = sum(x==y for x, y in zip(y_true, y_pred)) / len(y_true)
print('accuracy (overall):', acc)