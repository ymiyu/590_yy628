
#MODIFIED FROM CHOLLETT P120 
from keras import layers 
from keras import models
from keras.models import load_model
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import random


#USER PARAMETERS
DATASET = 'MNIST'
save_name = 'MNIST-aug.h5'
model_type = 'CNN'
#image_shape = (28, 28, 1)
batch_size = 64
epochs = 5
old_setting = np.seterr(all='ignore')


#-------------------------------------
#SET INPUT SHAPE
#-------------------------------------
if DATASET == 'MNIST':
	image_shape = (28, 28, 1)
if DATASET == 'MNIST Fashion':
	image_shape = (28, 28, 1)
if DATASET == 'CIFAR-10':
	image_shape = (32, 32, 3)


#-------------------------------------
#BUILD MODEL SEQUENTIALLY (LINEAR STACK)
#-------------------------------------
if model_type == 'CNN':
	
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

elif model_type == 'ANN':

	model = models.Sequential()
	model.add(layers.Flatten(input_shape=image_shape))
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))



model.summary()



# VISUALIZE
def visualize_image(images):
	index = random.randint(0, len(images)-1)
	plt.imshow(images[index], cmap=plt.get_cmap('gray'))
	plt.show()


#-------------------------------------
#GET DATA AND REFORMAT
#-------------------------------------

if DATASET == 'MNIST':
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.reshape((60000, 28, 28, 1))
	test_images = test_images.reshape((10000, 28, 28, 1))
elif DATASET == 'MNIST Fashion':
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	train_images = train_images.reshape((60000, 28, 28, 1))
	test_images = test_images.reshape((10000, 28, 28, 1))
elif DATASET == 'CIFAR-10':
	(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
	train_images = train_images.reshape((50000, 32, 32, 3))
	test_images = test_images.reshape((10000, 32, 32, 3))


visualize_image(train_images)

#NORMALIZE
train_images = train_images.astype('float32') / 255 
test_images = test_images.astype('float32') / 255  

#CONVERTS A CLASS VECTOR (INTEGERS) TO BINARY CLASS MATRIX.
tmp=train_labels[0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(tmp, '-->',train_labels[0])
print("train_labels shape:", train_labels.shape)

# Do 80-20 split of the “training” data into (train/validation)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2)


# Training the convnet using data-augmentation generators
train_datagen = ImageDataGenerator(
					featurewise_center=True,
					featurewise_std_normalization=True,
					rotation_range=20,
					width_shift_range=0.2,
					height_shift_range=0.2,
					horizontal_flip=True)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
val_generator = test_datagen.flow(val_images, val_labels, batch_size=batch_size)


#-------------------------------------
#COMPILE AND TRAIN MODEL
#-------------------------------------
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fitting the model
history = model.fit_generator(train_generator, 
			epochs=epochs, 
			validation_data=val_generator)

# Saving the model
model.save(save_name)

# Displaying curves of loss and accuracy during training
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#-------------------------------------
#EVALUATE ON TEST DATA
#-------------------------------------
train_loss, train_acc = model.evaluate(train_images, train_labels)
val_loss, val_acc = model.evaluate(val_images, val_labels)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('train_acc:', train_acc)
print('val_acc:', val_acc)
print('test_acc:', test_acc)


#--------------------------------------
# VISUALIZE
#--------------------------------------

# read model from a file
model = load_model(save_name)
model.summary()

# fetch a single image
img_tensor = test_images[0:1]
print(img_tensor.shape)

# Displaying the test picture
plt.imshow(img_tensor[0])
plt.show()

# Instantiating a model from an input tensor and a list of output tensors
layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# Running the model in predict mode
activations = activation_model.predict(img_tensor)

# Visualizing every channel in every intermediate activation

layer_names = []
for layer in model.layers:
	layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
	if layer_activation.ndim != 4:
		continue
	# Number of features in the feature map
	n_features = layer_activation.shape[-1]
	size = layer_activation.shape[1]

	# Tiles the activation channels in this matrix
	n_cols = n_features // images_per_row
	display_grid = np.zeros((size * n_cols, images_per_row * size))

	# Tiles each filter into a big horizontal grid
	for col in range(n_cols):
		for row in range(images_per_row):
			channel_image = layer_activation[0,:, :,col * images_per_row + row]

			# Post-processes the feature to make it visually palatable
			channel_image -= channel_image.mean()
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			# Displays the grid
			display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

	scale = 1. / size
	plt.figure(figsize=(scale * display_grid.shape[1],
	scale * display_grid.shape[0]))
	plt.title(layer_name)
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.show()