import os
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

from tqdm import tqdm

base_dir = 'dogs_cats_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

# Instantiating the VGG16 convolutional base
conv_base = VGG16(
				weights='imagenet',
				include_top=False,
				input_shape=(150, 150, 3))

conv_base.summary()

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

# Extracting features using the pretrained convolutional base
def extract_features(directory, sample_count):
	features = np.zeros(shape=(sample_count, 4, 4, 512))
	labels = np.zeros(shape=(sample_count))
	generator = datagen.flow_from_directory(directory,
							target_size=(150, 150),
							batch_size=batch_size,
							class_mode='binary')
	i = 0
	for inputs_batch, labels_batch in tqdm(generator):
		features_batch = conv_base.predict(inputs_batch)
		features[i * batch_size : (i + 1) * batch_size] = features_batch
		labels[i * batch_size : (i + 1) * batch_size] = labels_batch
		i += 1
		# Note that because generators directory yield data indefinitely in a loop,
		# you must break after every image has been seen once.
		if i * batch_size >= sample_count:
			break
	
	return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


# Defining and training the densely connected classifier
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# Configuring the model for training
model.compile(
		optimizer=keras.optimizers.RMSprop(lr=2e-5),
		loss='binary_crossentropy',
		metrics=['acc'])

# Fitting the model using a batch generator
history = model.fit(train_features, train_labels,
		epochs=30,
		batch_size=20,
		validation_data=(validation_features, validation_labels))


# Plotting the results

acc = history.history['acc']
val_acc = history.history['val_acc']
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