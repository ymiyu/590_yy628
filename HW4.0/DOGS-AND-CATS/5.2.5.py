import os
from tensorflow import keras

import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


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

# Defining a new convnet that includes dropout
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Configuring the model for training
model.compile(
		loss='binary_crossentropy',
		optimizer=keras.optimizers.RMSprop(lr=1e-4),
		metrics=['acc'])

# Setting up a data augmentation configuration via ImageDataGenerator
datagen = ImageDataGenerator(
					rotation_range=40,
					width_shift_range=0.2,
					height_shift_range=0.2,
					shear_range=0.2,
					zoom_range=0.2,
					horizontal_flip=True,
					fill_mode='nearest')

# Displaying some randomly augmented training images
fnames = [os.path.join(train_cats_dir, fname) for
				fname in os.listdir(train_cats_dir)]

# Chooses one image to augment
img_path = fnames[3]
# Reads the image and resizes it
img = image.load_img(img_path, target_size=(150, 150))
# Converts it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)
# Reshapes it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)
# Generates batches of randomly transformed images. 
# Loops indefinitely, so need to break the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
	plt.figure(i)
	imgplot = plt.imshow(image.array_to_img(batch[0]))
	i += 1
	if i % 4 == 0:
		break

plt.show()

# Training the convnet using data-augmentation generators
train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=40,
					width_shift_range=0.2,
					height_shift_range=0.2,
					shear_range=0.2,
					zoom_range=0.2,
					horizontal_flip=True,)

# the validation data shouldn't be augmented
test_datagen = ImageDataGenerator(rescale=1./255)

# # Resizes all images to 150 × 150
# # Because we use binary_crossentropy loss, we need binary labels.
train_generator = train_datagen.flow_from_directory(train_dir,
			target_size=(150, 150),
			batch_size=20,
			class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
			target_size=(150, 150),
			batch_size=20,
			class_mode='binary')

# Fitting the model using a batch generator
history = model.fit_generator(train_generator,
			steps_per_epoch=100,
			epochs=100,
			validation_data=validation_generator,
			validation_steps=50)

# Saving the model
model.save('dogs_cats_small_2.h5')

# Displaying curves of loss and accuracy during training
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