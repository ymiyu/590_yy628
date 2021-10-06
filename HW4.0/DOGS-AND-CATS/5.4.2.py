from keras.applications.vgg16 import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Defining the loss tensor for filter visualization
model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# Obtaining the gradient of the loss with regard to the input
grads = K.gradients(loss, model.input)[0]

# Gradient-normalization trick
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# Fetching Numpy output values given Numpy input values
iterate = K.function([model.input], [loss, grads])
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# Loss maximization via stochastic gradient descent

# Starts from a gray image with some noise
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

# Magnitude of each gradient update
step = 1.

# Runs gradient ascent for 40 steps
for i in range(40):
	# Computes the loss value and gradient value
	loss_value, grads_value = iterate([input_img_data])
	# Adjusts the input image in the direction that maximizes the loss
	input_img_data += grads_value * step

# Utility function to convert a tensor into a valid image
def deprocess_image(x):

	# Normalizes the tensor: centers on 0, ensures that std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# Clips to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# Converts to an RGB array
	x *= 255
	x = np.clip(x, 0, 255).astype('uint8')
	return x


# Function to generate filter visualizations
def generate_pattern(layer_name, filter_index, size=150):

	# Builds a loss function that maximizes the activation of the nth filter of the layer under consideration
	layer_output = model.get_layer(layer_name).output
	loss = K.mean(layer_output[:, :, :, filter_index])

	# Computes the gradient of the input picture with regard to this loss
	grads = K.gradients(loss, model.input)[0]
	# Normalization trick: normalizes the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
	# Returns the loss and grads given the input picture
	iterate = K.function([model.input], [loss, grads])
	# Starts from a gray image with some noise
	input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

	# Runs gradient ascent for 40 steps
	step = 1.
	for i in range(40):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step

	img = input_img_data[0]
	
	return deprocess_image(img)


plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

# Generating a grid of all filter response patterns in a layer

layer_name = 'block1_conv1'
size = 64
margin = 5

# Empty (black) image to store results
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

# Iterates over the rows of the results grid
for i in range(8):
	# Iterates over the columns of the results grid
	for j in range(8):
		# Generates the pattern for filter i + (j * 8) in layer_name
		filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
		# Puts the result in the square (i, j) of the results grid
		horizontal_start = i * size + i * margin
		horizontal_end = horizontal_start + size
		vertical_start = j * size + j * margin
		vertical_end = vertical_start + size
		results[horizontal_start: horizontal_end,
					vertical_start: vertical_end, :] = filter_img

# Displays the results grid
plt.figure(figsize=(20, 20))
plt.imshow(results.astype(np.uint8))
plt.show()