import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy
from scipy.optimize import least_squares, curve_fit


# define measurement
def MSE(y, y_pred):
	return np.mean((y-y_pred)**2)

# define numpy array normalizer
def standard_scaler(X):
	return (X - X.mean())/(X.std())

# Loading Data
class Data:

	def __init__(self):
		self.filename = None
		self.data = None
		self.X = None
		self.y = None

	def load_file(self, filename):
		with open(filename, 'r') as f:
			self.data = json.load(f)
			#self.data = pd.DataFrame(json_file)
		return self.data

	def set_variables(self, x_variable, y_variable):
		# set X variables and y variable for models
		# x_variables: a list of variables existed in the data used as predictors
		# y_variable: a column name representing the target variable
		df = self.data.copy()
		#print('SET_VARIABLES')
		#print(y_variable)
		#print(df)
		self.y = df.pop(str(y_variable))
		self.X = df[x_variable]
		return self.X, self.y


	def split_data(self, percent):
		# split the percentage as training set
		
		if self.X is not None:
			X_train, X_test, y_train, y_test=train_test_split(self.X, self.y,
				train_size=percent,
				random_state=23)
			return X_train, X_test, y_train, y_test
		train_data, test_data=train_test_split(self.data,
				train_size=percent,
				random_state=23)
		return train_data, test_data

	def plot_data(self, x_axis, y_axis):
		# plot scatter plot by x and y
		x = self.data[x_axis]
		y = self.data[y_axis]
		fig, ax = plt.subplots()
		ax.scatter(x,y, s=4)
		plt.show()




