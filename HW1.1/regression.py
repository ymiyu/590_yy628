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
			json_file = json.load(f)
			self.data = pd.DataFrame(json_file)
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



# Linear Regression
# Load data
data = Data()
data.load_file("weight.json")
#print("DATA")
#print(data.data)

# Slice ages to be less than 18
data.data = data.data[data.data['x']<18]

# Set predictors and target
data.set_variables(["x"], "y")
#print(data.X)

# Split data
X_train, X_test, y_train, y_test = data.split_data(0.8)
#print("X_train, y_train")
#print(X_train, y_train)

# Standardlize data
X_train = standard_scaler(X_train.to_numpy()).flatten()
y_train = standard_scaler(y_train.to_numpy())
X_test = standard_scaler(X_test.to_numpy()).flatten()
y_test = standard_scaler(y_test.to_numpy())


#print(standard_scaler(X_train))
#data.plot_data(x_axis='x', y_axis='y')
#print(standard_scaler(y_train))

# Scipy Optimzier Linear Regression
def fun(x, t, y):
	return np.mean((x[0]+x[1]*np.exp(x[2]*t)-y)**2)

x0 = np.array([1.0, 1.0, 0.0])
res_lsq = least_squares(fun, x0, args=(X_train, y_train))

def model(x, p1,p2):
	return p1+x*p2
popt, pcov = curve_fit(model, X_train, y_train)

print(popt)
# Plot
data = Data()
data.load_file("weight.json")
data.set_variables(["x"], "y")

X_train, X_test, y_train, y_test = data.split_data(0.8)
X_train = X_train.to_numpy().flatten()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy().flatten()
y_test = y_test.to_numpy()



fig, ax = plt.subplots()
ax.scatter(X_train, y_train)
x = np.linspace(0, 18, 100)
y = popt[0] + popt[1]*x
plt.plot(x, y)
plt.show()


