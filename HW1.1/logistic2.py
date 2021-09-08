from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import numpy as np
from utils import Data


# define numpy array normalizer
def standard_scaler(x, u, s):
	return (x - u) / s

# undo the normalization
def unnormalization(x, u, s):
	return x * s + u


weight_filename = 'weight.json'
data = Data().load_file(weight_filename)
#data = json.load(open(weight_filename, 'r'))
data = [(x, y) for x, y in zip(data['y'], data['is_adult'])]
x_all = np.array([d[0] for d in data])
y_all = np.array([d[1] for d in data])

# train_data = [x for x in data if x[0] < 18]
# x_data = np.array([x[0] for x in train_data])
# y_data = np.array([x[1] for x in train_data])

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, train_size = 0.8)


x_mean = x_train.mean()
x_std = x_train.std()
y_mean = y_train.mean()
y_std = y_train.std()


# Standardlize data
x_train = standard_scaler(x_train, x_mean, x_std).flatten()
# y_train = standard_scaler(y_train, y_mean, y_std).flatten()

x_test = standard_scaler(x_test, x_mean, x_std).flatten()
# y_test = standard_scaler(y_test, y_mean, y_std).flatten()


train_mses = []
test_mses = []
train_maes = []
test_maes = []
iteration = 0


def func(x, p):

	# return p[0] + p[1] * x
	y = p[0] / (1 + np.exp(-(x-p[1])/p[2])) + p[3]
	return y


def loss(p):

	global train_mses, test_mses, train_maes, test_maes, iteration

	train_loss = mean_squared_error(func(x_train, p), y_train)
	train_mses.append(train_loss)
	
	test_loss = mean_squared_error(func(x_test, p), y_test)
	test_mses.append(test_loss)

	train_mae = mean_absolute_error(func(x_train, p), y_train)
	train_maes.append(train_mae)

	test_mae = mean_absolute_error(func(x_test, p), y_test)
	test_maes.append(test_mae)

	iteration += 1

	return train_loss


n_fit = 4
p = np.random.uniform(0.5, 1, size = n_fit)

res = minimize(loss, p, method = 'Nelder-Mead', tol = 1e-5)
popt = res.x

print(popt)

fig, ax = plt.subplots()
ax.scatter(x_all, y_all)
x = np.linspace(0, 225, 225)
x = standard_scaler(x, x_mean, x_std)
# y = popt[0] + popt[1] * x
y = func(x, popt)
x = unnormalization(x, x_mean, x_std)
# y = unnormalization(y, y_mean, y_std)

plt.plot(x, y, color = 'black')
plt.xlabel('weight(lb)')
plt.ylabel('ADULT=1 CHILD=0')
plt.title('Logistic')
plt.show()


plt.cla() 

x = list(range(1, iteration + 1))
plt.plot(x, train_mses)
plt.plot(x, test_mses)
plt.plot(x, train_maes)
plt.plot(x, test_maes)
plt.legend(['training loss', 'test MSE', 'training MAE', 'test MAE'])
plt.xlabel('optimizer iterations')
plt.ylabel('loss')
plt.show()


plt.cla()

x = standard_scaler(x_all, x_mean, x_std)
# y = popt[0] + popt[1] * x
y = func(x, popt)
# y_pred = unnormalization(y, y_mean, y_std)
y_pred = y
y_true = y_all

## find the boundaries of X and Y values
bounds = (min(min(y_true), min(y_pred)) - int(0.1 * min(y_pred)), max(max(y_true), max(y_pred))+ int(0.1 * max(y_pred)))

# Reset the limits
ax = plt.gca()
ax.set_xlim(bounds)
ax.set_ylim(bounds)
ax.plot([0, 1], [0, 1], "r-",lw=2 ,transform=ax.transAxes)

plt.scatter(y_true, y_pred)
plt.xlabel('y_real')
plt.ylabel('y_pred')
plt.title('parity plot')
plt.show()















