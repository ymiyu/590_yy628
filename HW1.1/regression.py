import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from sklearn.model_selection import train_test_split


# Loading Data
class Data:

	def __init__(self, filename):
		self.filename = str(filename)

	def load_file(self):
		with open(self.filename, 'r') as f:
			data = json.load(f)
			data = pd.DataFrame(data)
		return data

	#def partition_data(self, percent):

	def split(self, percent):







data = Data('weight.json')
data = data.load_file()
#print(json.dumps(data))
data = json.dumps(data)
print(len(data))

#print(pd.DataFrame(data))