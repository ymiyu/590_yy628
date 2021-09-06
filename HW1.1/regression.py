import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd


# Loading Data
class Data:

	def __init__(self, filename):
		self.filename = str(filename)

	def load_file(self):
		with open(self.filename, 'r') as f:
			data = json.load(f)
		return data

	#def partition_data(self, percent):






data = Data('weight.json')
data = data.load_file()
#print(json.dumps(data))
print(pd.DataFrame(data))