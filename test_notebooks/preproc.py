import numpy as np

class pre_processing:

	def __init__(self, data, split_param):
		self.data = np.array(data)
		self.split_param = int(split_param)

	def is_number(self):
		'''
		Asserts that all the values in the provided data are a float or int
		'''
		boolean = True
		
		for i in self.data:
			try:
				number = float(i)
			except:
				boolean = False
		
		return boolean

	def length_feas(self):
		'''
		Asserts that enough data has been provided to be useful, and that the slit param and the series length make sense
		'''
		length = len(self.data)
		feasibility = True

		if length > 9 and self.split_param < length:
			pass
		else:
			feasibility = False

		return feasibility

	def train_test_split(self):
		'''
		splits data series based off split_param provided at class instantiation
		'''
		length = len(self.data)
		val = length - self.split_param

		train = self.data[:val]
		test = self.data[val:]

		return train, test
