import numpy as np

class accuracy_measures:
	
	def __init__(self, test, predicted):
		self.test_data = np.array(test)
		self.p_data = np.array(predicted)

	def MAPE(self):
		'''
		calculates MEAN ABSOLUTE PERCENTATE ERROR for two provided arrays
		
		'''
		x = np.mean(np.abs((self.test_data - self.p_data) / self.test_data)) * 100
		return x

	def MAE(self):
		'''
		calculates the MEAN ABSOLUTE ERROR for two provided arrays

		'''
        
		x = np.mean(np.abs((self.test_data - self.p_data)))
		return x

	def MSE(self):
		'''
		calculates the MEAN SQUARED ERROR for two provided arrays

		'''
		x = np.mean((self.test_data - self.p_data)**2)
		return x

	def RMSE(self):
		'''
		calculates the ROOT MEAN SQUARED ERROR for two provided arrays

		'''
		x = np.sqrt(((self.test_data - self.p_data) ** 2).mean())
		return x

