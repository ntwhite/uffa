import numpy as np

class accuracy_measures:
	
	def __init__(self, test, predicted):
		self.test_data = np.array(test)
		self.p_data = np.array(predicted)

	def MAPE(self):
		'''
		calculates MEAN ABSOLUTE PERCENTATE ERROR for two provided arrays
		
		'''
		mape = np.mean(np.abs((self.test_data - self.p_data) / self.test_data)) * 100
		return mape 

	def MAE(self):
		'''
		calculates the MEAN ABSOLUTE ERROR for two provided arrays

		'''
		mae = np.mean(np.abs((self.test_data - self.p_data)))
		return mae

	def MSE(self):
		'''
		calculates the MEAN SQUARED ERROR for two provided arrays

		'''
		mse = np.mean((self.test_data - self.p_data)**2)
		return mse

	def RMSE(self):
		'''
		calculates the ROOT MEAN SQUARED ERROR for two provided arrays

		'''
		rmse = np.sqrt(((self.test_data - self.p_data) ** 2).mean())
		return rmse

