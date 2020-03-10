import numpy as np

class individual_fcast_models:

	def __int__(self, train, test) # Do I want  a " predictions" list in my class in  my  initializzation or not?   probably not
		self.training_data = np.array(train)
		self.testing_data = np.array(test)

	def persistence_model(self):
		predictions  = list() #need more eval
		history = self.training_data[-1]

		for i in range(len(self.testing_data)):
			yhat = history
			predictions.append(yhat)
			history = test[i]
		
		return predictions

	def auto_arima(self): #how do I want to deal with the  auto_arima parameters?  do I want them to be initialized with the class?  Inputted by the user?
