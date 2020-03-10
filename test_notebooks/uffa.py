# unnivariate focatast feasibility assesment
import pandas as pd
import numpy as np
import pmdarima as aa
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from pmdarima.arima.utils import ndiffs, nsdiffs
from random import seed
from random import random
from tabulate import tabulate
seed(42)


class uffa_stack(object):

	def __init__(self, data, split_param):
		'''
		Summary Line: The consructor for the uffa_stack class

		Parameters:
			data : the univariate series data to be analyzed
			split_param : the training/test set split parameter

		'''

		self.data = np.array(data)
		self.split_param = int(split_param)
		self.model_name_p = 'persistence_model'
		self.model_name_el_rw = 'elementary_rw'
		self.model_name_el_rw_m = 'elementary_rw_m'
		self.model_name_auto = 'auto_arima_sarima'

	def is_number(self):
		'''
		Summary Line: Asserts that all the values in the provided data are a float or int
	
		Exteneded Description: intended as a simple type error catch function to be used in the runner_func

		Parameters:  
			self.data :the univariate data series passed in on class instantitiation

		Returns:
			boolean : a boolean value indicating if the series contains all numerical values

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
		Summary Line: Asserts that enough data has been provided to be useful, and that the split param and the series length make sense

		Extended Description: this function measures the length of the self.data object and the value of the self.split_param object.
		If the self.data object is longer than 9 observations AND the self.split_param value is less than the calculated length value, then it is technically possible to run forecasting fuctions.  However that does not mean this is the only quantitative threashold to clear before it comes usable.

		Parameters: 
			 self.data : the univariate data series passed in on class instantiation
			 self.split_param : the train/test set split param passed in on class instantiation

		Returns:
			boolean =  a boolean value indicating wheather the split_param value makes sense giving the data series
		'''
		length = len(self.data)
		boolean  = True

		if length > 9 and self.split_param < length:
			pass
		else:
			boolean = False

		return boolean

	def train_test_split(self):
		'''
		Summary Line: splits data series based off split_param provided at class instantiation

		Extended Description: only does a train/test split; but thats all this package is intented to facilitiate.

		Parameters: 
			self.data : the data object created on class instantiation
			self.split_param : the split_param created on class instantitation

		Returns:
			train (array) : the training dataset
			test (array) : the test dataset
	
		'''
		length = len(self.data)
		val = length - self.split_param

		train = self.data[:val]
		test = self.data[val:]

		return train, test

	def persistence_model(self, train, test):
		'''
		Summary Line: produces a persistance model

		Extended Description: a persistance model simply predicts t-1 for any given t, assuming t = time.  This is also commonly called a "naive forecast"

		Parameters:
			train (array) : the training dataset 
			test (array) : the test dataset

		Returns
			predictions (array) : the predictions produced by the naive algorithim

		'''
		predictions = list()  # need more eval
		history = train[-1]

		for i in range(len(test)):
			yhat = history
			predictions.append(yhat)
			history = test[i]

		predictions = np.array(predictions)

		return predictions

	def elementary_rw(self, train, test):
		'''
    		Summary Line: create an elementary rw model where step move sizes have an equal probability of moving up or down,
		and step size equals the standard deviation of the training set

		Extended Description:  This idea was dumb, and will probably be supressed from the runner function

		Parameters:
			train (array) : the training dataset
			test (array) : the test dataset

		Returns:
			predictions (array) : the predictions produced by the random_walk aglo
		'''
		seed(1)
		stand_dev = np.std(train)
		start_value = train[-1]
		predictions = list()

		for i in test:
			movement = -stand_dev if random() < 0.5 else stand_dev
			value = start_value + movement
			start_value = value
			predictions.append(value)
		predictions  = np.array(predictions)

		return predictions

	def elementary_rw_nm(self,train,test):
		'''
		Summary Line: create an elementary rw model where step move sizes have an equal probability of moving up or down or staying the same,
		and step size equals the standard deviation of the training set

		Extended Description:  This idea was dumb and will probably be supressed from the runner function

		Parameters:
			train (array) : the trainig dataset
			test (array) : the test dataset

		Returns:
			predictions (array) : gthe predictions produced by the random_walk_median algo

		'''
		seed(1)
		stand_dev = np.std(train)
		start_value = train[-1]
		predictions = list()

		for i in test:
			num = random()
			if num <= 0.33:
				movement = stand_dev
			elif num <= 0.66:
				movement = -stand_dev
			else:
				movement = 0
				
			value = start_value + movement
			start_value = value
			predictions.append(value)
		predictions = np.array(predictions)

		return predictions

	def auto_pmd(self,train,test):
		'''
		Summary Line: Create an auto_arima

		Extended

		'''
		little_d = ndiffs(train, test = 'kpss')
		big_D = nsdiffs(train, m = 52, max_D = 12, test = 'ocsb')

		model_1 = aa.auto_arima(train, 
	                        start_p = 0,
	                        start_q = 0,
	                        max_p = 5,
	                        max_q = 5,
	                        m = 52,
	                        start_P = 0,
	                        seasonal = True,
	                        d = little_d,
	                        D = big_D,
	                        suppress_warnings = True,
	                        stepwise = True,
	                        error_action = 'ignore',
	                        trace = False)

		predictions = model_1.predict(n_periods = len(test))
		predictions = np.array(predictions)
		return predictions

	def all_acc(self,test_data,p_data):
		'''
		calculates MEAN ABSOLUTE PERCENTATE ERROR for two provided arrays
		'''
		mape = np.mean(np.abs((test_data - p_data) / test_data)) * 100
		mae = np.mean(np.abs((test_data - p_data)))
		mse = np.mean((test_data - p_data)**2)
		rmse = np.sqrt(((test_data - p_data) ** 2).mean())
		df2 = pd.DataFrame({
							'MAPE' : [mape],
							'MAE' : [mae],
							'MSE' : [mse],
							'RMSE' : [rmse]})
		return df2
	
	def report_creation(self,test,pers_results,rw_results,rwm_results,auto_am_results):
		pers_acc = self.all_acc(test,pers_results)
		el_rw_acc = self.all_acc(test,rw_results)
		el_rw_m_acc = self.all_acc(test,rwm_results)
		auto_am_acc = self.all_acc(test,auto_am_results)
		x = pd.concat([pers_acc, el_rw_acc, el_rw_m_acc, auto_am_acc])
		x['model'] = ['persistence','random_walk','random_walk_m','auto_arima']
		return x

	def viz_train_test_predictions(self,train,test,fcast):
		test = pd.Series(test)
		fcast = pd.Series(fcast)
		index_end  = len(self.data)
		index_start = index_end - len(test)
		test.index = range(index_start,index_end)
		fcast.index = test.index

		plt.plot(train, label = "train")
		plt.plot(test, label = "test")
		plt.plot(fcast, label = "fcast")
		plt.title("Train-Test-Forecast Plot")
		plt.legend(loc = "lower left")
		plt.show()

	def viz_all_forecast_models(self,test, pers_results, auto_am_results):
		plt.plot(test, label = "test")
		plt.plot(pers_results,label = "persistence")
		#plt.plot(rw_results, label = "random_walk")
		#plt.plot(rwm_results, label = "random_walk_wm")
		plt.plot(auto_am_results, label = "best arima/sarimax")
		plt.title("Forecast Performance Plot")
		plt.legend(loc = "lower left")
		plt.show()

	def runner_func(self):
		train, test = self.train_test_split()
		pers_model_results = self.persistence_model(train,test)
		## Hold out bad stuff
		#el_rw_results = self.elementary_rw(train,test)
		#el_rw_rm = self.elementary_rw_nm(train,test)
		auto_am_results = self.auto_pmd(train,test)

		report = self.report_creation(test, pers_model_results, el_rw_results, el_rw_rm, auto_am_results)
		print(tabulate(report, headers = 'keys', tablefmt = 'psql'))
		print('\n')
		print(test)
		self.viz_train_test_predictions(train,test,auto_am_results)
		plt.clf()
		self.viz_all_forecast_models(test, pers_model_results, auto_am_results)
