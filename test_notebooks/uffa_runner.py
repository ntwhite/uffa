import uffa

class uffa_runner(uffa.uffa_stack):
	def __init__(self,data,split_param):
		self.class_object = uffa.uffa_stack.__init__(self,data,split_param)


	def runner_func(self):
		train, test = self.class_object.train_test_split()
		pers_model_results = self.class_object.persistence_model(train,test)
		el_rw_results = self.class_object.elementary_rw(train,test)
		el_rw_rm = self.class_object.elementary_rw_nm(train,test)
		auto_am_results = self.class_object.auto_pmd(train,test)

		report = self.class_object.report_creation(test,pers_model_results,el_rw_results,el_rw_rm,auto_am_results)
		
		viz = self.class_object.viz_train_test_predictions(train,test,auto_am_results)

		return report, viz
