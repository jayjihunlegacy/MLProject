from Jay_Classifiers import *
from SVM_Classifier import *
from Meta_Classifier import *
def main():
	
	print('Start program')
	SVM_HanDa = SVM_Classifier(probability=True, log_proba=False, max_iter=10000, verbose=True, vaildOn=True)
	
	print('loading data')
	SVM_HanDa.loaddata(print_legend=True)
	
	print('Scaling')
	SVM_HanDa.stdScaler()
	
	print('training')
	SVM_HanDa.train()
	SVM_HanDa.printInSampleError()
	

	print('saving')
	SVM_HanDa.save()
	
	'''
	print('loading')
	SVM_HanDa.load()
	
	'''

	SVM_HanDa.printLogLoss()

	#print('testing')
	#SVM_HanDa.test_y = SVM_HanDa.test(self.test_x)
	#SVM_HanDa.export_answer()
	'''
	
	getter = MetaClassifier()
	getter.generate_classifiers(
		Classifier_Type=FirstClassifer,
		min_valid_loss=0.7,
		num_classifiers=2,
		type=2,
		max_loss=0.67)
	#classifier = FirstClassifer(False)
	#classifier = getter.get_classifier(FirstClassifer,0.68)
	classifier.train(learning_rate=0.005, n_epoch = 100)
	a=
	print(a)
	result=classifier.test()
	
	classifier.export_answer()
	'''

if __name__=='__main__':
	main()