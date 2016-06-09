from Jay_Classifiers import *
from Meta_Classifier import *
def main():
	print('Start program')
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
	''' 
	a=
	print(a)
	result=classifier.test()
	
	classifier.export_answer()
	'''

if __name__=='__main__':
	main()