from Jay_Classifiers import *
from Meta_Classifier import *

def do_with_First():
	getter = MetaClassifier()
	'''
	getter.generate_classifiers(
		Classifier_Type=FirstClassifer,
		min_valid_loss=0.7,
		num_classifiers=2,
		type=2,
		max_loss=0.67)
	'''
	classifier = FirstClassifer(False)
	#classifier = getter.get_classifier(FirstClassifer,0.69)
	classifier.train(learning_rate=0.005, n_epoch = 100)

	#result=classifier.test()
	
	#classifier.export_answer()

def do_with_Second():
	classifier = SecondClassifier()
	train_loss = classifier.train()
	valid_loss = classifier.valid()
	print('Train loss : %.4f, Valid loss : %.4f'%(train_loss, valid_loss))

	test=True

	if test:
		classifier.test()
		classifier.export_answer()

def main():
	print('Start program')
	do_with_Second()
	

if __name__=='__main__':
	main()