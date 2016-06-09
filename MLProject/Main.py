from Jay_Classifiers import *
from Meta_Classifier import *
def main():
	print('Start program')
	getter = MetaClassifier()
	classifier = getter.get_classifier(FirstClassifer, 0.68)
	#classifier = FirstClassifer(False)
	classifier.train(learning_rate=0.005, n_epoch = 200)
	
	result=classifier.test()
	
	classifier.export_answer()

if __name__=='__main__':
	main()