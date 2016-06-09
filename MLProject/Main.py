from Jay_Classifiers import *
def main():
	print('Start program')
	classifier = FirstClassifer(False)
	
	classifier.train(learning_rate=0.012, n_epoch = 1000)
	#score=classifier.valid()
	#print('Accuracy :',score)
	
	
	result=classifier.test()
	
	print('Test results:')
	print(result)
	#classifier.export_answer()

if __name__=='__main__':
	main()