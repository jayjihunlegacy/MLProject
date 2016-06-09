from Jay_Classifiers import *
def main():
	print('Start program')
	classifier = FirstClassifer(False)
	classifier.train(learning_rate=0.01, n_epoch = 200)
	
	result=classifier.test()
	
	classifier.export_answer()

if __name__=='__main__':
	main()