from Jay_Classifiers import *

class MetaClassifier(object):
	def __init__(self):
		self.classifier = None

	def get_classifier(self, Classifier_Type, min_valid_loss):
		self.classifier = Classifier_Type()
		score = self.classifier.valid()
		while score[0] > min_valid_loss:
			print('Rebuild due to high valid loss : %.4f'%(score[0],))
			self.classifier.buildmodel()
			score = self.classifier.valid()
		return self.classifier

