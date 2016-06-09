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

	def generate_classifiers(self, Classifier_Type, min_valid_loss, num_classifiers, max_loss=1):
		classifier = Classifier_Type()
		best_loss=max_loss
		generation=1
		while generation!=num_classifiers:			
			
			score = classifier.valid()
			while score[0] > min_valid_loss:
				#rebuild.
				classifier.buildmodel()
				score = classifier.valid()
							
			#classifier selected.
			name = 'Jay_GEN_'+str(generation)+'.weight'
			classifier.weightname = name
			result=classifier.train(learning_rate=0.005, n_epoch=100,save=False)
			score=result['loss'][-1]

			if score < best_loss:
				classifier.save_weights()
				best_loss=score
				generation+=1