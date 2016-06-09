from Jay_Classifiers import *
import datetime
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

	def print_time(self):
		obj=datetime.datetime.now()
		info=(obj.hour,obj.minute,obj.second)
		print('Time : %i:%i:%i'%info)

	def generate_classifiers(self, Classifier_Type, min_valid_loss, num_classifiers, type, max_loss=1):
		classifier = Classifier_Type()
		best_loss=max_loss
		generation=1
		self.print_time()
		while generation<=num_classifiers:			
			
			score = classifier.valid()
			while score[0] > min_valid_loss:
				#rebuild.
				classifier.buildmodel()
				score = classifier.valid()
							
			#classifier selected.
			name = 'Jay_GEN'+str(type)+'_'+str(generation)+'.weight'
			classifier.weightname = name
			result=classifier.train(learning_rate=0.005, n_epoch=100,save=False,verbose=0)
			score=result['loss'][-1]

			if score < best_loss:
				print('Generated : %ith. Loss : %.4f'%(generation, score))
				name = 'Jay_GEN'+str(type)+'_T'+str(int(score*10000))+'.weight'
				classifier.weightname=name
				classifier.save_weights()
				best_loss=score
				generation+=1
				