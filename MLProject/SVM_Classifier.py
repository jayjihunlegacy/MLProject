from Classifiers import *
from sklearn import svm, preprocessing
from math import log
import scipy as sp
import pickle

################ LEGEND ################
#    action_type	| categoricals
# combined_shot_type| unnecessary
#   game_event_id	| unnecessary
#      game_id		| unnecessary
#        lat		| unnecessary
#       loc_x		| numericals
#       loc_y		| numericals
#        lon		| unnecessary
# minutes_remaining	| #
#      period		| categoricals
#     playoffs		| unnecessary
#      season		| unnecessary
# seconds_remaining	| #
#   shot_distance	| numericals
#  shot_made_flag	| #
#    shot_type		| categoricals
#  shot_zone_area	| categoricals
#  shot_zone_basic	| categoricals
#  shot_zone_range	| categoricals
#     team_id		| unnecessary
#     team_name		| unnecessary
#     game_date		| #
#			- year	| categoricals
#			- month	| unnecessary
#			- day	| unnecessary
#      matchup		| unnecessary
#     opponent		| categoricals
#      shot_id		| unnecessary
########################################

class SVM_Classifier(Classifier):
	def __init__(self, probability=True, log_proba=False, max_iter = 10000, verbose=False, vaildOn=False):
		super().__init__()

		self.probability = probability
		self.log_proba = log_proba
		self.max_iter = max_iter
		self.verbose = verbose
		self.vaildOn = vaildOn

		self.numericals = ['loc_x','loc_y','shot_distance']
		self.unnecessary=['combined_shot_type','game_event_id','lat','lon','playoffs','season','shot_zone_area','shot_zone_basic', 'shot_zone_range',
							'team_id','team_name','matchup','opponent','shot_id']
		self.categoricals=['action_type','period','shot_type']
	
	def OneHotEncoding(self, total, legend):
		#One-hot encode 'Categorical attributes'
		cat_indices={}
		categories={}
		keys={}
		next_keys={}
		for attr in self.categoricals:
			categories[attr]={}
			cat_indices[attr] = legend.index(attr)
			next_keys[attr]=0
			keys[attr]=[]

		#1. Observe which attributes exist.
		for record in total:
			for attr in self.categoricals:
				idx = cat_indices[attr]
				if record[idx] not in categories[attr].keys():
					categories[attr][record[idx]] = next_keys[attr]
					next_keys[attr]+=1
					keys[attr].append(record[idx])

		#2. Edit records and legend.
		for attr in self.categoricals:
			idx = cat_indices[attr]
			classes = len(keys[attr])
			# Edit legend.
			for one_hot in range(classes):
				new_name = '_'+attr+'('+str(keys[attr][one_hot])+'['+str(one_hot)+'])'
				legend.append(new_name)

			# Edit all records.
			for record in total:			
				class_i = categories[attr][record[idx]]
				for one_hot in range(classes):
					if one_hot == class_i:
						record.append(1)
					else:
						record.append(0)
		#3. Remove old from records and legend.		
		for idx in reversed(range(len(legend))):
			attr = legend[idx]
			if attr in self.categoricals:
				legend.remove(attr)
				for record in total:
					record.pop(idx)

	def loaddata_specific(self, removed, legend):
		postUnnecessary = ['month', 'day']
		for attr in postUnnecessary:
			idx=legend.index(attr)
			removed = [record[:idx] + record[idx+1:] for record in removed]
			legend.remove(attr)
		self.OneHotEncoding(removed, legend)
		return (removed, legend)

	def train(self):
		#self.model = svm.LinearSVC(probability=self.probability, max_iter=self.max_iter, verbose=self.verbose)
		self.model = svm.SVC(C=0.01, probability=self.probability, max_iter=self.max_iter, verbose=self.verbose)
		if not self.vaildOn:
			self.model.fit(self.train_x, self.train_y)
		else:
			self.model.fit(self.train_x, self.train_y)

	def test(self, data):
		if not self.probability:
			result = self.model.predict(data)
		elif self.log_proba:
			result = self.model.predict_log_proba(data)
		else:
			result = self.model.predict_proba(data)
		result = [value[1] for value in result]
		return result

	def stdScaler(self):
		stdScaler = preprocessing.StandardScaler().fit(self.train_x)
		stdScaler.transform(self.train_x)
		stdScaler = preprocessing.StandardScaler().fit(self.valid_x)
		stdScaler.transform(self.valid_x)

	def printInSampleError(self):
		if self.vaildOn:
			print('In-sample accuarcy = ',self.model.score(self.train_x, self.train_y))
			print('vaild-sample accuarcy = ', self.model.score(self.valid_x, self.valid_y))
		else:
			print('In-sample accuarcy = ',self.model.score(self.train_x, self.train_y))

	def save(self):
		with open('SVM_classifier.pkl', 'wb') as fid:
			pickle.dump(self.model, fid) 

	def load(self):
		try:
			with open('SVM_classifier.pkl', 'rb') as fid:
				self.model = pickle.load(fid)
		except:
			print('can\'t load model')

	def printLogLoss(self):
		if self.vaildOn:
			vaild_ll = self.test(self.valid_x)
			vaild_ll = self.postProcess(vaild_ll)
			print('vaild log loss = ',self.logloss(self.valid_y, vaild_ll))

	def logloss(self, act, pred):
		epsilon = 1e-15
		pred = sp.maximum(epsilon, pred)
		pred = sp.minimum(1-epsilon, pred)
		ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
		ll = ll * -1.0/len(act)
		return ll

	def postProcess(self, result):
		returnValue = []
		ratio=1.0
		for value in result:
			temp = 0.5 + ratio*(value-0.5)
			returnValue.append(temp)
		return returnValue