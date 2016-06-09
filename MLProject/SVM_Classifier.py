from Classifiers import *
from sklearn import svm
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
#  shot_zone_area	| unnecessary
#  shot_zone_basic	| unnecessary
#  shot_zone_range	| unnecessary
#     team_id		| unnecessary
#     team_name		| unnecessary
#     game_date		| #
#			- year	| categoricals
#			- month	| unnecessary
#			- day	| unnecessary
#      matchup		| unnecessary
#     opponent		| unnecessary
#      shot_id		| unnecessary
########################################

class SVM_Classifier(Classifier):
	def __init__(self, probability=True, log_proba=False, max_iter = 10000, verbose=False):
		super().__init__()

		self.probability = probability
		self.log_proba = log_proba
		self.max_iter = max_iter
		self.verbose = verbose

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
		self.model = svm.SVC(probability=self.probability, max_iter=self.max_iter, verbose=self.verbose)
		self.model.fit(self.train_x, self.train_y)

	def test(self):
		if not self.probability:
			result = self.model.predict(self.test_x)
		elif self.log_proba:
			result = self.model.predict_log_proba(self.test_x)
		else:
			result = self.model.predict_proba(self.test_x)
		result = [value[0] for value in result]
		self.test_y = result

	def printInSampleError(self):
		print('Ein() = ',self.model.score(self.train_x, self.train_y))

	def save(self):
		with open('SVM_classifier.pkl', 'wb') as fid:
			pickle.dump(self.model, fid)   

	def load(self):
		try:
			with open('SVM_classifier.pkl', 'rb') as fid:
				self.model = pickle.load(fid)
		except:
			print('can\'t load model')