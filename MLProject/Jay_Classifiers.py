# for FirstClassifier
import os
os.environ['THEANO_FLAGS']='floatX=float32,device=cpu'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from Classifiers import *
import numpy as np

# for SecondClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

class FirstClassifer(Classifier):
	'''
	MLP only using 'numerical attributes'.
	Never use 'categorical attributes'.
	'''
	def __init__(self, load=False):
		super().__init__()

		dramatic=True
		if dramatic:
			self.numericals = ['loc_x','loc_y', 'shot_distance']
			self.unnecessary=['game_event_id','season','team_id','team_name','matchup',\
					 'shot_id','lon','period','playoffs','lat','shot_zone_area','shot_zone_basic',\
					 'shot_zone_range','opponent','shot_type','game_date','game_id',\
					 'minutes_remaining','seconds_remaining']
			self.categoricals=['action_type','combined_shot_type']
			#self.unnecessary=['action_type','combined_shot_type','lat','lon','combined_shot_type','game_event_id','season','team_id','team_name','matchup','shot_id','lon','period','playoffs','lat']
			
			#self.categoricals=['shot_zone_area','shot_zone_basic','shot_zone_range','opponent','month','day','shot_type','period']
		else:
			self.numericals = ['lat','loc_x','loc_y', 'lon','period','playoffs','shot_distance']
			self.unnecessary=['action_type','combined_shot_type','lat','lon','combined_shot_type','game_event_id','season','team_id','team_name','matchup','shot_id']
			self.categoricals=['shot_zone_area','shot_zone_basic','shot_zone_range','opponent','month','day','shot_type','period']
		self.loaddata(True)
		#self.weightname = 'Jay_Weights.weight'
		self.weightname='Jay_Weights_T6632.weight'
		
		self.buildmodel()
		#self.build_until_good()

		if load:
			self.load_weights()
		self.compile_model()
		print('Classifier ready.')



	def build_until_good(self):
		self.buildmodel()
		score=self.valid()
		while score[0] > 0.67:
			print('Rebuild due to low valid accuracy:',score)
			self.buildmodel()
			score=self.valid()


	def buildmodel(self):
		self.model1()
		self.compile_model()
		
	def model1(self):
		self.n_input=len(self.train_x[0])
		self.n_hidden = self.n_input * 2
		self.n_output = 1
		print(self.n_input,self.n_hidden,self.n_output)
		self.model = Sequential()
		self.model.add(Dense(output_dim=self.n_hidden, input_dim=self.n_input))
		#self.model.add(Activation('relu'))
		self.model.add(LeakyReLU(0.1))
		#self.model.add(Dropout(0.4))
		self.model.add(Dense(output_dim=self.n_output))
		self.model.add(Activation('sigmoid'))

	def model2(self):
		dim0=len(self.train_x[0])
		dim1=60
		dims=[30,20,10]

		self.model = Sequential()
		self.model.add(Dense(output_dim=dim1, input_dim=dim0))
		self.model.add(Activation('relu'))
		#self.model.add(Dropout(0.5))
		for i in dims:
			self.model.add(Dense(output_dim=i))
			self.model.add(Activation('relu'))
			#self.model.add(Dropout(0.5))
		self.model.add(Dense(output_dim=1))
		self.model.add(Activation('hard_sigmoid'))

	def compile_model(self, learning_rate=0.01):
		sgd = SGD(lr=learning_rate,decay=0.99)
		self.model.compile(loss='binary_crossentropy',
					 optimizer=sgd,
					 metrics=['accuracy'])

	def run(self, n_epoch, learning_rate, verbose):
		self.compile_model(learning_rate)
		result = self.model.fit(self.train_x,self.train_y,
						  nb_epoch=n_epoch, batch_size=32,verbose=verbose,
						  validation_data=(self.valid_x, self.valid_y))
		return result.history

	def loaddata_specific(self, removed, legend):
		return (removed, legend)

	def train(self,learning_rate=0.01, n_epoch=10, save=True, verbose=2):
		result=self.run(n_epoch=n_epoch, learning_rate=learning_rate, verbose=verbose)
		if save:
			self.save_weights()
		return result

	def valid(self):
		score=self.model.evaluate(self.valid_x, self.valid_y,verbose=0)
		return score

	def load_weights(self):
		self.model.load_weights(self.weightname)
		print('Weights loaded')

	def save_weights(self):
		self.model.save_weights(self.weightname)
		print('Weights saved')

	def test(self):
		result = self.model.predict(self.test_x, batch_size=10)
		result = [value[0] for value in result]
		self.test_y = result
		return result
		
class SecondClassifier(Classifier):
	'''
	Classifier using logistic regression.
	'''
	def __init__(self):
		super().__init__()
		self.numericals = ['loc_x','loc_y', 'shot_distance']
		self.unnecessary=['game_event_id','season','team_id','matchup','team_name',\
					 'shot_id','lon','playoffs','lat',\
					 'shot_type','game_date','game_id',\
					 'combined_shot_type',\
					 ]
					 #'minutes_remaining','seconds_remaining']
		self.categoricals=['action_type','opponent','shot_zone_area','shot_zone_basic','shot_zone_range','period',]
		self.loaddata(True)
		print('Classifier ready.')

	def train(self):
		self.agent = LogisticRegression(solver='newton-cg')
		sd_idx=self.legend.index('shot_distance')
		weights=[10 if more else 1 for more in self.train_distance_large]
		weights=np.array(weights)
		use_weights=False
		if use_weights:
			self.agent.fit(self.train_x, self.train_y, weights)
		else:
			self.agent.fit(self.train_x, self.train_y)
		return self.get_train_loss()
		
	def get_train_loss(self):		
		pred_y = self.agent.predict_proba(self.train_x)
		result=[record[1] for record in pred_y]
		return log_loss(self.train_y, result)

	def valid(self):
		pred_y = self.agent.predict_proba(self.valid_x)
		result=[record[1] for record in pred_y]
		return log_loss(self.valid_y,result)

	def test(self):
		pred_y = self.agent.predict_proba(self.test_x)
		result = [record[1] for record in pred_y]
		self.test_y = result
		
