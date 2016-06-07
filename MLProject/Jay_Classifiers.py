﻿import os
os.environ['THEANO_FLAGS']='floatX=float32,device=cpu'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from Classifiers import *
import numpy as np

class FirstClassifer(Classifier):
	'''
	MLP only using 'numerical attributes'.
	Never use 'categorical attributes'.
	'''
	def __init__(self, load):
		super().__init__()
		self.loaddata()
		self.weightname = 'Jay_Weights.weight'

		self.buildmodel()

		if load:
			self.load_weights()
		self.compile_model()

	def buildmodel(self):
		self.n_input=len(self.train_x[0])
		self.n_hidden = self.n_input // 2
		self.n_output = 1
		print(self.n_input, self.n_hidden, self.n_output)
		self.model = Sequential()
		self.model.add(Dense(output_dim=self.n_hidden, input_dim=self.n_input))
		self.model.add(Activation('relu'))
		#self.model.add(LeakyReLU(0.1))
		#self.model.add(Dropout(0.4))
		self.model.add(Dense(output_dim=self.n_output))
		self.model.add(Activation('sigmoid'))
		

	def compile_model(self, learning_rate=0.01):
		sgd = SGD(lr=learning_rate,decay=0.99)
		self.model.compile(loss='binary_crossentropy',
					 optimizer=sgd,
					 metrics=['accuracy'])

	def run(self, n_epoch, learning_rate):
		self.compile_model(learning_rate)
		result = self.model.fit(self.train_x,self.train_y,
						  nb_epoch=n_epoch, batch_size=100,verbose=2,
						  validation_data=(self.valid_x, self.valid_y))
		return result.history

	def loaddata_specific(self, total, legend):
		numerize_category=True
		#1. Numerize 'Numerical categorical attributes'
		for idx, attr in enumerate(legend):
			if attr=='shot_type':
				categories={'2PT Field Goal' : 1, '3PT Field Goal' : 2}
			else:
				continue

			for idx_record, record in enumerate(total):
				total[idx_record][idx]=categories[record[idx]]
		
		#2. One-hot encode 'Categorical attributes'
		cat_attrs=['shot_zone_area','shot_zone_basic','shot_zone_range','opponent','month','day','shot_type','period','action_type']
		cat_indices={}
		categories={}
		keys={}
		next_keys={}
		for attr in cat_attrs:
			categories[attr]={}
			cat_indices[attr] = legend.index(attr)
			next_keys[attr]=0
			keys[attr]=[]

		#2-1. Observe which attributes exist.
		for record in total:
			for attr in cat_attrs:
				idx = cat_indices[attr]
				if record[idx] not in categories[attr].keys():
					categories[attr][record[idx]] = next_keys[attr]
					next_keys[attr]+=1
					keys[attr].append(record[idx])

		#2-2. Edit records and legend.
		for attr in cat_attrs:
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
		#2-3. Remove old from records and legend.		
		for idx in reversed(range(len(legend))):
			attr = legend[idx]
			if attr in cat_attrs:
				legend.remove(attr)
				for record in total:
					record.pop(idx)

		
		return (total, legend)

	def train(self,learning_rate=0.01, n_epoch=10):
		self.run(n_epoch=n_epoch, learning_rate=learning_rate)
		self.save_weights()

	def valid(self):
		score=self.model.evaluate(self.valid_x, self.valid_y)
		return score[1]

	def load_weights(self):
		self.model.load_weights(self.weightname)
		print('Loaded!')

	def save_weights(self):
		self.model.save_weights(self.weightname)
		print('Saved!')

	def test(self):
		result = self.model.predict(self.test_x, batch_size=10)
		result = [1 if value[0]>0.5 else 0 for value in result]
		self.test_y = result
		return result
		