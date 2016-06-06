import os
os.environ['THEANO_FLAGS']='floatX=float32,device=cpu'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD
from Main import import_data

class FirstClassifer(object):
	'''
	MLP only using 'numerical attributes'.
	Never use 'categorical attributes'.
	'''
	def __init__(self):
		self.loaddata()

		self.buildmodel()

	def buildmodel(self):
		self.n_input=len(self.train_x[0])
		self.n_hidden = self.n_input//2
		self.n_output = 2
		print(self.n_input, self.n_hidden, self.n_output)

		self.model = Sequential()
		self.model.add(Dense(output_dim=self.n_hidden, input_dim=self.n_input))
		self.model.add(LeakyReLU(0.1))
		self.model.add(Dense(output_dim=self.n_output))
		self.model.add(Activation('softmax'))

	def compile_model(self, learning_rate=0.01):
		sgd = SGD(lr=learning_rate,decay=0.99)
		self.model.compile(loss='categorical_crossentropy',
					 optimizer=sgd,
					 metrics=['accuracy'])

	def run(self, n_epoch, learning_rate):
		self.compile_model(learning_rate)
		result = self.model.fit(self.train_x,self.train_y,
						  nb_epoch=n_epoch, batch_size=10,verbose=2,
						  validation_data=(self.valid_x, self.valid_y))
		return result.history

	def loaddata(self):
		train_set, valid_set, test_x= import_data(
			numerize=True,
			numerize_category=True,
			process_attribute=True
			)

		self.train_x, self.train_y = train_set
		self.valid_x, self.valid_y = valid_set
		self.test_x = test_x

		self.train_y = np_utils.to_categorical(self.train_y,2)
		self.valid_y = np_utils.to_categorical(self.valid_y,2)

	def train(self,learning_rate=0.01, n_epoch=10):
		self.run(n_epoch=n_epoch, learning_rate=learning_rate)

	def valid(self):
		score=self.model.evaluate(self.valid_x, self.valid_y)
		return score[1]

	def test(self):
		pass