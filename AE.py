import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from keras.layers import BatchNormalization, Input, Lambda
from keras import regularizers
# Use seaborn style defaults and set the default figure size

class MyModel(tf.keras.Model):

	def __init__(self):
		super(MyModel, self).__init__()
		self.dense1 = tf.keras.layers.Dense(59, activation=tf.nn.tanh ,use_bias = True)
		self.dense2 = tf.keras.layers.Dense(50, activation=tf.nn.tanh,use_bias = True)
		self.dense3 = tf.keras.layers.Dense(40, activation=tf.nn.tanh,use_bias = True)

		self.dense4 = tf.keras.layers.Dense(30, activation=tf.nn.tanh,use_bias = True)
		self.dense5 = tf.keras.layers.Dense(20, activation=tf.nn.tanh,use_bias = True)
		self.dense6 = tf.keras.layers.Dense(15, activation=tf.nn.tanh,use_bias = True)
		self.dense7 = tf.keras.layers.Dense(10, activation=tf.nn.tanh,use_bias = True)

		self.dense8 = tf.keras.layers.Dense(5, activation=tf.nn.tanh,use_bias = True)
		self.dense9 = tf.keras.layers.Dense(10, activation=tf.nn.tanh,use_bias = True)


		self.dense10 = tf.keras.layers.Dense(15, activation=tf.nn.tanh,use_bias = True)

		self.dense11 = tf.keras.layers.Dense(20, activation=tf.nn.tanh,use_bias = True)

		self.dense12 = tf.keras.layers.Dense(40, activation=tf.nn.tanh,use_bias = True)

		self.dense13 = tf.keras.layers.Dense(50, activation=tf.nn.tanh,use_bias = True)

		self.dense14 = tf.keras.layers.Dense(59, activation=tf.nn.tanh,use_bias = True)


	def encode(self,X):
		
		x= self.dense1(X)
		x= self.dense2(x)
		x= self.dense3(x)
		x= self.dense4(x)
		x= self.dense5(x)
		x= self.dense6(x)
		x= self.dense7(x)
	 
		return self.dense8(x)

	def decode(self,H):
		return self.dense14(H)

	def call(self, inputs):
		x = self.dense1(inputs)
		x = self.dense2(x)
		x = self.dense3(x)
		x = self.dense4(x)
		x = self.dense5(x)
		x = self.dense6(x)
		x = self.dense7(x)
		x = self.dense8(x)
		x = self.dense9(x)
		x = self.dense10(x)
		x = self.dense11(x)
		x = self.dense12(x)
		x= self.dense13(x)
		
		

		return self.dense14(x)