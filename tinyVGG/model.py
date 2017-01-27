''' Author - Akshay Chawla 
    Date   - 26.01.2017

    This module contains the architecture and implementation of 
    a scaled down VGGNet model. This model is thus named tinyVGGNet.

    In accordance with the full-scale model, all convolutions are 
    performed using F=3 and S=1. Additionally, max-pooling
    is performed with F=2 and a stride of S=1. This is done so that 
    the Conv filters are responsible for extracting features and the
    pooling layers are responsible for downsampling and retaining only
    the most important features.   

'''

from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import numpy as np 

class tinyVGGNet():
	
	@staticmethod
	def build(width,height,depth,classes):
		''' Build a tiny VGG net as shown on the stanford cs231n page to classify CIFAR-10 dataset'''

		# init model 
		model = Sequential()

		# First Set of CONV->RELU->POOL
		model.add(ZeroPadding2D(padding=(1,1), input_shape=(depth,height,width)))
		model.add(Convolution2D(16, 3, 3, subsample=(1,1), activation="relu"))
		model.add(ZeroPadding2D(padding=(1,1)))
		model.add(Convolution2D(16, 3, 3, subsample=(1,1), activation="relu"))
		model.add(MaxPooling2D(pool_size=(2,2)))

		# Second set of CONV->RELU->POOL
		model.add(ZeroPadding2D(padding=(1,1)))
		model.add(Convolution2D(32, 3, 3, subsample=(1,1), activation="relu"))
		model.add(ZeroPadding2D(padding=(1,1)))
		model.add(Convolution2D(32, 3, 3, subsample=(1,1), activation="relu"))
		model.add(MaxPooling2D(pool_size=(2,2)))
		
		# Third set 
		model.add(ZeroPadding2D(padding=(1,1)))
		model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation="relu"))
		model.add(ZeroPadding2D(padding=(1,1)))
		model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation="relu"))
		model.add(MaxPooling2D(pool_size=(2,2)))

		# FC layer
		model.add(Flatten())
		model.add(Dense(10))
		model.add(Activation("softmax"))

		return model