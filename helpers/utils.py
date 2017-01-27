''' Author - Akshay Chawla 
    Date   - 26.01.2017

    This module contains some commonly used utilities to work with 
    the cifar-10 dataset in keras.
'''


from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import os

def genmplImg(ip_img):
	''' Reformat the image from BGR to RGB format for 
	displaying in matplotlib '''

	op_img = ip_img
	op_img[:,:,0] = ip_img[:,:,2]
	op_img[:,:,2] = ip_img[:,:,0]
	return op_img

def reformat(inputData):

	outputData = np.zeros((inputData.shape[0],3,32,32))
	for i in xrange(inputData.shape[0]):
		img = inputData[i]
		outputData[i,0,:,:] = img[:,:,0]
		outputData[i,1,:,:] = img[:,:,1]
		outputData[i,2,:,:] = img[:,:,2]
	return outputData/255.0

def oneHot(labels, num_classes=10):
	''' One hot encoding for labels '''
	one_hot = np_utils.to_categorical(labels, num_classes)
	return one_hot

def unpickle(file):
    
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def reformat_wts(ip_wt):
  op_wt = np.zeros((ip_wt.shape[0], 3, 3, 3))
  for i in xrange(ip_wt.shape[0]):
    single_wt = ip_wt[i]
    op_wt[i,:,:,0] = single_wt[0,:,:]
    op_wt[i,:,:,1] = single_wt[1,:,:]
    op_wt[i,:,:,2] = single_wt[2,:,:]
  return op_wt