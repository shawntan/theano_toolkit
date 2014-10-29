from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import theano
import random
import numpy as np
from collections import OrderedDict
import cPickle as pickle
class Parameters():
	def __init__(self):
		self.__dict__['params'] = {}
	
	def __setattr__(self,name,array):
		params = self.__dict__['params']
		if name not in params:
			params[name] = theano.shared(
				value = np.asarray(
					array,
					dtype = np.float32
				),
				name = name
			)
		else:
			params[name].set_value(np.asarray(
					array,
					dtype = dtype
				))
	def __setitem__(self,name,array):
		self.__setattr__(name,array)
	def __getitem__(self,name):
		return self.__getattr__(name)
	
	def __getattr__(self,name):
		params = self.__dict__['params']
		return self.params[name]
	
	def remove(self,name):
		del self.__dict__['params'][name]


	def values(self):
		params = self.__dict__['params']
		return params.values()

	def save(self,filename):
		params = self.__dict__['params']
		pickle.dump({p.name:p.get_value() for p in params.values()},open(filename,'wb'),2)

	def load(self,filename):
		params = self.__dict__['params']
		loaded = pickle.load(open(filename,'rb'))
		for k in params:
			params[k].set_value(loaded[k])
