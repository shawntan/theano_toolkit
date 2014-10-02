from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as T
import theano
import random
import numpy as np
from collections import OrderedDict
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
	
	def __getattr__(self,name):
		params = self.__dict__['params']
		return self.params[name]

	def values(self):
		params = self.__dict__['params']
		return params.values()




