import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams
theano_rng = RandomStreams(np.random.RandomState(1234).randint(2**30))

def initial_weights(*argv):
	"""
	return np.random.uniform(
			low  = -4 * np.sqrt(6./(visible+hidden)),
			high =  4 * np.sqrt(6./(visible+hidden)),
			size = (visible,hidden))
	"""
	return 0.1 * np.random.randn(*argv)

def create_shared(array, dtype=theano.config.floatX, name=None):
	return theano.shared(
			value = np.asarray(
				array,
				dtype = dtype
			),
			name = name
		)

