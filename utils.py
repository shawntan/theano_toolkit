import numpy as np
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import random
theano_rng = RandomStreams(np.random.RandomState(1234).randint(2**30))
np.random.seed(1234)
random.seed(1234)
def initial_weights(*argv):
	"""
	return np.random.uniform(
			low  = -4 * np.sqrt(6./(visible+hidden)),
			high =  4 * np.sqrt(6./(visible+hidden)),
			size = (visible,hidden))
	"""
#	return 0.1 * np.random.randn(*argv)
	if len(argv) == 2:
		scale = np.sqrt(6./(argv[0]+argv[1]))
		return (8 * scale) * np.random.rand(*argv) - (4 * scale)
	else:
		return 0.5 * np.random.randn(*argv)


def create_shared(array, dtype=theano.config.floatX, name=None):
	return theano.shared(
			value = np.asarray(
				array,
				dtype = dtype
			),
			name = name
		)

