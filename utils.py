import numpy as np
import theano
theano.config.floatX='float32'
from theano.tensor.shared_randomstreams import RandomStreams
import random
theano_rng = RandomStreams(np.random.RandomState(1234).randint(2**30))
np.random.seed(1234)
random.seed(1234)
def initial_weights(*argv):
	return 0.1 * np.random.randn(*argv)
#	scale = np.sqrt(6./sum(argv))
#	return (8 * scale) * np.random.rand(*argv) - (4 * scale)


def create_shared(array, dtype=theano.config.floatX, name=None):
	return theano.shared(
			value = np.asarray(
				array,
				dtype = dtype
			),
			name = name
		)

