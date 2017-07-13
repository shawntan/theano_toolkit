import theano
import numpy as np
import cPickle as pickle
from functools import reduce
import inspect

import sys


class ParamsError(Exception):
    pass


class AlreadyExistsError(ParamsError):
    pass


class KeysMissingError(ParamsError):
    def __init__(self, missing_keys, src_missing_keys=[]):
        self.missing_keys = missing_keys
        self.src_missing_keys = src_missing_keys


class Parameters():

    def __init__(self, allow_overrides=False):
        self.__dict__['params'] = {}
        self.allow_overrides = allow_overrides

    def __setattr__(self, name, array):
        params = self.__dict__['params']
        if name not in params:
            params[name] = theano.shared(
                value=np.asarray(
                    array,
                    dtype=theano.config.floatX
                ),
                borrow=True,
                name=name
            )
        else:
            if self.allow_overrides:
                params[name].set_value(np.asarray(
                    array,
                    dtype=theano.config.floatX
                ))
            else:
                raise AlreadyExistsError()

    def __setitem__(self, name, array):
        self.__setattr__(name, array)

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __getattr__(self, name):
        params = self.__dict__['params']
        return params[name]

    def remove(self, name):
        del self.__dict__['params'][name]

    def values(self):
        params = self.__dict__['params']
        values = params.values()
        values.sort(key=lambda x: x.name)
        return values

    def save(self, filename):
        params = self.__dict__['params']
        with open(filename, 'wb') as f:
            pickle.dump({p.name: p.get_value() for p in params.values()}, f, 2)

    def load(self, filename, mode='strict'):
        params = self.__dict__['params']
        loaded = pickle.load(open(filename, 'rb'))
        if mode == 'strict':
            if loaded != params:
                raise KeysMissingError()

        for k in params:
            if k in loaded:
                params[k].set_value(loaded[k])
            else:
                print >> sys.stderr, "%s does not exist." % k
    def __contains__(self, key):
        return key in self.__dict__
    def __enter__(self):
        _, _, _, env_locals = inspect.getargvalues(
            inspect.currentframe().f_back)
        self.__dict__['_env_locals'] = env_locals.keys()

    def __exit__(self, type, value, traceback):
        _, _, _, env_locals = inspect.getargvalues(
            inspect.currentframe().f_back)
        prev_env_locals = self.__dict__['_env_locals']
        del self.__dict__['_env_locals']
        for k in env_locals.keys():
            if k not in prev_env_locals:
                self.__setattr__(k, env_locals[k])
                env_locals[k] = self.__getattr__(k)
        return True

    def parameter_count(self):
        import operator
        params = self.__dict__['params']
        count = 0
        for p in params.values():
            shape = p.get_value().shape
            if len(shape) == 0:
                count += 1
            else:
                count += reduce(operator.mul, shape)
        return count
