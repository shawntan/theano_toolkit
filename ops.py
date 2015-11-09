import theano
import theano.tensor as T
import numpy as np

def log_sum_exp(x,axis=None,keepdims=False):
    k = T.max(x,axis=axis,keepdims=True)
    sum_x_ = T.log(T.sum(T.exp(x - k),axis=axis,keepdims=keepdims))
    return sum_x_ + k.reshape(sum_x_.shape)

def log_softmax(x):
    return x - log_sum_exp(x)

def log_sigmoid(x):
    return -T.nnet.softplus(-x)

def softmax(x):
    e_x = T.exp(x - T.max(x,axis=-1,keepdims=True))
    out = e_x / T.sum(e_x,axis=-1,keepdims=True)
    return out

def log_add(x,y):
    k = T.maximum(x,y)
    return T.log(T.exp(x-k) + T.exp(y-k)) + k

if __name__ == "__main__":
    print T.exp(log_add(
            T.log(np.array([1.5,0.5])),
            T.log(np.array([[1,1],[2,2]]))
            )).eval()
