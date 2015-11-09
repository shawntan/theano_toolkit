import theano
import theano.tensor as T
import numpy as np

def log_sum_exp(x,axis=None,keepdims=False):
    k = T.max(x,axis=axis,keepdims=True)
    sum_x_ = T.log(T.sum(T.exp(x - k),axis=axis,keepdims=keepdims))
    return sum_x_ + k.reshape(sum_x_.shape)

def log_softmax(x):
    return x - log_sum_exp(x,axis=-1,keepdims=True)

def log_sigmoid(x):
    return -T.nnet.softplus(-x)

def softmax(x):
    e_x = T.exp(x - T.max(x,axis=-1,keepdims=True))
    out = e_x / T.sum(e_x,axis=-1,keepdims=True)
    return out

def log_add(x,y):
    k = T.maximum(x,y)
    return T.log(T.exp(x-k) + T.exp(y-k)) + k

def binary_crossentropy(sigmoid_x,y):
    if sigmoid_x.owner.op == T.nnet.sigmoid:
        softplus = lambda x: T.switch(x > 20,x,T.nnet.softplus(x))
        x = sigmoid_x.owner.inputs[0]
        return - (y * -softplus(-x) + (1 - y) * -softplus(x))
    else:
        return T.nnet.binary_crossentropy(sigmoid_x,y)



if __name__ == "__main__":
    X = T.dmatrix('X')
    Y = T.bmatrix('Y')
    sig_X = T.nnet.sigmoid(X)
    error_mine = binary_crossentropy(sig_X,Y)
    error_default = T.nnet.binary_crossentropy(sig_X,Y)
    f = theano.function(
            inputs=[X,Y],
            outputs=T.max(error_mine - error_default)
            )

    for i in xrange(10000):
        print f(i * np.random.randn(10,10).astype(np.float32),
                np.random.binomial(1,0.5,size=(10,10)).astype(np.int8))


