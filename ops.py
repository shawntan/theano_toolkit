import theano.tensor as T


def log_sum_exp(x, axis=None, keepdims=False):
    k = T.max(x, axis=axis, keepdims=True)
    sum_x_ = T.log(T.sum(T.exp(x - k), axis=axis, keepdims=keepdims))
    return sum_x_ + k.reshape(sum_x_.shape)


def log_softmax(x):
    return x - log_sum_exp(x, axis=-1, keepdims=True)


def log_sigmoid(x):
    return -T.nnet.softplus(-x)


def softmax(x):
    e_x = T.exp(x - T.max(x, axis=-1, keepdims=True))
    out = e_x / T.sum(e_x, axis=-1, keepdims=True)
    return out


def log_add(x, y):
    k = T.maximum(x, y)
    return T.log(T.exp(x - k) + T.exp(y - k)) + k


def binary_crossentropy(sigmoid_x, y):
    if sigmoid_x.owner.op == T.nnet.sigmoid:
        def softplus(x):
            return T.switch(x > 20, x, T.nnet.softplus(x))
        x = sigmoid_x.owner.inputs[0]
        return - (y * -softplus(-x) + (1 - y) * -softplus(x))
    else:
        return T.nnet.binary_crossentropy(sigmoid_x, y)
