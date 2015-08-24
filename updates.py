from itertools import izip
import theano
import theano.tensor as T
import numpy         as np
import utils         as U
from parameters import Parameters

def clip(magnitude):
    def clipper(deltas):
        grads_norms = [ T.sqrt(T.sum(T.sqr(g))) for g in deltas ]
        return [ 
            T.switch(
                T.gt(n,magnitude),
                magnitude * (g/n),g
            ) for n,g in zip(grads_norms,deltas)
        ]
    return clipper


def track_parameters(update_fun):
    def decorated_fun(parameters,gradients,**kwargs):
        if "P" not in kwargs:
            kwargs["P"] = Parameters()
        if "delta_preprocess" in kwargs:
            delta_preprocess = kwargs["delta_preprocess"]
            del kwargs["delta_preprocess"]
        else: delta_preprocess = lambda x: x
        deltas, updates = update_fun(parameters,gradients,**kwargs)
        deltas = delta_preprocess(deltas)
        assert(len(deltas) == len(parameters))
        return zip(parameters,( p - d for p,d in izip(parameters,deltas) )) + updates
    return decorated_fun
        
def create_param(P,name,w):
    P[name] = w
    return P[name]

def get_shapes(parameters):
    return [ p.get_value().shape for p in parameters ]

@track_parameters
def adadelta(parameters,gradients,rho=np.float32(0.95),learning_rate=np.float32(1e-4),P=None):
    eps = learning_rate
    shapes = get_shapes(parameters)

    acc_gradients_sq = [ create_param(P,"grad_sq_" + p.name,np.zeros(s))   for p,s in izip(parameters,shapes) ]
    acc_deltas_sq    = [ create_param(P,"deltas_sq_" + p.name,np.zeros(s)) for p,s in izip(parameters,shapes) ]

    gradients_sq = [ T.sqr(g) for g in gradients ]
    gradients_sq_new = [ rho * acc_g_sq + (np.float32(1.) - rho) * g_sq for acc_g_sq,g_sq in izip(acc_gradients_sq,gradients_sq) ]
    learning_rate_sq = [ (d_sq + eps) / (g_sq + eps) for d_sq,g_sq in izip(acc_deltas_sq,gradients_sq_new) ]

    deltas_sq = [ lr_sq * g_sq for lr_sq,g_sq in izip(learning_rate_sq,gradients_sq) ]
    deltas_sq_new = [ rho * acc_d_sq + (np.float32(1.) - rho) * d_sq for acc_d_sq,d_sq in izip(acc_deltas_sq,deltas_sq) ]

    deltas = [ T.sqrt(lr_sq) * g for lr_sq,g in izip(learning_rate_sq,gradients) ]

    gradient_sq_updates = zip(acc_gradients_sq,gradients_sq_new)
    deltas_sq_updates   = zip(acc_deltas_sq,deltas_sq_new)
    return deltas, gradient_sq_updates + deltas_sq_updates

@track_parameters
def adagrad(parameters,gradients,learning_rate=1e-4,P=None):
    shapes = get_shapes(parameters)

    grad_sq = [ create_param(P,"acc_sq_" + p.name,np.zeros(s)) for p,s in izip(parameters,shapes) ]

    grad_sq_new = [ g_sq + g**2        for g,g_sq in izip(gradients,grad_sq) ]
    deltas = [ learning_rate * g / T.sqrt(g_sq + 1e-6) for g,g_sq in izip(gradients,grad_sq_new) ]
    grad_sq_update = zip(grad_sq,grad_sq_new)

    return deltas,grad_sq_update



@track_parameters
def momentum(parameters,gradients,mu=0.9,learning_rate=1e-3,P=None):
    eps = learning_rate
    P.t = 1
    m = (1 - 3.0/(P.t+5) < mu)
    mu = m * (1 - 3.0/(P.t+5)) + (1-m) * mu
    shapes = get_shapes(parameters) 
    deltas = [ create_param(P,"deltas_" + p.name,np.zeros(s)) for p,s in izip(parameters,shapes) ]
    delta_nexts = [ mu*delta + eps*grad for delta,grad in zip(deltas,gradients) ]
    delta_updates = [ (delta, delta_next) for delta,delta_next in zip(deltas,delta_nexts) ]
    return delta_nexts, delta_updates  + [ (P.t,P.t + 1) ]

@track_parameters
def rmsprop(parameters,gradients,discount=0.95,momentum=0.9,learning_rate=1e-4,epsilon=1e-4,P=None):
    shapes = get_shapes(parameters)
    sq_acc    = [ create_param(P,"sq_acc_" + p.name,np.zeros(s))    for p,s in izip(parameters,shapes) ]
    acc       = [ create_param(P,"acc_" + p.name,np.zeros(s))       for p,s in izip(parameters,shapes) ]
    delta_acc = [ create_param(P,"delta_acc_" + p.name,np.zeros(s)) for p,s in izip(parameters,shapes) ]

    sq_avg = [ discount * sq_a + (1-discount)*(g**2) for sq_a,g in izip(sq_acc,gradients) ]
    avg    = [ discount * a    + (1-discount)*g      for a,   g in izip(acc,gradients) ]
    scaled_grads = [ g / T.sqrt(sq_a - a**2 + epsilon) for g,a,sq_a in izip(gradients,acc,sq_acc) ]
    deltas = [ momentum * d_a + learning_rate * s_g for d_a,s_g in izip(delta_acc,scaled_grads) ]


    sq_acc_updates = [ (sq_a, sq_aa) for sq_a,sq_aa in izip(sq_acc,sq_avg) ]
    acc_updates    = [ (a,    aa)    for a,   aa    in izip(acc,avg) ]
    delta_updates  = [ (d_a,d) for d_a,d in izip(delta_acc,deltas) ]
    parameters_updates = [ (p, p - d) for p,d in izip(parameters,deltas) ]

    return deltas, acc_updates + sq_acc_updates + delta_updates

