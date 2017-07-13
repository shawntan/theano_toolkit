from itertools import izip
import theano.tensor as T
import numpy as np
from parameters import Parameters


def clip_deltas(gradients, clip_size):
    grad_mag = T.sqrt(sum(T.sum(T.sqr(w)) for w in gradients))
    scale = clip_size / T.maximum(clip_size, grad_mag)
    return [scale * g for g in gradients]


def nan_shield(parameters, deltas, other_updates):
    delta_sum = sum(T.sum(d) for d in deltas)
    not_finite = T.isnan(delta_sum) | T.isinf(delta_sum)
    parameter_updates = [(p, T.switch(not_finite, 0.9 * p, p - d))
                         for p, d in izip(parameters, deltas)]
    other_updates = [(p, T.switch(not_finite, p, u))
                     for p, u in other_updates]
    return parameter_updates, other_updates


def track_parameters(update_fun):
    def decorated_fun(parameters, gradients, **kwargs):
        if "P" not in kwargs:
            kwargs["P"] = Parameters()
        deltas, updates = update_fun(parameters, gradients, **kwargs)
        assert(len(deltas) == len(parameters))
        parameter_updates, other_updates = nan_shield(parameters,
                                                      deltas, updates)
        return parameter_updates + other_updates
    return decorated_fun


def create_param(P, name, w):
    P[name] = w
    return P[name]


def get_shapes(parameters):
    return [p.get_value().shape for p in parameters]


@track_parameters
def adadelta(parameters, gradients,
             rho=np.float32(0.95),
             learning_rate=np.float32(1e-4),
             P=None):
    eps = learning_rate
    shapes = get_shapes(parameters)

    acc_gradients_sq = [create_param(P, "grad_sq_" + p.name, np.zeros(s))
                        for p, s in izip(parameters, shapes)]
    acc_deltas_sq = [create_param(P, "deltas_sq_" + p.name, np.zeros(s))
                     for p, s in izip(parameters, shapes)]

    gradients_sq = [T.sqr(g) for g in gradients]
    gradients_sq_new = [rho * acc_g_sq + (np.float32(1) - rho) * g_sq
                        for acc_g_sq, g_sq in izip(
                            acc_gradients_sq, gradients_sq)]
    learning_rate_sq = [(d_sq + eps) / (g_sq + eps)
                        for d_sq, g_sq in izip(
                            acc_deltas_sq, gradients_sq_new)]

    deltas_sq = [lr_sq * g_sq for lr_sq,
                 g_sq in izip(learning_rate_sq, gradients_sq)]
    deltas_sq_new = [rho * acc_d_sq + (np.float32(1.) - rho) *
                     d_sq for acc_d_sq, d_sq in izip(acc_deltas_sq, deltas_sq)]

    deltas = [T.sqrt(lr_sq) * g for lr_sq,
              g in izip(learning_rate_sq, gradients)]

    gradient_sq_updates = zip(acc_gradients_sq, gradients_sq_new)
    deltas_sq_updates = zip(acc_deltas_sq, deltas_sq_new)
    return deltas, gradient_sq_updates + deltas_sq_updates


@track_parameters
def adagrad(parameters, gradients, learning_rate=1e-4, P=None):
    shapes = get_shapes(parameters)

    grad_sq = [create_param(P, "acc_sq_" + p.name, np.zeros(s))
               for p, s in izip(parameters, shapes)]

    grad_sq_new = [g_sq + g**2 for g, g_sq in izip(gradients, grad_sq)]
    deltas = [learning_rate * g / T.sqrt(g_sq + 1e-6)
              for g, g_sq in izip(gradients, grad_sq_new)]
    grad_sq_update = zip(grad_sq, grad_sq_new)

    return deltas, grad_sq_update


@track_parameters
def momentum(parameters, gradients, mu=0.9, learning_rate=1e-3, P=None):
    eps = learning_rate
    P.t = 1
    m = (1 - 3.0 / (P.t + 5) < mu)
    mu = m * (1 - 3.0 / (P.t + 5)) + (1 - m) * mu
    shapes = get_shapes(parameters)
    deltas = [create_param(P, "deltas_" + p.name, np.zeros(s))
              for p, s in izip(parameters, shapes)]
    delta_nexts = [mu * delta + eps * grad for delta,
                   grad in zip(deltas, gradients)]
    delta_updates = [(delta, delta_next)
                     for delta, delta_next in zip(deltas, delta_nexts)]
    return delta_nexts, delta_updates + [(P.t, P.t + 1)]


@track_parameters
def rmsprop(parameters, gradients,
            discount=0.95,
            momentum=0.9,
            learning_rate=1e-4,
            epsilon=1e-4,
            P=None):
    shapes = get_shapes(parameters)
    sq_acc = [create_param(P, "sq_acc_" + p.name, np.zeros(s))
              for p, s in izip(parameters, shapes)]
    acc = [create_param(P, "acc_" + p.name, np.zeros(s))
           for p, s in izip(parameters, shapes)]
    delta_acc = [create_param(P, "delta_acc_" + p.name, np.zeros(s))
                 for p, s in izip(parameters, shapes)]

    sq_avg = [discount * sq_a + (1 - discount) * (g**2)
              for sq_a, g in izip(sq_acc, gradients)]
    avg = [discount * a + (1 - discount) * g for a,
           g in izip(acc, gradients)]
    scaled_grads = [g / T.sqrt(sq_a - a**2 + epsilon)
                    for g, a, sq_a in izip(gradients, acc, sq_acc)]
    deltas = [momentum * d_a + learning_rate *
              s_g for d_a, s_g in izip(delta_acc, scaled_grads)]

    sq_acc_updates = [(sq_a, sq_aa) for sq_a, sq_aa in izip(sq_acc, sq_avg)]
    acc_updates = [(a,    aa) for a,   aa in izip(acc, avg)]
    delta_updates = [(d_a, d) for d_a, d in izip(delta_acc, deltas)]

    return deltas, acc_updates + sq_acc_updates + delta_updates


@track_parameters
def adam(parameters, gradients,
         learning_rate=0.001,
         moment1_decay=0.9,
         moment2_decay=0.999,
         epsilon=1e-8,
         P=None):
    shapes = get_shapes(parameters)
    P.t = np.float32(1)

    moment1_acc = [create_param(P, "moment1_" + p.name, np.zeros(s))
                   for p, s in izip(parameters, shapes)]

    moment2_acc = [create_param(P, "moment2_" + p.name, np.zeros(s))
                   for p, s in izip(parameters, shapes)]

    deltas = []
    updates = []
    updates.append((P.t, P.t + 1))
    for m1, m2, g in izip(moment1_acc, moment2_acc, gradients):
        new_m1 = moment1_decay * m1 + (1 - moment1_decay) * g
        new_m2 = moment2_decay * m2 + (1 - moment2_decay) * T.sqr(g)
        bc_m1 = new_m1 / (1 - moment1_decay**P.t)
        bc_m2 = new_m2 / (1 - moment2_decay**P.t)
        delta = learning_rate * bc_m1 / (T.sqrt(bc_m2) + epsilon)

        deltas.append(delta)
        updates.append((m1, new_m1))
        updates.append((m2, new_m2))

    return deltas, updates
