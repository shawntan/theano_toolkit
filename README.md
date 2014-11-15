Helper methods for Theano
=========================

Some helper methods for working with Theano for neural networks.

## `hinton.py`
![Hinton Diagrams](https://blog.wtf.sg/wp-content/uploads/2014/05/Screenshot-from-2014-05-04-013804.png)
Quick visualisation method of numpy matrices in the terminal. See the [blog post](https://blog.wtf.sg/2014/05/04/terminal-hinton-diagrams/).

## `utils.py`

Miscellaneous helper functions for initialising weight matrices, vector softmax, etc.

## `parameters.py`

Syntax sugar when declaring parameters.

```python
import theano_toolkit.parameters as Parameters

P = Parameters()

P.W = np.zeros(10,10)
P.b = np.zeros(10)

# build model here.

params = P.values()
gradients = T.grad(cost,wrt=params)
```

#### Experimental

More syntax sugar for initialising parameters.
```python
P = Parameters()

with P:
	W = np.zeros(10,10)
	b = np.zeros(10)

# use W and b to define model instead of P.W and P.b

params = P.values()
:
```
