<!-- This file is machine generated: DO NOT EDIT! -->

# BayesFlow Stochastic Tensors (contrib)
[TOC]

Classes and helper functions for creating Stochastic Tensors.

`StochasticTensor` objects wrap `Distribution` objects.  Their
values may be samples from the underlying distribution, or the distribution
mean (as governed by `value_type`).  These objects provide a `loss`
method for use when sampling from a non-reparameterized distribution.
The `loss`method is used in conjunction with `stochastic_graph.surrogate_loss`
to produce a single differentiable loss in stochastic graphs having
both continuous and discrete stochastic nodes.

## Stochastic Tensor Classes

- - -

### `class tf.contrib.bayesflow.stochastic_tensor.BaseStochasticTensor` {#BaseStochasticTensor}

Base Class for Tensor-like objects that emit stochastic values.
- - -

#### `tf.contrib.bayesflow.stochastic_tensor.BaseStochasticTensor.__init__()` {#BaseStochasticTensor.__init__}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.BaseStochasticTensor.dtype` {#BaseStochasticTensor.dtype}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.BaseStochasticTensor.graph` {#BaseStochasticTensor.graph}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.BaseStochasticTensor.loss(sample_loss)` {#BaseStochasticTensor.loss}

Returns the term to add to the surrogate loss.

This method is called by `surrogate_loss`.  The input `sample_loss` should
have already had `stop_gradient` applied to it.  This is because the
surrogate_loss usually provides a Monte Carlo sample term of the form
`differentiable_surrogate * sample_loss` where `sample_loss` is considered
constant with respect to the input for purposes of the gradient.

##### Args:


*  <b>`sample_loss`</b>: `Tensor`, sample loss downstream of this `StochasticTensor`.

##### Returns:

  Either `None` or a `Tensor`.


- - -

#### `tf.contrib.bayesflow.stochastic_tensor.BaseStochasticTensor.name` {#BaseStochasticTensor.name}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.BaseStochasticTensor.value(name=None)` {#BaseStochasticTensor.value}





- - -

### `class tf.contrib.bayesflow.stochastic_tensor.StochasticTensor` {#StochasticTensor}

StochasticTensor is a BaseStochasticTensor backed by a distribution.
- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.__init__(dist, name='StochasticTensor', dist_value_type=None, loss_fn=score_function)` {#StochasticTensor.__init__}

Construct a `StochasticTensor`.

`StochasticTensor` is backed by the `dist` distribution and its `value`
method will return the same value each time it is called. What `value` is
returned is controlled by the `dist_value_type` (defaults to
`SampleValue`).

Some distributions' sample functions are not differentiable (e.g. a sample
from a discrete distribution like a Bernoulli) and so to differentiate
wrt parameters upstream of the sample requires a gradient estimator like
the score function estimator. This is accomplished by passing a
differentiable `loss_fn` to the `StochasticTensor`, which
defaults to a function whose derivative is the score function estimator.
Calling `stochastic_graph.surrogate_loss(final_losses)` will call
`loss()` on every `StochasticTensor` upstream of final losses.

`loss()` will return None for `StochasticTensor`s backed by
reparameterized distributions; it will also return None if the value type is
`MeanValueType` or if `loss_fn=None`.

##### Args:


*  <b>`dist`</b>: an instance of `Distribution`.
*  <b>`name`</b>: a name for this `StochasticTensor` and its ops.
*  <b>`dist_value_type`</b>: a `_StochasticValueType`, which will determine what the
      `value` of this `StochasticTensor` will be. If not provided, the
      value type set with the `value_type` context manager will be used.
*  <b>`loss_fn`</b>: callable that takes
      `(st, st.value(), influenced_loss)`, where
      `st` is this `StochasticTensor`, and returns a `Tensor` loss. By
      default, `loss_fn` is the `score_function`, or more precisely, the
      integral of the score function, such that when the gradient is taken,
      the score function results. See the `stochastic_gradient_estimators`
      module for additional loss functions and baselines.

##### Raises:


*  <b>`TypeError`</b>: if `dist` is not an instance of `Distribution`.
*  <b>`TypeError`</b>: if `loss_fn` is not `callable`.


- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.distribution` {#StochasticTensor.distribution}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.dtype` {#StochasticTensor.dtype}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.entropy(name='entropy')` {#StochasticTensor.entropy}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.graph` {#StochasticTensor.graph}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.loss(final_loss, name='Loss')` {#StochasticTensor.loss}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.mean(name='mean')` {#StochasticTensor.mean}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.name` {#StochasticTensor.name}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.value(name='value')` {#StochasticTensor.value}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.StochasticTensor.value_type` {#StochasticTensor.value_type}






## Stochastic Tensor Value Types

- - -

### `class tf.contrib.bayesflow.stochastic_tensor.MeanValue` {#MeanValue}


- - -

#### `tf.contrib.bayesflow.stochastic_tensor.MeanValue.__init__(stop_gradient=False)` {#MeanValue.__init__}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.MeanValue.declare_inputs(unused_stochastic_tensor, unused_inputs_dict)` {#MeanValue.declare_inputs}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.MeanValue.popped_above(unused_value_type)` {#MeanValue.popped_above}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.MeanValue.pushed_above(unused_value_type)` {#MeanValue.pushed_above}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.MeanValue.stop_gradient` {#MeanValue.stop_gradient}





- - -

### `class tf.contrib.bayesflow.stochastic_tensor.SampleValue` {#SampleValue}

Draw samples, possibly adding new outer dimensions along the way.

This ValueType draws samples from StochasticTensors run within its
context, increasing the rank according to the requested shape.

Examples:

```python
mu = tf.zeros((2,3))
sigma = tf.ones((2, 3))
with sg.value_type(sg.SampleValue()):
  st = sg.StochasticTensor(
    tf.contrib.distributions.Normal, mu=mu, sigma=sigma)
# draws 1 sample and does not reshape
assertEqual(st.value().get_shape(), (2, 3))
```

```python
mu = tf.zeros((2,3))
sigma = tf.ones((2, 3))
with sg.value_type(sg.SampleValue(4)):
  st = sg.StochasticTensor(
    tf.contrib.distributions.Normal, mu=mu, sigma=sigma)
# draws 4 samples each with shape (2, 3) and concatenates
assertEqual(st.value().get_shape(), (4, 2, 3))
```
- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.__init__(shape=(), stop_gradient=False)` {#SampleValue.__init__}

Sample according to shape.

For the given StochasticTensor `st` using this value type,
the shape of `st.value()` will match that of
`st.distribution.sample(shape)`.

##### Args:


*  <b>`shape`</b>: A shape tuple or int32 tensor.  The sample shape.
    Default is a scalar: take one sample and do not change the size.
*  <b>`stop_gradient`</b>: If `True`, StochasticTensors' values are wrapped in
    `stop_gradient`, to avoid backpropagation through.


- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.declare_inputs(unused_stochastic_tensor, unused_inputs_dict)` {#SampleValue.declare_inputs}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.popped_above(unused_value_type)` {#SampleValue.popped_above}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.pushed_above(unused_value_type)` {#SampleValue.pushed_above}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.shape` {#SampleValue.shape}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.stop_gradient` {#SampleValue.stop_gradient}






- - -

### `tf.contrib.bayesflow.stochastic_tensor.value_type(dist_value_type)` {#value_type}

Creates a value type context for any StochasticTensor created within.

Typical usage:

```
with sg.value_type(sg.MeanValue(stop_gradients=True)):
  st = sg.StochasticTensor(tf.contrib.distributions.Normal, mu=mu,
                           sigma=sigma)
```

In the example above, `st.value()` (or equivalently, `tf.identity(st)`) will
be the mean value of the Normal distribution, i.e., `mu` (possibly
broadcasted to the shape of `sigma`).  Furthermore, because the `MeanValue`
was marked with `stop_gradients=True`, this value will have been wrapped
in a `stop_gradients` call to disable any possible backpropagation.

##### Args:


*  <b>`dist_value_type`</b>: An instance of `MeanValue`, `SampleValue`, or
    any other stochastic value type.

##### Yields:

  A context for `StochasticTensor` objects that controls the
  value created when they are initialized.

##### Raises:


*  <b>`TypeError`</b>: if `dist_value_type` is not an instance of a stochastic value
    type.


- - -

### `tf.contrib.bayesflow.stochastic_tensor.get_current_value_type()` {#get_current_value_type}





## Other Functions and Classes
- - -

### `class tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor` {#ObservedStochasticTensor}

A StochasticTensor with an observed value.
- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.__init__(dist, value, name=None)` {#ObservedStochasticTensor.__init__}

Construct an `ObservedStochasticTensor`.

`ObservedStochasticTensor` is backed by distribution `dist` and uses the
provided value instead of using the current value type to draw a value from
the distribution. The provided value argument must be appropriately shaped
to have come from the distribution.

##### Args:


*  <b>`dist`</b>: an instance of `Distribution`.
*  <b>`value`</b>: a Tensor containing the observed value
*  <b>`name`</b>: a name for this `ObservedStochasticTensor` and its ops.

##### Raises:


*  <b>`TypeError`</b>: if `dist` is not an instance of `Distribution`.
*  <b>`ValueError`</b>: if `value` is not compatible with the distribution.


- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.distribution` {#ObservedStochasticTensor.distribution}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.dtype` {#ObservedStochasticTensor.dtype}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.entropy(name='entropy')` {#ObservedStochasticTensor.entropy}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.graph` {#ObservedStochasticTensor.graph}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.loss(final_loss, name=None)` {#ObservedStochasticTensor.loss}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.mean(name='mean')` {#ObservedStochasticTensor.mean}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.name` {#ObservedStochasticTensor.name}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.value(name='value')` {#ObservedStochasticTensor.value}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.value_type` {#ObservedStochasticTensor.value_type}





