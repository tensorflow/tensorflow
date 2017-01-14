DistributionTensor is a StochasticTensor backed by a distribution.
- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.__init__(dist_cls, name=None, dist_value_type=None, loss_fn=score_function, **dist_args)` {#DistributionTensor.__init__}

Construct a `DistributionTensor`.

`DistributionTensor` will instantiate a distribution from `dist_cls` and
`dist_args` and its `value` method will return the same value each time
it is called. What `value` is returned is controlled by the
`dist_value_type` (defaults to `SampleAndReshapeValue`).

Some distributions' sample functions are not differentiable (e.g. a sample
from a discrete distribution like a Bernoulli) and so to differentiate
wrt parameters upstream of the sample requires a gradient estimator like
the score function estimator. This is accomplished by passing a
differentiable `loss_fn` to the `DistributionTensor`, which
defaults to a function whose derivative is the score function estimator.
Calling `stochastic_graph.surrogate_loss(final_losses)` will call
`loss()` on every `DistributionTensor` upstream of final losses.

`loss()` will return None for `DistributionTensor`s backed by
reparameterized distributions; it will also return None if the value type is
`MeanValueType` or if `loss_fn=None`.

##### Args:


*  <b>`dist_cls`</b>: a class deriving from `BaseDistribution`.
*  <b>`name`</b>: a name for this `DistributionTensor` and its ops.
*  <b>`dist_value_type`</b>: a `_StochasticValueType`, which will determine what the
      `value` of this `DistributionTensor` will be. If not provided, the
      value type set with the `value_type` context manager will be used.
*  <b>`loss_fn`</b>: callable that takes `(dt, dt.value(), influenced_losses)`, where
      `dt` is this `DistributionTensor`, and returns a `Tensor` loss.
*  <b>`**dist_args`</b>: keyword arguments to be passed through to `dist_cls` on
      construction.


- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.clone(name=None, **dist_args)` {#DistributionTensor.clone}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.distribution` {#DistributionTensor.distribution}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.dtype` {#DistributionTensor.dtype}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.entropy(name='entropy')` {#DistributionTensor.entropy}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.graph` {#DistributionTensor.graph}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.input_dict` {#DistributionTensor.input_dict}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.loss(final_losses, name='Loss')` {#DistributionTensor.loss}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.mean(name='mean')` {#DistributionTensor.mean}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.name` {#DistributionTensor.name}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.value(name='value')` {#DistributionTensor.value}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.value_type` {#DistributionTensor.value_type}




