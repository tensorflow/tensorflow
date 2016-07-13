DistributionTensor is a StochasticTensor backed by a distribution.
- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.__init__(dist_cls, name=None, dist_value_type=None, surrogate_loss_fn=score_function, **dist_args)` {#DistributionTensor.__init__}

Construct a `DistributionTensor`.

`surrogate_loss_fn` controls what `surrogate_loss` returns, which is used
in conjunction with the `surrogate_losses` function in this module.
`surrogate_loss_fn` is a callable that takes this `DistributionTensor`, a
`Tensor` with this `DistributionTensor`'s value, and a list of `Tensor`
losses influenced by this `DistributionTensor`; it should return a `Tensor`
surrogate loss. If not provided, it defaults to the score function
surrogate loss: `log_prob(value) * sum(losses)`. If `surrogate_loss_fn` is
None, no surrogate loss will be returned. Currently, a surrogate loss will
only be used if `dist_value_type.stop_gradient=True` or if the value is a
sample from a non-reparameterized distribution.

##### Args:


*  <b>`dist_cls`</b>: a class deriving from `BaseDistribution`.
*  <b>`name`</b>: a name for this `DistributionTensor` and its ops.
*  <b>`dist_value_type`</b>: a `_StochasticValueType`, which will determine what the
      `value` of this `DistributionTensor` will be. If not provided, the
      value type set with the `value_type` context manager will be used.
*  <b>`surrogate_loss_fn`</b>: callable that takes
      `(dt, dt.value(), influenced_losses)`, where `dt` is this
      `DistributionTensor`, and returns a `Tensor` surrogate loss.
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

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.mean(name='mean')` {#DistributionTensor.mean}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.name` {#DistributionTensor.name}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.surrogate_loss(losses, name='DistributionSurrogateLoss')` {#DistributionTensor.surrogate_loss}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.DistributionTensor.value(name='value')` {#DistributionTensor.value}




