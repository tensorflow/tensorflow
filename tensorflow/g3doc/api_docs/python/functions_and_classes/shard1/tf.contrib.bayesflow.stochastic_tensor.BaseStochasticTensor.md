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




