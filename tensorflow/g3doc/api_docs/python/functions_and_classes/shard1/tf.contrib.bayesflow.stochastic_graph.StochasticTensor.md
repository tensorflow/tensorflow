Base Class for Tensor-like objects that emit stochastic values.
- - -

#### `tf.contrib.bayesflow.stochastic_graph.StochasticTensor.__init__()` {#StochasticTensor.__init__}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.StochasticTensor.dtype` {#StochasticTensor.dtype}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.StochasticTensor.graph` {#StochasticTensor.graph}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.StochasticTensor.input_dict` {#StochasticTensor.input_dict}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.StochasticTensor.loss(sample_losses)` {#StochasticTensor.loss}

Returns the term to add to the surrogate loss.

This method is called by `surrogate_loss`.  The input `sample_losses`
should have already had `stop_gradient` applied to them.  This is
because the surrogate_loss usually provides a monte carlo sample term
of the form `differentiable_surrogate * sum(sample_losses)` where
`sample_losses` is considered constant with respect to the input
for purposes of the gradient.

##### Args:


*  <b>`sample_losses`</b>: a list of Tensors, the sample losses downstream of this
    `StochasticTensor`.

##### Returns:

  Either `None` or a `Tensor`.


- - -

#### `tf.contrib.bayesflow.stochastic_graph.StochasticTensor.name` {#StochasticTensor.name}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.StochasticTensor.value(name=None)` {#StochasticTensor.value}




