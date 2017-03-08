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




