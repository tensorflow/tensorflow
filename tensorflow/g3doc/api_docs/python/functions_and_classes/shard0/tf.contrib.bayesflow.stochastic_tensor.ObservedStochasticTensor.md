A StochasticTensor with an observed value.
- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.__init__(dist_cls, value, name=None, **dist_args)` {#ObservedStochasticTensor.__init__}

Construct an `ObservedStochasticTensor`.

`ObservedStochasticTensor` will instantiate a distribution from `dist_cls`
and `dist_args` but use the provided value instead of sampling from the
distribution. The provided value argument must be appropriately shaped
to have come from the constructed distribution.

##### Args:


*  <b>`dist_cls`</b>: a `Distribution` class.
*  <b>`value`</b>: a Tensor containing the observed value
*  <b>`name`</b>: a name for this `ObservedStochasticTensor` and its ops.
*  <b>`**dist_args`</b>: keyword arguments to be passed through to `dist_cls` on
      construction.

##### Raises:


*  <b>`TypeError`</b>: if `dist_cls` is not a `Distribution`.
*  <b>`ValueError`</b>: if `value` is not compatible with the distribution.


- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.clone(name=None, **dist_args)` {#ObservedStochasticTensor.clone}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.distribution` {#ObservedStochasticTensor.distribution}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.dtype` {#ObservedStochasticTensor.dtype}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.entropy(name='entropy')` {#ObservedStochasticTensor.entropy}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.graph` {#ObservedStochasticTensor.graph}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.ObservedStochasticTensor.input_dict` {#ObservedStochasticTensor.input_dict}




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




