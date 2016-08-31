Draw n samples along a new outer dimension.

This ValueType draws `n` samples from StochasticTensors run within its
context, increasing the rank by one along a new outer dimension.

Example:

```python
mu = tf.zeros((2,3))
sigma = tf.ones((2, 3))
with sg.value_type(sg.SampleValue(n=4)):
  dt = sg.DistributionTensor(
    distributions.Normal, mu=mu, sigma=sigma)
# draws 4 samples each with shape (2, 3) and concatenates
assertEqual(dt.value().get_shape(), (4, 2, 3))
```
- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.__init__(n=1, stop_gradient=False)` {#SampleValue.__init__}

Sample `n` times and concatenate along a new outer dimension.

##### Args:


*  <b>`n`</b>: A python integer or int32 tensor. The number of samples to take.
*  <b>`stop_gradient`</b>: If `True`, StochasticTensors' values are wrapped in
    `stop_gradient`, to avoid backpropagation through.


- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.declare_inputs(unused_stochastic_tensor, unused_inputs_dict)` {#SampleValue.declare_inputs}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.n` {#SampleValue.n}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.popped_above(unused_value_type)` {#SampleValue.popped_above}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.pushed_above(unused_value_type)` {#SampleValue.pushed_above}




- - -

#### `tf.contrib.bayesflow.stochastic_tensor.SampleValue.stop_gradient` {#SampleValue.stop_gradient}




