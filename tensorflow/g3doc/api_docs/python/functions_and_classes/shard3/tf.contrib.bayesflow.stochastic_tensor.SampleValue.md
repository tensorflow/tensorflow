Draw samples, possibly adding new outer dimensions along the way.

This ValueType draws samples from StochasticTensors run within its
context, increasing the rank according to the requested shape.

Examples:

```python
mu = tf.zeros((2,3))
sigma = tf.ones((2, 3))
with sg.value_type(sg.SampleValue()):
  st = sg.StochasticTensor(
    distributions.Normal, mu=mu, sigma=sigma)
# draws 1 sample and does not reshape
assertEqual(st.value().get_shape(), (2, 3))
```

```python
mu = tf.zeros((2,3))
sigma = tf.ones((2, 3))
with sg.value_type(sg.SampleValue(4)):
  st = sg.StochasticTensor(
    distributions.Normal, mu=mu, sigma=sigma)
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




