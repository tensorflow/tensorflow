Ask the StochasticTensor for n samples and reshape the result.

Sampling from a StochasticTensor increases the rank of the value by 1
(because each sample represents a new outer dimension).

This ValueType requests `n` samples from StochasticTensors run within its
context that the outer two dimensions are reshaped to intermix the samples
with the outermost (usually batch) dimension.

Example:

```python
# mu and sigma are both shaped (2, 3)
mu = [[0.0, -1.0, 1.0], [0.0, -1.0, 1.0]]
sigma = tf.constant([[1.1, 1.2, 1.3], [1.1, 1.2, 1.3]])

with sg.value_type(sg.SampleAndReshapeValue(n=2)):
  dt = sg.DistributionTensor(
      distributions.Normal, mu=mu, sigma=sigma)

# sample(2) creates a (2, 2, 3) tensor, and the two outermost dimensions
# are reshaped into one: the final value is a (4, 3) tensor.
dt_value = dt.value()
assertEqual(dt_value.get_shape(), (4, 3))

dt_value_val = sess.run([dt_value])[0]  # or e.g. run([tf.identity(dt)])[0]
assertEqual(dt_value_val.shape, (4, 3))
```
- - -

#### `tf.contrib.bayesflow.stochastic_graph.SampleAndReshapeValue.__init__(n=1, stop_gradient=False)` {#SampleAndReshapeValue.__init__}

Sample `n` times and reshape the outer 2 axes so rank does not change.

##### Args:


*  <b>`n`</b>: A python integer or int32 tensor.  The number of samples to take.
*  <b>`stop_gradient`</b>: If `True`, StochasticTensors' values are wrapped in
    `stop_gradient`, to avoid backpropagation through.


- - -

#### `tf.contrib.bayesflow.stochastic_graph.SampleAndReshapeValue.declare_inputs(unused_stochastic_tensor, unused_inputs_dict)` {#SampleAndReshapeValue.declare_inputs}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.SampleAndReshapeValue.n` {#SampleAndReshapeValue.n}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.SampleAndReshapeValue.popped_above(unused_value_type)` {#SampleAndReshapeValue.popped_above}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.SampleAndReshapeValue.pushed_above(unused_value_type)` {#SampleAndReshapeValue.pushed_above}




- - -

#### `tf.contrib.bayesflow.stochastic_graph.SampleAndReshapeValue.stop_gradient` {#SampleAndReshapeValue.stop_gradient}




