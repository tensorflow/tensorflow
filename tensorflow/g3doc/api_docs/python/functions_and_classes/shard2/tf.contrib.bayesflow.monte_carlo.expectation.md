### `tf.contrib.bayesflow.monte_carlo.expectation(f, p, z=None, n=None, seed=None, name='expectation')` {#expectation}

Monte Carlo estimate of an expectation:  `E_p[f(Z)]` with sample mean.

This `Op` returns

```
n^{-1} sum_{i=1}^n f(z_i),  where z_i ~ p
\approx E_p[f(Z)]
```

User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

##### Args:


*  <b>`f`</b>: Callable mapping samples from `p` to `Tensors`.
*  <b>`p`</b>: `tf.contrib.distributions.Distribution`.
*  <b>`z`</b>: `Tensor` of samples from `p`, produced by `p.sample` for some `n`.
*  <b>`n`</b>: Integer `Tensor`.  Number of samples to generate if `z` is not provided.
*  <b>`seed`</b>: Python integer to seed the random number generator.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with the same `dtype` as `p`.


*  <b>`Example`</b>: 

```python
N_samples = 10000

distributions = tf.contrib.distributions

dist = distributions.Uniform([0.0, 0.0], [1.0, 2.0])
elementwise_mean = lambda x: x
mean_sum = lambda x: tf.reduce_sum(x, 1)

estimate_elementwise_mean_tf = monte_carlo.expectation(elementwise_mean,
                                                       dist,
                                                       n=N_samples)
estimate_mean_sum_tf = monte_carlo.expectation(mean_sum,
                                               dist,
                                               n=N_samples)

with tf.Session() as sess:
  estimate_elementwise_mean, estimate_mean_sum = (
      sess.run([estimate_elementwise_mean_tf, estimate_mean_sum_tf]))
print estimate_elementwise_mean
>>> np.array([ 0.50018013  1.00097895], dtype=np.float32)
print estimate_mean_sum
>>> 1.49571

```

