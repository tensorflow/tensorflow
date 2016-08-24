### `tf.contrib.bayesflow.monte_carlo.expectation(f, p, z=None, n=None, seed=None, name='expectation')` {#expectation}

Monte Carlo estimate of an expectation:  `E_p[f(Z)]` with sample mean.

This `Op` returns

```
n^{-1} sum_{i=1}^n f(z_i),  where z_i ~ p
\approx E_p[f(Z)]
```

User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

##### Args:


*  <b>`f`</b>: Callable mapping samples from `sampling_dist_q` to `Tensors` with
    shape broadcastable to `q.batch_shape`.
    For example, `f` works "just like" `sampling_dist_q.log_prob`.
*  <b>`p`</b>: `tf.contrib.distributions.BaseDistribution`.
*  <b>`z`</b>: `Tensor` of samples from `p`, produced by `p.sample_n`.
*  <b>`n`</b>: Integer `Tensor`.  Number of samples to generate if `z` is not provided.
*  <b>`seed`</b>: Python integer to seed the random number generator.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with same `dtype` as `p`, and shape equal to `p.batch_shape`.

