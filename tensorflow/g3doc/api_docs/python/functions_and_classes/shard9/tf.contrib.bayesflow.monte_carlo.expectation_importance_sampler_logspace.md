### `tf.contrib.bayesflow.monte_carlo.expectation_importance_sampler_logspace(log_f, log_p, sampling_dist_q, z=None, n=None, seed=None, name='expectation_importance_sampler_logspace')` {#expectation_importance_sampler_logspace}

Importance sampling with a positive function, in log-space.

With `p(z) := exp{log_p(z)}`, and `f(z) = exp{log_f(z)}`, this `Op`
returns

```
Log[ n^{-1} sum_{i=1}^n [ f(z_i) p(z_i) / q(z_i) ] ],  z_i ~ q,
\approx Log[ E_q[ f(Z) p(Z) / q(Z) ] ]
=       Log[E_p[f(Z)]]
```

This integral is done in log-space with max-subtraction to better handle the
often extreme values that `f(z) p(z) / q(z)` can take on.

In contrast to `expectation_importance_sampler`, this `Op` returns values in
log-space.


User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

##### Args:


*  <b>`log_f`</b>: Callable mapping samples from `sampling_dist_q` to `Tensors` with
    shape broadcastable to `q.batch_shape`.
    For example, `log_f` works "just like" `sampling_dist_q.log_prob`.
*  <b>`log_p`</b>: Callable mapping samples from `sampling_dist_q` to `Tensors` with
    shape broadcastable to `q.batch_shape`.
    For example, `log_p` works "just like" `q.log_prob`.
*  <b>`sampling_dist_q`</b>: The sampling distribution.
    `tf.contrib.distributions.BaseDistribution`.
    `float64` `dtype` recommended.
    `log_p` and `q` should be supported on the same set.
*  <b>`z`</b>: `Tensor` of samples from `q`, produced by `q.sample_n`.
*  <b>`n`</b>: Integer `Tensor`.  Number of samples to generate if `z` is not provided.
*  <b>`seed`</b>: Python integer to seed the random number generator.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  Logarithm of the importance sampling estimate.  `Tensor` with `shape` equal
    to batch shape of `q`, and `dtype` = `q.dtype`.

