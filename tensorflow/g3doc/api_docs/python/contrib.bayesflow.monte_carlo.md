<!-- This file is machine generated: DO NOT EDIT! -->

# BayesFlow Monte Carlo (contrib)
[TOC]

Monte Carlo integration and helpers.

## Background

Monte Carlo integration refers to the practice of estimating an expectation with
a sample mean.  For example, given random variable `Z in R^k` with density `p`,
the expectation of function `f` can be approximated like:

```
E_p[f(Z)] = \int f(z) p(z) dz
          ~ S_n
          := n^{-1} \sum_{i=1}^n f(z_i),  z_i iid samples from p.
```

If `E_p[|f(Z)|] < infinity`, then `S_n --> E_p[f(Z)]` by the strong law of large
numbers.  If `E_p[f(Z)^2] < infinity`, then `S_n` is asymptotically normal with
variance `Var[f(Z)] / n`.

Practitioners of Bayesian statistics often find themselves wanting to estimate
`E_p[f(Z)]` when the distribution `p` is known only up to a constant.  For
example, the joint distribution `p(z, x)` may be known, but the evidence
`p(x) = \int p(z, x) dz` may be intractable.  In that case, a parameterized
distribution family `q_lambda(z)` may be chosen, and the optimal `lambda` is the
one minimizing the KL divergence between `q_lambda(z)` and
`p(z | x)`.  We only know `p(z, x)`, but that is sufficient to find `lambda`.


## Log-space evaluation and subtracting the maximum.

Care must be taken when the random variable lives in a high dimensional space.
For example, the naive importance sample estimate `E_q[f(Z) p(Z) / q(Z)]`
involves the ratio of two terms `p(Z) / q(Z)`, each of which must have tails
dropping off faster than `O(|z|^{-(k + 1)})` in order to have finite integral.
This ratio would often be zero or infinity up to numerical precision.

For that reason, we write

```
Log E_q[ f(Z) p(Z) / q(Z) ]
   = Log E_q[ exp{Log[f(Z)] + Log[p(Z)] - Log[q(Z)] - C} ] + C,  where
C := Max[ Log[f(Z)] + Log[p(Z)] - Log[q(Z)] ].
```

The maximum value of the exponentiated term will be 0.0, and the expectation
can be evaluated in a stable manner.

## Ops

- - -

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
*  <b>`z`</b>: `Tensor` of samples from `p`, produced by `p.sample_n`.
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


- - -

### `tf.contrib.bayesflow.monte_carlo.expectation_importance_sampler(f, log_p, sampling_dist_q, z=None, n=None, seed=None, name='expectation_importance_sampler')` {#expectation_importance_sampler}

Monte Carlo estimate of `E_p[f(Z)] = E_q[f(Z) p(Z) / q(Z)]`.

With `p(z) := exp{log_p(z)}`, this `Op` returns

```
n^{-1} sum_{i=1}^n [ f(z_i) p(z_i) / q(z_i) ],  z_i ~ q,
\approx E_q[ f(Z) p(Z) / q(Z) ]
=       E_p[f(Z)]
```

This integral is done in log-space with max-subtraction to better handle the
often extreme values that `f(z) p(z) / q(z)` can take on.

If `f >= 0`, it is up to 2x more efficient to exponentiate the result of
`expectation_importance_sampler_logspace` applied to `Log[f]`.

User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

##### Args:


*  <b>`f`</b>: Callable mapping samples from `sampling_dist_q` to `Tensors` with shape
    broadcastable to `q.batch_shape`.
    For example, `f` works "just like" `q.log_prob`.
*  <b>`log_p`</b>: Callable mapping samples from `sampling_dist_q` to `Tensors` with
    shape broadcastable to `q.batch_shape`.
    For example, `log_p` works "just like" `sampling_dist_q.log_prob`.
*  <b>`sampling_dist_q`</b>: The sampling distribution.
    `tf.contrib.distributions.Distribution`.
    `float64` `dtype` recommended.
    `log_p` and `q` should be supported on the same set.
*  <b>`z`</b>: `Tensor` of samples from `q`, produced by `q.sample_n`.
*  <b>`n`</b>: Integer `Tensor`.  Number of samples to generate if `z` is not provided.
*  <b>`seed`</b>: Python integer to seed the random number generator.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  The importance sampling estimate.  `Tensor` with `shape` equal
    to batch shape of `q`, and `dtype` = `q.dtype`.


- - -

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
    `tf.contrib.distributions.Distribution`.
    `float64` `dtype` recommended.
    `log_p` and `q` should be supported on the same set.
*  <b>`z`</b>: `Tensor` of samples from `q`, produced by `q.sample_n`.
*  <b>`n`</b>: Integer `Tensor`.  Number of samples to generate if `z` is not provided.
*  <b>`seed`</b>: Python integer to seed the random number generator.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  Logarithm of the importance sampling estimate.  `Tensor` with `shape` equal
    to batch shape of `q`, and `dtype` = `q.dtype`.


