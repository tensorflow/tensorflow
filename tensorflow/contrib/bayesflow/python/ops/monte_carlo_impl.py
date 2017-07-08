# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Monte Carlo integration and helpers.

See the @{$python/contrib.bayesflow.monte_carlo} guide.

@@expectation
@@expectation_importance_sampler
@@expectation_importance_sampler_logspace
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

__all__ = [
    'expectation',
    'expectation_v2',
    'expectation_importance_sampler',
    'expectation_importance_sampler_logspace',
]


def expectation_importance_sampler(f,
                                   log_p,
                                   sampling_dist_q,
                                   z=None,
                                   n=None,
                                   seed=None,
                                   name='expectation_importance_sampler'):
  r"""Monte Carlo estimate of `E_p[f(Z)] = E_q[f(Z) p(Z) / q(Z)]`.

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

  Args:
    f: Callable mapping samples from `sampling_dist_q` to `Tensors` with shape
      broadcastable to `q.batch_shape`.
      For example, `f` works "just like" `q.log_prob`.
    log_p:  Callable mapping samples from `sampling_dist_q` to `Tensors` with
      shape broadcastable to `q.batch_shape`.
      For example, `log_p` works "just like" `sampling_dist_q.log_prob`.
    sampling_dist_q:  The sampling distribution.
      `tf.contrib.distributions.Distribution`.
      `float64` `dtype` recommended.
      `log_p` and `q` should be supported on the same set.
    z:  `Tensor` of samples from `q`, produced by `q.sample` for some `n`.
    n:  Integer `Tensor`.  Number of samples to generate if `z` is not provided.
    seed:  Python integer to seed the random number generator.
    name:  A name to give this `Op`.

  Returns:
    The importance sampling estimate.  `Tensor` with `shape` equal
      to batch shape of `q`, and `dtype` = `q.dtype`.
  """
  q = sampling_dist_q
  with ops.name_scope(name, values=[z, n]):
    z = _get_samples(q, z, n, seed)

    log_p_z = log_p(z)
    q_log_prob_z = q.log_prob(z)

    def _importance_sampler_positive_f(log_f_z):
      # Same as expectation_importance_sampler_logspace, but using Tensors
      # rather than samples and functions.  Allows us to sample once.
      log_values = log_f_z + log_p_z - q_log_prob_z
      return _logspace_mean(log_values)

    # With f_plus(z) = max(0, f(z)), f_minus(z) = max(0, -f(z)),
    # E_p[f(Z)] = E_p[f_plus(Z)] - E_p[f_minus(Z)]
    #           = E_p[f_plus(Z) + 1] - E_p[f_minus(Z) + 1]
    # Without incurring bias, 1 is added to each to prevent zeros in logspace.
    # The logarithm is approximately linear around 1 + epsilon, so this is good
    # for small values of 'z' as well.
    f_z = f(z)
    log_f_plus_z = math_ops.log(nn.relu(f_z) + 1.)
    log_f_minus_z = math_ops.log(nn.relu(-1. * f_z) + 1.)

    log_f_plus_integral = _importance_sampler_positive_f(log_f_plus_z)
    log_f_minus_integral = _importance_sampler_positive_f(log_f_minus_z)

  return math_ops.exp(log_f_plus_integral) - math_ops.exp(log_f_minus_integral)


def expectation_importance_sampler_logspace(
    log_f,
    log_p,
    sampling_dist_q,
    z=None,
    n=None,
    seed=None,
    name='expectation_importance_sampler_logspace'):
  r"""Importance sampling with a positive function, in log-space.

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

  Args:
    log_f: Callable mapping samples from `sampling_dist_q` to `Tensors` with
      shape broadcastable to `q.batch_shape`.
      For example, `log_f` works "just like" `sampling_dist_q.log_prob`.
    log_p:  Callable mapping samples from `sampling_dist_q` to `Tensors` with
      shape broadcastable to `q.batch_shape`.
      For example, `log_p` works "just like" `q.log_prob`.
    sampling_dist_q:  The sampling distribution.
      `tf.contrib.distributions.Distribution`.
      `float64` `dtype` recommended.
      `log_p` and `q` should be supported on the same set.
    z:  `Tensor` of samples from `q`, produced by `q.sample` for some `n`.
    n:  Integer `Tensor`.  Number of samples to generate if `z` is not provided.
    seed:  Python integer to seed the random number generator.
    name:  A name to give this `Op`.

  Returns:
    Logarithm of the importance sampling estimate.  `Tensor` with `shape` equal
      to batch shape of `q`, and `dtype` = `q.dtype`.
  """
  q = sampling_dist_q
  with ops.name_scope(name, values=[z, n]):
    z = _get_samples(q, z, n, seed)
    log_values = log_f(z) + log_p(z) - q.log_prob(z)
    return _logspace_mean(log_values)


def _logspace_mean(log_values):
  """Evaluate `Log[E[values]]` in a stable manner.

  Args:
    log_values:  `Tensor` holding `Log[values]`.

  Returns:
    `Tensor` of same `dtype` as `log_values`, reduced across dim 0.
      `Log[Mean[values]]`.
  """
  # center = Max[Log[values]],  with stop-gradient
  # The center hopefully keep the exponentiated term small.  It is canceled
  # from the final result, so putting stop gradient on it will not change the
  # final result.  We put stop gradient on to eliminate unnecessary computation.
  center = array_ops.stop_gradient(_sample_max(log_values))

  # centered_values = exp{Log[values] - E[Log[values]]}
  centered_values = math_ops.exp(log_values - center)

  # log_mean_of_values = Log[ E[centered_values] ] + center
  #                    = Log[ E[exp{log_values - E[log_values]}] ] + center
  #                    = Log[E[values]] - E[log_values] + center
  #                    = Log[E[values]]
  log_mean_of_values = math_ops.log(_sample_mean(centered_values)) + center

  return log_mean_of_values


def expectation(f, p, z=None, n=None, seed=None, name='expectation'):
  r"""Monte Carlo estimate of an expectation:  `E_p[f(Z)]` with sample mean.

  This `Op` returns

  ```
  n^{-1} sum_{i=1}^n f(z_i),  where z_i ~ p
  \approx E_p[f(Z)]
  ```

  User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

  Args:
    f: Callable mapping samples from `p` to `Tensors`.
    p:  `tf.contrib.distributions.Distribution`.
    z:  `Tensor` of samples from `p`, produced by `p.sample` for some `n`.
    n:  Integer `Tensor`.  Number of samples to generate if `z` is not provided.
    seed:  Python integer to seed the random number generator.
    name:  A name to give this `Op`.

  Returns:
    A `Tensor` with the same `dtype` as `p`.

  Example:

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

  """
  with ops.name_scope(name, values=[n, z]):
    z = _get_samples(p, z, n, seed)
    return _sample_mean(f(z))


def expectation_v2(f, samples, log_prob=None, use_reparametrization=True,
                   axis=0, keep_dims=False, name=None):
  """Computes the Monte-Carlo approximation of `E_p[f(X)]`.

  This function computes the Monte-Carlo approximation of an expectation, i.e.,

  ```none
  E_p[f(X)] approx= m**-1 sum_i^m f(x_j),  x_j ~iid p(X)
  ```

  where:

  - `x_j = samples[j, ...]`,
  - `log(p(samples)) = log_prob(samples)` and
  - `m = prod(shape(samples)[axis])`.

  Tricks: Reparameterization and Score-Gradient

  When p is "reparameterized", i.e., a diffeomorphic transformation of a
  parameterless distribution (e.g.,
  `Normal(Y; m, s) <=> Y = sX + m, X ~ Normal(0,1)`), we can swap gradient and
  expectation, i.e.,
  `grad[ Avg{ s_i : i=1...n } ] = Avg{ grad[s_i] : i=1...n }` where
  `S_n = Avg{s_i}` and `s_i = f(x_i), x_i ~ p`.

  However, if p is not reparameterized, TensorFlow's gradient will be incorrect
  since the chain-rule stops at samples of unreparameterized distributions. In
  this circumstance using the Score-Gradient trick results in an unbiased
  gradient, i.e.,

  ```none
  grad[ E_p[f(X)] ]
  = grad[ int dx p(x) f(x) ]
  = int dx grad[ p(x) f(x) ]
  = int dx [ p'(x) f(x) + p(x) f'(x) ]
  = int dx p(x) [p'(x) / p(x) f(x) + f'(x) ]
  = int dx p(x) grad[ f(x) p(x) / stop_grad[p(x)] ]
  = E_p[ grad[ f(x) p(x) / stop_grad[p(x)] ] ]
  ```

  Unless p is not reparametrized, it is usually preferable to
  `use_reparametrization = True`.

  Warning: users are responsible for verifying `p` is a "reparameterized"
  distribution.

  Args:
    f: Python callable which can return `f(samples)`.
    samples: `Tensor` of samples used to form the Monte-Carlo approximation of
      `E_p[f(X)]`.  A batch of samples should be indexed by `axis` dimensions.
    log_prob: Python callable which can return `log_prob(samples)`. Must
      correspond to the natural-logarithm of the pdf/pmf of each sample. Only
      required/used if `use_reparametrization=False`.
    use_reparametrization: Python `bool` indicating that the approximation
      should use the fact that the gradient of samples is unbiased.
    axis: The dimensions to average. If `None` (the default), averages all
      dimensions.
    keep_dims: If true, retains averaged dimensions with length 1.
    name: A `name_scope` for operations created by this function (optional).
      Default value: "expectation_v2".

  Returns:
    approx_expectation: `Tensor` corresponding to the Monte-Carlo approximation
      of `E_p[f(X)]`.

  Raises:
    ValueError: if `f` is not `callable`.
    ValueError: if `use_reparametrization=False` and `log_prob` is not
      `callable`.
  """

  with ops.name_scope(name, 'expectation_v2', [samples]):
    if not callable(f):
      raise ValueError('`f` must be a callable function.')
    if use_reparametrization:
      return math_ops.reduce_mean(f(samples), axis=axis, keep_dims=keep_dims)
    else:
      if not callable(log_prob):
        raise ValueError('`log_prob` must be a callable function.')
      x = array_ops.stop_gradient(samples)
      logpx = log_prob(x)
      # Numerically, exp(g(x) - stop[g(x)]) is always 1, even if exp(g(x)) is
      # unstable. But the gradient is also stable, ie,
      # d/dx exp(g(x) - stop[g(x)])
      # = exp(g(x) - stop[g(x)]) d/dx g(x)
      # = d/dx g(x)   [numerically exact since IEEE754 has the property
      #                that for any finite floating-point number x:
      #                x - x == 0.0]
      return math_ops.reduce_mean(
          f(x) * math_ops.exp(logpx - array_ops.stop_gradient(logpx)),
          axis=axis,
          keep_dims=keep_dims)


def _sample_mean(values):
  """Mean over sample indices.  In this module this is always [0]."""
  return math_ops.reduce_mean(values, reduction_indices=[0])


def _sample_max(values):
  """Max over sample indices.  In this module this is always [0]."""
  return math_ops.reduce_max(values, reduction_indices=[0])


def _get_samples(dist, z, n, seed):
  """Check args and return samples."""
  with ops.name_scope('get_samples', values=[z, n]):
    if (n is None) == (z is None):
      raise ValueError(
          'Must specify exactly one of arguments "n" and "z".  Found: '
          'n = %s, z = %s' % (n, z))
    if n is not None:
      return dist.sample(n, seed=seed)
    else:
      return ops.convert_to_tensor(z, name='z')
