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
from tensorflow.python.util import deprecation

__all__ = [
    'expectation',
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
  r"""Monte Carlo estimate of \\(E_p[f(Z)] = E_q[f(Z) p(Z) / q(Z)]\\).

  With \\(p(z) := exp^{log_p(z)}\\), this `Op` returns

  \\(n^{-1} sum_{i=1}^n [ f(z_i) p(z_i) / q(z_i) ],  z_i ~ q,\\)
  \\(\approx E_q[ f(Z) p(Z) / q(Z) ]\\)
  \\(=       E_p[f(Z)]\\)

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
      `tfp.distributions.Distribution`.
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

    # With \\(f_{plus}(z) = max(0, f(z)), f_{minus}(z) = max(0, -f(z))\\),
    # \\(E_p[f(Z)] = E_p[f_{plus}(Z)] - E_p[f_{minus}(Z)]\\)
    # \\(          = E_p[f_{plus}(Z) + 1] - E_p[f_{minus}(Z) + 1]\\)
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

  With \\(p(z) := exp^{log_p(z)}\\), and \\(f(z) = exp{log_f(z)}\\),
  this `Op` returns

  \\(Log[ n^{-1} sum_{i=1}^n [ f(z_i) p(z_i) / q(z_i) ] ],  z_i ~ q,\\)
  \\(\approx Log[ E_q[ f(Z) p(Z) / q(Z) ] ]\\)
  \\(=       Log[E_p[f(Z)]]\\)

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
      `tfp.distributions.Distribution`.
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


@deprecation.deprecated(
    '2018-10-01',
    'The tf.contrib.bayesflow library has moved to '
    'TensorFlow Probability (https://github.com/tensorflow/probability). '
    'Use `tfp.monte_carlo.expectation` instead.',
    warn_once=True)
def expectation(f, samples, log_prob=None, use_reparametrization=True,
                axis=0, keep_dims=False, name=None):
  r"""Computes the Monte-Carlo approximation of \\(E_p[f(X)]\\).

  This function computes the Monte-Carlo approximation of an expectation, i.e.,

  \\(E_p[f(X)] \approx= m^{-1} sum_i^m f(x_j),  x_j\  ~iid\ p(X)\\)

  where:

  - `x_j = samples[j, ...]`,
  - `log(p(samples)) = log_prob(samples)` and
  - `m = prod(shape(samples)[axis])`.

  Tricks: Reparameterization and Score-Gradient

  When p is "reparameterized", i.e., a diffeomorphic transformation of a
  parameterless distribution (e.g.,
  `Normal(Y; m, s) <=> Y = sX + m, X ~ Normal(0,1)`), we can swap gradient and
  expectation, i.e.,
  grad[ Avg{ \\(s_i : i=1...n\\) } ] = Avg{ grad[\\(s_i\\)] : i=1...n } where
  S_n = Avg{\\(s_i\\)}` and `\\(s_i = f(x_i), x_i ~ p\\).

  However, if p is not reparameterized, TensorFlow's gradient will be incorrect
  since the chain-rule stops at samples of non-reparameterized distributions.
  (The non-differentiated result, `approx_expectation`, is the same regardless
  of `use_reparametrization`.) In this circumstance using the Score-Gradient
  trick results in an unbiased gradient, i.e.,

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

  Example Use:

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Monte-Carlo approximation of a reparameterized distribution, e.g., Normal.

  num_draws = int(1e5)
  p = tfd.Normal(loc=0., scale=1.)
  q = tfd.Normal(loc=1., scale=2.)
  exact_kl_normal_normal = tfd.kl_divergence(p, q)
  # ==> 0.44314718
  approx_kl_normal_normal = tfp.monte_carlo.expectation(
      f=lambda x: p.log_prob(x) - q.log_prob(x),
      samples=p.sample(num_draws, seed=42),
      log_prob=p.log_prob,
      use_reparametrization=(p.reparameterization_type
                             == distribution.FULLY_REPARAMETERIZED))
  # ==> 0.44632751
  # Relative Error: <1%

  # Monte-Carlo approximation of non-reparameterized distribution, e.g., Gamma.

  num_draws = int(1e5)
  p = ds.Gamma(concentration=1., rate=1.)
  q = ds.Gamma(concentration=2., rate=3.)
  exact_kl_gamma_gamma = tfd.kl_divergence(p, q)
  # ==> 0.37999129
  approx_kl_gamma_gamma = tfp.monte_carlo.expectation(
      f=lambda x: p.log_prob(x) - q.log_prob(x),
      samples=p.sample(num_draws, seed=42),
      log_prob=p.log_prob,
      use_reparametrization=(p.reparameterization_type
                             == distribution.FULLY_REPARAMETERIZED))
  # ==> 0.37696719
  # Relative Error: <1%

  # For comparing the gradients, see `monte_carlo_test.py`.
  ```

  Note: The above example is for illustration only. To compute approximate
  KL-divergence, the following is preferred:

  ```python
  approx_kl_p_q = tfp.vi.monte_carlo_csiszar_f_divergence(
      f=bf.kl_reverse,
      p_log_prob=q.log_prob,
      q=p,
      num_draws=num_draws)
  ```

  Args:
    f: Python callable which can return `f(samples)`.
    samples: `Tensor` of samples used to form the Monte-Carlo approximation of
      \\(E_p[f(X)]\\).  A batch of samples should be indexed by `axis`
      dimensions.
    log_prob: Python callable which can return `log_prob(samples)`. Must
      correspond to the natural-logarithm of the pdf/pmf of each sample. Only
      required/used if `use_reparametrization=False`.
      Default value: `None`.
    use_reparametrization: Python `bool` indicating that the approximation
      should use the fact that the gradient of samples is unbiased. Whether
      `True` or `False`, this arg only affects the gradient of the resulting
      `approx_expectation`.
      Default value: `True`.
    axis: The dimensions to average. If `None`, averages all
      dimensions.
      Default value: `0` (the left-most dimension).
    keep_dims: If True, retains averaged dimensions using size `1`.
      Default value: `False`.
    name: A `name_scope` for operations created by this function.
      Default value: `None` (which implies "expectation").

  Returns:
    approx_expectation: `Tensor` corresponding to the Monte-Carlo approximation
      of \\(E_p[f(X)]\\).

  Raises:
    ValueError: if `f` is not a Python `callable`.
    ValueError: if `use_reparametrization=False` and `log_prob` is not a Python
      `callable`.
  """

  with ops.name_scope(name, 'expectation', [samples]):
    if not callable(f):
      raise ValueError('`f` must be a callable function.')
    if use_reparametrization:
      return math_ops.reduce_mean(f(samples), axis=axis, keepdims=keep_dims)
    else:
      if not callable(log_prob):
        raise ValueError('`log_prob` must be a callable function.')
      stop = array_ops.stop_gradient  # For readability.
      x = stop(samples)
      logpx = log_prob(x)
      fx = f(x)  # Call `f` once in case it has side-effects.
      # We now rewrite f(x) so that:
      #   `grad[f(x)] := grad[f(x)] + f(x) * grad[logqx]`.
      # To achieve this, we use a trick that
      #   `h(x) - stop(h(x)) == zeros_like(h(x))`
      # but its gradient is grad[h(x)].
      # Note that IEEE754 specifies that `x - x == 0.` and `x + 0. == x`, hence
      # this trick loses no precision. For more discussion regarding the
      # relevant portions of the IEEE754 standard, see the StackOverflow
      # question,
      # "Is there a floating point value of x, for which x-x == 0 is false?"
      # http://stackoverflow.com/q/2686644
      fx += stop(fx) * (logpx - stop(logpx))  # Add zeros_like(logpx).
      return math_ops.reduce_mean(fx, axis=axis, keepdims=keep_dims)


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
