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
r"""Entropy Ops.

## Background

Common Shannon entropy, the Evidence Lower BOund (ELBO), KL divergence, and more
all have information theoretic use and interpretations.  They are also often
used in variational inference.  This library brings together `Ops` for
estimating them, e.g. using Monte Carlo expectations.

## Examples

Example of fitting a variational posterior with the ELBO.

```python
# We start by assuming knowledge of the log of a joint density p(z, x) over
# latent variable z and fixed measurement x.  Since x is fixed, the Python
# function does not take x as an argument.
def log_joint(z):
  theta = tf.Variable(0.)  # Trainable variable that helps define log_joint.
  ...

# Next, define a Normal distribution with trainable parameters.
q = distributions.Normal(mu=tf.Variable(0.), sigma=tf.Variable(1.))

# Now, define a loss function (negative ELBO) that, when minimized, will adjust
# mu, sigma, and theta, increasing the ELBO, which we hope will both reduce the
# KL divergence between q(z) and p(z | x), and increase p(x).  Note that we
# cannot guarantee both, but in general we expect both to happen.
elbo = entropy.elbo_ratio(log_p, q, n=10)
loss = -elbo

# Minimize the loss
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
tf.initialize_all_variables().run()
for step in range(100):
  train_op.run()
```

## Ops

@@elbo_ratio
@@entropy_shannon
@@renyi_ratio
@@renyi_alpha

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib.bayesflow.python.ops import monte_carlo
from tensorflow.contrib.bayesflow.python.ops import variational_inference
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

# Make utility functions from monte_carlo available.
# pylint: disable=protected-access
_get_samples = monte_carlo._get_samples
_logspace_mean = monte_carlo._logspace_mean
_sample_mean = monte_carlo._sample_mean

# pylint: enable=protected-access

__all__ = [
    'elbo_ratio',
    'entropy_shannon',
    'renyi_ratio',
    'renyi_alpha',
]


ELBOForms = variational_inference.ELBOForms  # pylint: disable=invalid-name


def elbo_ratio(log_p,
               q,
               z=None,
               n=None,
               seed=None,
               form=None,
               name='elbo_ratio'):
  r"""Estimate of the ratio appearing in the `ELBO` and `KL` divergence.

  With `p(z) := exp{log_p(z)}`, this `Op` returns an approximation of

  ```
  E_q[ Log[p(Z) / q(Z)] ]
  ```

  The term `E_q[ Log[p(Z)] ]` is always computed as a sample mean.
  The term `E_q[ Log[q(z)] ]` can be computed with samples, or an exact formula
  if `q.entropy()` is defined.  This is controlled with the kwarg `form`.

  This log-ratio appears in different contexts:

  #### `KL[q || p]`

  If `log_p(z) = Log[p(z)]` for distribution `p`, this `Op` approximates
  the negative Kullback-Leibler divergence.

  ```
  elbo_ratio(log_p, q, n=100) = -1 * KL[q || p],
  KL[q || p] = E[ Log[q(Z)] - Log[p(Z)] ]
  ```

  Note that if `p` is a `Distribution`, then `distributions.kl(q, p)` may be
  defined and available as an exact result.

  #### ELBO

  If `log_p(z) = Log[p(z, x)]` is the log joint of a distribution `p`, this is
  the Evidence Lower BOund (ELBO):

  ```
  ELBO ~= E[ Log[p(Z, x)] - Log[q(Z)] ]
        = Log[p(x)] - KL[q || p]
       <= Log[p(x)]
  ```

  User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

  Args:
    log_p:  Callable mapping samples from `q` to `Tensors` with
      shape broadcastable to `q.batch_shape`.
      For example, `log_p` works "just like" `q.log_prob`.
    q:  `tf.contrib.distributions.BaseDistribution`.
    z:  `Tensor` of samples from `q`, produced by `q.sample_n`.
    n:  Integer `Tensor`.  Number of samples to generate if `z` is not provided.
    seed:  Python integer to seed the random number generator.
    form:  Either `ELBOForms.analytic_entropy` (use formula for entropy of `q`)
      or `ELBOForms.sample` (sample estimate of entropy), or `ELBOForms.default`
      (attempt analytic entropy, fallback on sample).
      Default value is `ELBOForms.default`.
    name:  A name to give this `Op`.

  Returns:
    Scalar `Tensor` holding sample mean KL divergence.  `shape` is the batch
      shape of `q`, and `dtype` is the same as `q`.

  Raises:
    ValueError:  If `form` is not handled by this function.
  """
  form = ELBOForms.default if form is None else form

  with ops.name_scope(name, values=[n, z]):
    z = _get_samples(q, z, n, seed)

    entropy = entropy_shannon(q, z=z, form=form)

    # If log_p(z) = Log[p(z)], cross entropy = -E_q[log(p(Z))]
    negative_cross_entropy = _sample_mean(log_p(z))

    return entropy + negative_cross_entropy


def entropy_shannon(p,
                    z=None,
                    n=None,
                    seed=None,
                    form=None,
                    name='entropy_shannon'):
  r"""Monte Carlo or deterministic computation of Shannon's entropy.

  Depending on the kwarg `form`, this `Op` returns either the analytic entropy
  of the distribution `p`, or the sampled entropy:

  ```
  -n^{-1} sum_{i=1}^n p.log_prob(z_i),  where z_i ~ p,
      \approx - E_p[ Log[p(Z)] ]
      = Entropy[p]
  ```

  User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

  Args:
    p:  `tf.contrib.distributions.BaseDistribution`
    z:  `Tensor` of samples from `p`, produced by `p.sample_n(n)` for some `n`.
    n:  Integer `Tensor`.  Number of samples to generate if `z` is not provided.
    seed:  Python integer to seed the random number generator.
    form:  Either `ELBOForms.analytic_entropy` (use formula for entropy of `q`)
      or `ELBOForms.sample` (sample estimate of entropy), or `ELBOForms.default`
      (attempt analytic entropy, fallback on sample).
      Default value is `ELBOForms.default`.
    name:  A name to give this `Op`.

  Returns:
    A `Tensor` with same `dtype` as `p`, and shape equal to `p.batch_shape`.

  Raises:
    ValueError:  If `form` not handled by this function.
    ValueError:  If `form` is `ELBOForms.analytic_entropy` and `n` was provided.
  """
  form = ELBOForms.default if form is None else form

  if n is not None and form == ELBOForms.analytic_entropy:
    raise ValueError('If form == ELBOForms.analytic_entropy, n must be None.')

  with ops.name_scope(name, values=[n, z]):
    # Entropy: -E_p[log(p(Z))].
    entropy = None

    # Try analytic path
    if form in [ELBOForms.default, ELBOForms.analytic_entropy]:
      try:
        entropy = p.entropy()
        logging.info('Using analytic entropy(p:%s)', p)
      except NotImplementedError as e:
        if form == ELBOForms.analytic_entropy:
          raise e
    elif form != ELBOForms.sample:
      raise ValueError('ELBOForm not handled by this function: %s' % form)

    # Sample path
    if entropy is None:
      logging.info('Using sampled entropy(p:%s)', p)
      entropy = -1. * monte_carlo.expectation(
          p.log_prob, p, z=z, n=n, seed=seed)

    return entropy


def renyi_ratio(log_p, q, alpha, z=None, n=None, seed=None, name='renyi_ratio'):
  r"""Monte Carlo estimate of the ratio appearing in Renyi divergence.

  This can be used to compute the Renyi (alpha) divergence, or a log evidence
  approximation based on Renyi divergence.

  #### Definition

  With `z_i` iid samples from `q`, and `exp{log_p(z)} = p(z)`, this `Op` returns
  the (biased for finite `n`) estimate:

  ```
  (1 - alpha)^{-1} Log[ n^{-1} sum_{i=1}^n ( p(z_i) / q(z_i) )^{1 - alpha},
  \approx (1 - alpha)^{-1} Log[ E_q[ (p(Z) / q(Z))^{1 - alpha} ]  ]
  ```

  This ratio appears in different contexts:

  #### Renyi divergence

  If `log_p(z) = Log[p(z)]` is the log prob of a distribution, and
  `alpha > 0`, `alpha != 1`, this `Op` approximates `-1` times Renyi divergence:

  ```
  # Choose reasonably high n to limit bias, see below.
  renyi_ratio(log_p, q, alpha, n=100)
                  \approx -1 * D_alpha[q || p],  where
  D_alpha[q || p] := (1 - alpha)^{-1} Log E_q[(p(Z) / q(Z))^{1 - alpha}]
  ```

  The Renyi (or "alpha") divergence is non-negative and equal to zero iff
  `q = p`.  Various limits of `alpha` lead to different special case results:

  ```
  alpha       D_alpha[q || p]
  -----       ---------------
  --> 0       Log[ int_{q > 0} p(z) dz ]
  = 0.5,      -2 Log[1 - Hel^2[q || p]],  (\propto squared Hellinger distance)
  --> 1       KL[q || p]
  = 2         Log[ 1 + chi^2[q || p] ],   (\propto squared Chi-2 divergence)
  --> infty   Log[ max_z{q(z) / p(z)} ],  (min description length principle).
  ```

  See "Renyi Divergence Variational Inference", by Li and Turner.

  #### Log evidence approximation

  If `log_p(z) = Log[p(z, x)]` is the log of the joint distribution `p`, this is
  an alternative to the ELBO common in variational inference.

  ```
  L_alpha(q, p) = Log[p(x)] - D_alpha[q || p]
  ```

  If `q` and `p` have the same support, and `0 < a <= b < 1`, one can show
  `ELBO <= D_b <= D_a <= Log[p(x)]`.  Thus, this `Op` allows a smooth
  interpolation between the ELBO and the true evidence.

  #### Stability notes

  Note that when `1 - alpha` is not small, the ratio `(p(z) / q(z))^{1 - alpha}`
  is subject to underflow/overflow issues.  For that reason, it is evaluated in
  log-space after centering.  Nonetheless, infinite/NaN results may occur.  For
  that reason, one may wish to shrink `alpha` gradually.  See the `Op`
  `renyi_alpha`.  Using `float64` will also help.


  #### Bias for finite sample size

  Due to nonlinearity of the logarithm, for random variables `{X_1,...,X_n}`,
  `E[ Log[sum_{i=1}^n X_i] ] != Log[ E[sum_{i=1}^n X_i] ]`.  As a result, this
  estimate is biased for finite `n`.  For `alpha < 1`, it is non-decreasing
  with `n` (in expectation).  For example, if `n = 1`, this estimator yields the
  same result as `elbo_ratio`, and as `n` increases the expected value
  of the estimator increases.

  #### Call signature

  User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

  Args:
    log_p:  Callable mapping samples from `q` to `Tensors` with
      shape broadcastable to `q.batch_shape`.
      For example, `log_p` works "just like" `q.log_prob`.
    q: `tf.contrib.distributions.BaseDistribution`.
       `float64` `dtype` recommended.
       `log_p` and `q` should be supported on the same set.
    alpha:  `Tensor` with shape `q.batch_shape` and values not equal to 1.
    z:  `Tensor` of samples from `q`, produced by `q.sample_n`.
    n:  Integer `Tensor`.  The number of samples to use if `z` is not provided.
      Note that this can be highly biased for small `n`, see docstring.
    seed:  Python integer to seed the random number generator.
    name:  A name to give this `Op`.

  Returns:
    renyi_result:  The scaled log of sample mean.  `Tensor` with `shape` equal
      to batch shape of `q`, and `dtype` = `q.dtype`.
  """
  with ops.name_scope(name, values=[alpha, n, z]):
    z = _get_samples(q, z, n, seed)

    # Evaluate sample mean in logspace.  Note that _logspace_mean will compute
    # (among other things) the mean of q.log_prob(z), which could also be
    # obtained with q.entropy().  However, DON'T use analytic entropy, because
    # that increases variance, and could result in NaN/Inf values of a sensitive
    # term.

    # log_values
    # = (1 - alpha) * ( Log p - Log q )
    log_values = (1. - alpha) * (log_p(z) - q.log_prob(z))

    # log_mean_values
    # = Log[ E[ values ] ]
    # = Log[ E[ (p / q)^{1-alpha} ] ]
    log_mean_values = _logspace_mean(log_values)

    return log_mean_values / (1. - alpha)


def renyi_alpha(step,
                decay_time,
                alpha_min,
                alpha_max=0.99999,
                name='renyi_alpha'):
  r"""Exponentially decaying `Tensor` appropriate for Renyi ratios.

  When minimizing the Renyi divergence for `0 <= alpha < 1` (or maximizing the
  Renyi equivalent of elbo) in high dimensions, it is not uncommon to experience
  `NaN` and `inf` values when `alpha` is far from `1`.

  For that reason, it is often desirable to start the optimization with `alpha`
  very close to 1, and reduce it to a final `alpha_min` according to some
  schedule.  The user may even want to optimize using `elbo_ratio` for
  some fixed time before switching to Renyi based methods.

  This `Op` returns an `alpha` decaying exponentially with step:

  ```
  s(step) = (exp{step / decay_time} - 1) / (e - 1)
  t(s) = max(0, min(s, 1)),  (smooth growth from 0 to 1)
  alpha(t) = (1 - t) alpha_min + t alpha_max
  ```

  Args:
    step:  Non-negative scalar `Tensor`.  Typically the global step or an
      offset version thereof.
    decay_time:  Positive scalar `Tensor`.
    alpha_min:  `float` or `double` `Tensor`.
      The minimal, final value of `alpha`, achieved when `step >= decay_time`
    alpha_max:  `Tensor` of same `dtype` as `alpha_min`.
      The maximal, beginning value of `alpha`, achieved when `step == 0`
    name:  A name to give this `Op`.

  Returns:
    alpha:  A `Tensor` of same `dtype` as `alpha_min`.
  """
  with ops.name_scope(name, values=[step, decay_time, alpha_min, alpha_max]):
    alpha_min = ops.convert_to_tensor(alpha_min, name='alpha_min')
    dtype = alpha_min.dtype

    alpha_max = ops.convert_to_tensor(alpha_max, dtype=dtype, name='alpha_max')
    decay_time = math_ops.cast(decay_time, dtype)
    step = math_ops.cast(step, dtype)

    check_scalars = [
        check_ops.assert_rank(step, 0, message='step must be scalar'),
        check_ops.assert_rank(
            decay_time, 0, message='decay_time must be scalar'),
        check_ops.assert_rank(alpha_min, 0, message='alpha_min must be scalar'),
        check_ops.assert_rank(alpha_max, 0, message='alpha_max must be scalar'),
    ]
    check_sign = [
        check_ops.assert_non_negative(
            step, message='step must be non-negative'),
        check_ops.assert_positive(
            decay_time, message='decay_time must be positive'),
    ]

    with ops.control_dependencies(check_scalars + check_sign):
      theta = (math_ops.exp(step / decay_time) - 1.) / (math.e - 1.)
      theta = math_ops.minimum(math_ops.maximum(theta, 0.), 1.)
      return alpha_max * (1. - theta) + alpha_min * theta
