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
"""The Gamma distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.distributions.python.ops import kullback_leibler
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops


class Gamma(distribution.Distribution):
  """The `Gamma` distribution with parameter alpha and beta.

  The parameters are the shape and inverse scale parameters alpha, beta.

  The PDF of this distribution is:

  ```pdf(x) = (beta^alpha)(x^(alpha-1))e^(-x*beta)/Gamma(alpha), x > 0```

  and the CDF of this distribution is:

  ```cdf(x) =  GammaInc(alpha, beta * x) / Gamma(alpha), x > 0```

  where GammaInc is the incomplete lower Gamma function.

  WARNING: This distribution may draw 0-valued samples for small alpha values.
      See the note on `tf.random_gamma`.

  Examples:

  ```python
  dist = Gamma(alpha=3.0, beta=2.0)
  dist2 = Gamma(alpha=[3.0, 4.0], beta=[2.0, 3.0])
  ```

  """

  def __init__(self,
               alpha,
               beta,
               validate_args=False,
               allow_nan_stats=True,
               name="Gamma"):
    """Construct Gamma distributions with parameters `alpha` and `beta`.

    The parameters `alpha` and `beta` must be shaped in a way that supports
    broadcasting (e.g. `alpha + beta` is a valid operation).

    Args:
      alpha: Floating point tensor, the shape params of the
        distribution(s).
        alpha must contain only positive values.
      beta: Floating point tensor, the inverse scale params of the
        distribution(s).
        beta must contain only positive values.
      validate_args: `Boolean`, default `False`.  Whether to assert that
        `a > 0`, `b > 0`, and that `x > 0` in the methods `prob(x)` and
        `log_prob(x)`.  If `validate_args` is `False` and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prepend to all ops created by this distribution.

    Raises:
      TypeError: if `alpha` and `beta` are different dtypes.
    """
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[alpha, beta]) as ns:
      with ops.control_dependencies([
          check_ops.assert_positive(alpha),
          check_ops.assert_positive(beta),
      ] if validate_args else []):
        self._alpha = array_ops.identity(alpha, name="alpha")
        self._beta = array_ops.identity(beta, name="beta")
        contrib_tensor_util.assert_same_float_dtype((self._alpha, self._beta))
    super(Gamma, self).__init__(
        dtype=self._alpha.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        is_continuous=True,
        is_reparameterized=False,
        parameters=parameters,
        graph_parents=[self._alpha, self._beta],
        name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("alpha", "beta"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def alpha(self):
    """Shape parameter."""
    return self._alpha

  @property
  def beta(self):
    """Inverse scale parameter."""
    return self._beta

  def _batch_shape(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.alpha), array_ops.shape(self.beta))

  def _get_batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.alpha.get_shape(), self.beta.get_shape())

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    """See the documentation for tf.random_gamma for more details."""
    return random_ops.random_gamma([n],
                                   self.alpha,
                                   beta=self.beta,
                                   dtype=self.dtype,
                                   seed=seed)

  def _log_prob(self, x):
    x = control_flow_ops.with_dependencies([check_ops.assert_positive(x)] if
                                           self.validate_args else [], x)
    contrib_tensor_util.assert_same_float_dtype(tensors=[x],
                                                dtype=self.dtype)
    return (self.alpha * math_ops.log(self.beta) +
            (self.alpha - 1.) * math_ops.log(x) -
            self.beta * x -
            math_ops.lgamma(self.alpha))

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    x = control_flow_ops.with_dependencies([check_ops.assert_positive(x)] if
                                           self.validate_args else [], x)
    contrib_tensor_util.assert_same_float_dtype(tensors=[x], dtype=self.dtype)
    # Note that igamma returns the regularized incomplete gamma function,
    # which is what we want for the CDF.
    return math_ops.log(math_ops.igamma(self.alpha, self.beta * x))

  def _cdf(self, x):
    return math_ops.igamma(self.alpha, self.beta * x)

  @distribution_util.AppendDocstring(
      """This is defined to be

      ```
      entropy = alpha - log(beta) + log(Gamma(alpha))
      + (1-alpha)digamma(alpha)
      ```

      where digamma(alpha) is the digamma function.
      """)
  def _entropy(self):
    return (self.alpha -
            math_ops.log(self.beta) +
            math_ops.lgamma(self.alpha) +
            (1. - self.alpha) * math_ops.digamma(self.alpha))

  def _mean(self):
    return self.alpha / self.beta

  def _variance(self):
    return self.alpha / math_ops.square(self.beta)

  def _std(self):
    return math_ops.sqrt(self.alpha) / self.beta

  @distribution_util.AppendDocstring(
      """The mode of a gamma distribution is `(alpha - 1) / beta` when
      `alpha > 1`, and `NaN` otherwise.  If `self.allow_nan_stats` is `False`,
      an exception will be raised rather than returning `NaN`.""")
  def _mode(self):
    mode = (self.alpha - 1.) / self.beta
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return array_ops.where(
          self.alpha >= 1.,
          mode,
          array_ops.fill(self.batch_shape(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              array_ops.ones((), self.dtype),
              self.alpha,
              message="mode not defined for components of alpha <= 1"),
          ], mode)


class GammaWithSoftplusAlphaBeta(Gamma):
  """Gamma with softplus transform on `alpha` and `beta`."""

  def __init__(self,
               alpha,
               beta,
               validate_args=False,
               allow_nan_stats=True,
               name="GammaWithSoftplusAlphaBeta"):
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[alpha, beta]) as ns:
      super(GammaWithSoftplusAlphaBeta, self).__init__(
          alpha=nn.softplus(alpha, name="softplus_alpha"),
          beta=nn.softplus(beta, name="softplus_beta"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters


@kullback_leibler.RegisterKL(Gamma, Gamma)
def _kl_gamma_gamma(g0, g1, name=None):
  """Calculate the batched KL divergence KL(g0 || g1) with g0 and g1 Gamma.

  Args:
    g0: instance of a Gamma distribution object.
    g1: instance of a Gamma distribution object.
    name: (optional) Name to use for created operations.
      Default is "kl_gamma_gamma".

  Returns:
    kl_gamma_gamma: `Tensor`. The batchwise KL(g0 || g1).
  """
  with ops.name_scope(name, "kl_gamma_gamma",
                      values=[g0.alpha, g0.beta, g1.alpha, g1.beta]):
    # Result from:
    #   http://www.fil.ion.ucl.ac.uk/~wpenny/publications/densities.ps
    # For derivation see:
    #   http://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions   pylint: disable=line-too-long
    return ((g0.alpha - g1.alpha) * math_ops.digamma(g0.alpha)
            + math_ops.lgamma(g1.alpha)
            - math_ops.lgamma(g0.alpha)
            + g1.alpha * math_ops.log(g0.beta)
            - g1.alpha * math_ops.log(g1.beta)
            + g0.alpha * (g1.beta / g0.beta - 1.))
