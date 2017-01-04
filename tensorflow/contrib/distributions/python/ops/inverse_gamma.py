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
"""The InverseGamma distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
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


class InverseGamma(distribution.Distribution):
  """The `InverseGamma` distribution with parameter alpha and beta.

  The parameters are the shape and inverse scale parameters alpha, beta.

  The PDF of this distribution is:

  ```pdf(x) = (beta^alpha)/Gamma(alpha)(x^(-alpha-1))e^(-beta/x), x > 0```

  and the CDF of this distribution is:

  ```cdf(x) =  GammaInc(alpha, beta / x) / Gamma(alpha), x > 0```

  where GammaInc is the upper incomplete Gamma function.

  Examples:

  ```python
  dist = InverseGamma(alpha=3.0, beta=2.0)
  dist2 = InverseGamma(alpha=[3.0, 4.0], beta=[2.0, 3.0])
  ```

  """

  def __init__(self,
               alpha,
               beta,
               validate_args=False,
               allow_nan_stats=True,
               name="InverseGamma"):
    """Construct InverseGamma distributions with parameters `alpha` and `beta`.

    The parameters `alpha` and `beta` must be shaped in a way that supports
    broadcasting (e.g. `alpha + beta` is a valid operation).

    Args:
      alpha: Floating point tensor, the shape params of the
        distribution(s).
        alpha must contain only positive values.
      beta: Floating point tensor, the scale params of the distribution(s).
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
    super(InverseGamma, self).__init__(
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
    """Scale parameter."""
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
    return 1. / random_ops.random_gamma([n], self.alpha, beta=self.beta,
                                        dtype=self.dtype, seed=seed)

  def _log_prob(self, x):
    x = control_flow_ops.with_dependencies([check_ops.assert_positive(x)] if
                                           self.validate_args else [], x)
    return (self.alpha * math_ops.log(self.beta) -
            math_ops.lgamma(self.alpha) -
            (self.alpha + 1.) * math_ops.log(x) - self.beta / x)

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return math_ops.log(self._cdf(x))

  def _cdf(self, x):
    x = control_flow_ops.with_dependencies([check_ops.assert_positive(x)] if
                                           self.validate_args else [], x)
    # Note that igammac returns the upper regularized incomplete gamma
    # function Q(a, x), which is what we want for the CDF.
    return math_ops.igammac(self.alpha, self.beta / x)

  @distribution_util.AppendDocstring(
      """This is defined to be

      ```
      entropy = alpha - log(beta) + log(Gamma(alpha))
      + (1-alpha)digamma(alpha)
      ```

      where digamma(alpha) is the digamma function.""")
  def _entropy(self):
    return (self.alpha +
            math_ops.log(self.beta) +
            math_ops.lgamma(self.alpha) -
            (1. + self.alpha) * math_ops.digamma(self.alpha))

  @distribution_util.AppendDocstring(
      """The mean of an inverse gamma distribution is `beta / (alpha - 1)`,
      when `alpha > 1`, and `NaN` otherwise.  If `self.allow_nan_stats` is
      `False`, an exception will be raised rather than returning `NaN`""")
  def _mean(self):
    mean = self.beta / (self.alpha - 1.)
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return array_ops.where(
          self.alpha > 1., mean,
          array_ops.fill(self.batch_shape(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              array_ops.ones((), self.dtype), self.alpha,
              message="mean not defined for components of self.alpha <= 1"),
      ], mean)

  @distribution_util.AppendDocstring(
      """Variance for inverse gamma is defined only for `alpha > 2`. If
      `self.allow_nan_stats` is `False`, an exception will be raised rather
      than returning `NaN`.""")
  def _variance(self):
    var = (math_ops.square(self.beta) /
           (math_ops.square(self.alpha - 1.) * (self.alpha - 2.)))
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return array_ops.where(
          self.alpha > 2., var,
          array_ops.fill(self.batch_shape(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              constant_op.constant(2., dtype=self.dtype), self.alpha,
              message="variance not defined for components of alpha <= 2"),
      ], var)

  def _mode(self):
    """The mode of an inverse gamma distribution is `beta / (alpha + 1)`."""
    return self.beta / (self.alpha + 1.)


class InverseGammaWithSoftplusAlphaBeta(InverseGamma):
  """Inverse Gamma with softplus applied to `alpha` and `beta`."""

  def __init__(self,
               alpha,
               beta,
               validate_args=False,
               allow_nan_stats=True,
               name="InverseGammaWithSoftplusAlphaBeta"):
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[alpha, beta]) as ns:
      super(InverseGammaWithSoftplusAlphaBeta, self).__init__(
          alpha=nn.softplus(alpha),
          beta=nn.softplus(beta),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
    self._parameters = parameters
