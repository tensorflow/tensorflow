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
"""The Beta distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.distributions.python.ops import distribution_util
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


class Beta(distribution.Distribution):
  """Beta distribution.

  This distribution is parameterized by `a` and `b` which are shape
  parameters.

  #### Mathematical details

  The Beta is a distribution over the interval (0, 1).
  The distribution has hyperparameters `a` and `b` and
  probability mass function (pdf):

  ```pdf(x) = 1 / Beta(a, b) * x^(a - 1) * (1 - x)^(b - 1)```

  where `Beta(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b)`
  is the beta function.


  This class provides methods to create indexed batches of Beta
  distributions. One entry of the broacasted
  shape represents of `a` and `b` represents one single Beta distribution.
  When calling distribution functions (e.g. `dist.pdf(x)`), `a`, `b`
  and `x` are broadcast to the same shape (if possible).
  Every entry in a/b/x corresponds to a single Beta distribution.

  #### Examples

  Creates 3 distributions.
  The distribution functions can be evaluated on x.

  ```python
  a = [1, 2, 3]
  b = [1, 2, 3]
  dist = Beta(a, b)
  ```

  ```python
  # x same shape as a.
  x = [.2, .3, .7]
  dist.pdf(x)  # Shape [3]

  # a/b will be broadcast to [[1, 2, 3], [1, 2, 3]] to match x.
  x = [[.1, .4, .5], [.2, .3, .5]]
  dist.pdf(x)  # Shape [2, 3]

  # a/b will be broadcast to shape [5, 7, 3] to match x.
  x = [[...]]  # Shape [5, 7, 3]
  dist.pdf(x)  # Shape [5, 7, 3]
  ```

  Creates a 2-batch of 3-class distributions.

  ```python
  a = [[1, 2, 3], [4, 5, 6]]  # Shape [2, 3]
  b = 5  # Shape []
  dist = Beta(a, b)

  # x will be broadcast to [[.2, .3, .9], [.2, .3, .9]] to match a/b.
  x = [.2, .3, .9]
  dist.pdf(x)  # Shape [2]
  ```

  """

  def __init__(self, a, b, validate_args=False, allow_nan_stats=True,
               name="Beta"):
    """Initialize a batch of Beta distributions.

    Args:
      a:  Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
         different Beta distributions. This also defines the
         dtype of the distribution.
      b:  Positive floating point tensor with shape broadcastable to
        `[N1,..., Nm]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
         different Beta distributions.
      validate_args: `Boolean`, default `False`.  Whether to assert valid
        values for parameters `a`, `b`, and `x` in `prob` and `log_prob`.
        If `False` and inputs are invalid, correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prefix Ops created by this distribution class.

    Examples:

    ```python
    # Define 1-batch.
    dist = Beta(1.1, 2.0)

    # Define a 2-batch.
    dist = Beta([1.0, 2.0], [4.0, 5.0])
    ```

    """
    with ops.name_scope(name, values=[a, b]) as ns:
      with ops.control_dependencies([
          check_ops.assert_positive(a),
          check_ops.assert_positive(b),
      ] if validate_args else []):
        self._a = array_ops.identity(a, name="a")
        self._b = array_ops.identity(b, name="b")
        contrib_tensor_util.assert_same_float_dtype((self._a, self._b))
        # Used for mean/mode/variance/entropy/sampling computations
        self._a_b_sum = self._a + self._b
        super(Beta, self).__init__(
            dtype=self._a_b_sum.dtype,
            parameters={"a": self._a, "b": self._b, "a_b_sum": self._a_b_sum},
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            is_continuous=True,
            is_reparameterized=False,
            name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("a", "b"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def a(self):
    """Shape parameter."""
    return self._a

  @property
  def b(self):
    """Shape parameter."""
    return self._b

  @property
  def a_b_sum(self):
    """Sum of parameters."""
    return self._a_b_sum

  def _batch_shape(self):
    return array_ops.shape(self.a_b_sum)

  def _get_batch_shape(self):
    return self.a_b_sum.get_shape()

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    a = array_ops.ones_like(self.a_b_sum, dtype=self.dtype) * self.a
    b = array_ops.ones_like(self.a_b_sum, dtype=self.dtype) * self.b
    gamma1_sample = random_ops.random_gamma(
        [n,], a, dtype=self.dtype, seed=seed)
    gamma2_sample = random_ops.random_gamma(
        [n,], b, dtype=self.dtype, seed=seed)
    beta_sample = gamma1_sample / (gamma1_sample + gamma2_sample)
    return beta_sample

  def _log_prob(self, x):
    x = self._assert_valid_sample(x)
    log_unnormalized_prob = ((self.a - 1.) * math_ops.log(x) +
                             (self.b - 1.) * math_ops.log(1. - x))
    log_normalization = (math_ops.lgamma(self.a) +
                         math_ops.lgamma(self.b) -
                         math_ops.lgamma(self.a_b_sum))
    return log_unnormalized_prob - log_normalization

  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))

  def _log_cdf(self, x):
    return math_ops.log(self._cdf(x))

  def _cdf(self, x):
    return math_ops.betainc(self.a, self.b, x)

  def _entropy(self):
    return (math_ops.lgamma(self.a) -
            (self.a - 1.) * math_ops.digamma(self.a) +
            math_ops.lgamma(self.b) -
            (self.b - 1.) * math_ops.digamma(self.b) -
            math_ops.lgamma(self.a_b_sum) +
            (self.a_b_sum - 2.) * math_ops.digamma(self.a_b_sum))

  def _mean(self):
    return self.a / self.a_b_sum

  def _variance(self):
    return (self.a * self.b) / (self.a_b_sum**2. * (self.a_b_sum + 1.))

  def _std(self):
    return math_ops.sqrt(self.variance())

  def _mode(self):
    mode = (self.a - 1.)/ (self.a_b_sum - 2.)
    if self.allow_nan_stats:
      nan = np.array(np.nan, dtype=self.dtype.as_numpy_dtype())
      return math_ops.select(
          math_ops.logical_and(
              math_ops.greater(self.a, 1.),
              math_ops.greater(self.b, 1.)),
          mode,
          array_ops.fill(self.batch_shape(), nan, name="nan"))
    else:
      return control_flow_ops.with_dependencies([
          check_ops.assert_less(
              array_ops.ones((), dtype=self.dtype), self.a,
              message="Mode not defined for components of a <= 1."),
          check_ops.assert_less(
              array_ops.ones((), dtype=self.dtype), self.b,
              message="Mode not defined for components of b <= 1."),
      ], mode)

  def _assert_valid_sample(self, x):
    """Check x for proper shape, values, then return tensor version."""
    if not self.validate_args: return x
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(
            x,
            message="Negative events lie outside Beta distribution support."),
        check_ops.assert_less(
            x, array_ops.ones((), self.dtype),
            message="Event>=1 lies outside Beta distribution support."),
    ], x)


_prob_note = """

    Note that the argument `x` must be a non-negative floating point tensor
    whose shape can be broadcast with `self.a` and `self.b`.  For fixed leading
    dimensions, the last dimension represents counts for the corresponding Beta
    distribution in `self.a` and `self.b`. `x` is only legal if `0 < x < 1`.
"""

distribution_util.append_class_fun_doc(Beta.log_prob, doc_str=_prob_note)
distribution_util.append_class_fun_doc(Beta.prob, doc_str=_prob_note)

distribution_util.append_class_fun_doc(Beta.mode, doc_str="""

    Note that the mode for the Beta distribution is only defined
    when `a > 1`, `b > 1`. This returns the mode when `a > 1` and `b > 1`,
    and `NaN` otherwise. If `self.allow_nan_stats` is `False`, an exception
    will be raised rather than returning `NaN`.
""")


class BetaWithSoftplusAB(Beta):
  """Beta with softplus transform on `a` and `b`."""

  def __init__(self,
               a,
               b,
               validate_args=False,
               allow_nan_stats=True,
               name="BetaWithSoftplusAB"):
    with ops.name_scope(name, values=[a, b]) as ns:
      super(BetaWithSoftplusAB, self).__init__(
          a=nn.softplus(a),
          b=nn.softplus(b),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)
