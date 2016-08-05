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

# pylint: disable=line-too-long

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

# pylint: enable=line-too-long


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

  def __init__(self, a, b, validate_args=True, allow_nan_stats=False,
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
      validate_args: Whether to assert valid values for parameters `a` and `b`,
        and `x` in `prob` and `log_prob`.  If `False`, correct behavior is not
        guaranteed.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
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
    with ops.op_scope([a, b], name):
      with ops.control_dependencies([
          check_ops.assert_positive(a),
          check_ops.assert_positive(b)] if validate_args else []):
        a = array_ops.identity(a, name="a")
        b = array_ops.identity(b, name="b")

      self._a = a
      self._b = b
      self._name = name

      # Used for mean/mode/variance/entropy/sampling computations
      self._a_b_sum = self._a + self._b

      self._get_batch_shape = self._a_b_sum.get_shape()
      self._get_event_shape = tensor_shape.TensorShape([])
      self._validate_args = validate_args
      self._allow_nan_stats = allow_nan_stats

  @property
  def a(self):
    """Shape parameter."""
    return self._a

  @property
  def b(self):
    """Shape parameter."""
    return self._b

  @property
  def name(self):
    """Name to prepend to all ops."""
    return self._name

  @property
  def dtype(self):
    """dtype of samples from this distribution."""
    return self._a_b_sum.dtype

  @property
  def allow_nan_stats(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._allow_nan_stats

  @property
  def validate_args(self):
    """Boolean describing behavior on invalid input."""
    return self._validate_args

  def batch_shape(self, name="batch_shape"):
    """Batch dimensions of this instance as a 1-D int32 `Tensor`.

    The product of the dimensions of the `batch_shape` is the number of
    independent distributions of this kind the instance represents.

    Args:
      name: name to give to the op

    Returns:
      `Tensor` `batch_shape`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._a_b_sum], name):
        return array_ops.shape(self._a_b_sum)

  def get_batch_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `batch_shape`. May be only partially defined.

    Returns:
      batch shape
    """
    return self._get_batch_shape

  def event_shape(self, name="event_shape"):
    """Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      `Tensor` `event_shape`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return constant_op.constant([], name=name, dtype=dtypes.int32)

  def get_event_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      event shape
    """
    return self._get_event_shape

  def mean(self, name="mean"):
    """Mean of the distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._a, self._a_b_sum], name):
        return self._a / self._a_b_sum

  def variance(self, name="variance"):
    """Variance of the distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._a, self._b, self._a_b_sum], name):
        return (self._a * self._b) / (
            self._a_b_sum **2 * (self._a_b_sum + 1))

  def std(self, name="std"):
    """Standard deviation of the distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return math_ops.sqrt(self.variance())

  def mode(self, name="mode"):
    """Mode of the distribution.

    Note that the mode for the Beta distribution is only defined
    when `a > 1`, `b > 1`. This returns the mode when `a > 1` and `b > 1`,
    and NaN otherwise. If `self.allow_nan_stats` is `False`, an exception
    will be raised rather than returning `NaN`.

    Args:
      name: The name for this op.

    Returns:
      Mode of the Beta distribution.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._a, self._b, self._a_b_sum], name):
        a = self._a
        b = self._b
        a_b_sum = self._a_b_sum
        one = constant_op.constant(1, self.dtype)
        mode = (a - 1)/ (a_b_sum - 2)

        if self.allow_nan_stats:
          return math_ops.select(
              math_ops.logical_and(
                  math_ops.greater(a, 1), math_ops.greater(b, 1)),
              mode,
              (constant_op.constant(float("NaN"), dtype=self.dtype) *
               array_ops.ones_like(a_b_sum, dtype=self.dtype)))
        else:
          return control_flow_ops.with_dependencies([
              check_ops.assert_less(
                  one, a,
                  message="mode not defined for components of a <= 1"
              ),
              check_ops.assert_less(
                  one, b,
                  message="mode not defined for components of b <= 1"
              )], mode)

  def entropy(self, name="entropy"):
    """Entropy of the distribution in nats."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._a, self._b, self._a_b_sum], name):
        a = self._a
        b = self._b
        a_b_sum = self._a_b_sum

        entropy = math_ops.lgamma(a) - (a - 1) * math_ops.digamma(a)
        entropy += math_ops.lgamma(b) - (b - 1) * math_ops.digamma(b)
        entropy += -math_ops.lgamma(a_b_sum) + (
            a_b_sum - 2) * math_ops.digamma(a_b_sum)
        return entropy

  def cdf(self, x, name="cdf"):
    """Cumulative distribution function."""
    # TODO(srvasude): Implement this once betainc op is checked in.
    raise NotImplementedError("Beta cdf not implemented.")

  def log_cdf(self, x, name="log_cdf"):
    """Log CDF."""
    raise NotImplementedError("Beta cdf not implemented.")

  def log_prob(self, x, name="log_prob"):
    """`Log(P[counts])`, computed for every batch member.

    Args:
      x:  Non-negative floating point tensor whose shape can
        be broadcast with `self.a` and `self.b`.  For fixed leading
        dimensions, the last dimension represents counts for the corresponding
        Beta distribution in `self.a` and `self.b`. `x` is only legal if
        0 < x < 1.
      name:  Name to give this Op, defaults to "log_prob".

    Returns:
      Log probabilities for each record, shape `[N1,...,Nm]`.
    """
    a = self._a
    b = self._b
    with ops.name_scope(self.name):
      with ops.op_scope([a, x], name):
        x = self._check_x(x)

        unnorm_pdf = (a - 1) * math_ops.log(x) + (
            b - 1) * math_ops.log(1 - x)
        normalization_factor = -(math_ops.lgamma(a) + math_ops.lgamma(b)
                                 - math_ops.lgamma(a + b))
        log_prob = unnorm_pdf + normalization_factor

        return log_prob

  def prob(self, x, name="prob"):
    """`P[x]`, computed for every batch member.

    Args:
      x:  Non-negative floating point tensor whose shape can
        be broadcast with `self.a` and `self.b`.  For fixed leading
        dimensions, the last dimension represents x for the corresponding Beta
        distribution in `self.a` and `self.b`. `x` is only legal if is
        between 0 and 1.
      name:  Name to give this Op, defaults to "pdf".

    Returns:
      Probabilities for each record, shape `[N1,...,Nm]`.
    """
    return super(Beta, self).prob(x, name=name)

  def sample_n(self, n, seed=None, name="sample_n"):
    """Sample `n` observations from the Beta Distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: `[n, ...]`, a `Tensor` of `n` samples for each
        of the distributions determined by broadcasting the hyperparameters.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.a, self.b, n], name):
        a = array_ops.ones_like(self._a_b_sum, dtype=self.dtype) * self.a
        b = array_ops.ones_like(self._a_b_sum, dtype=self.dtype) * self.b
        gamma1_sample = random_ops.random_gamma(
            [n,], a, dtype=self.dtype, seed=seed)
        gamma2_sample = random_ops.random_gamma(
            [n,], b, dtype=self.dtype, seed=seed)

        # This is equal to gamma1_sample / (gamma1_sample + gamma2_sample)
        # but is more numerically stable.
        beta_sample = gamma1_sample / (gamma1_sample + gamma2_sample)

        n_val = tensor_util.constant_value(n)
        final_shape = tensor_shape.vector(n_val).concatenate(
            self._a_b_sum.get_shape())

        beta_sample.set_shape(final_shape)
        return beta_sample

  @property
  def is_continuous(self):
    return True

  @property
  def is_reparameterized(self):
    return False

  def _check_x(self, x):
    """Check x for proper shape, values, then return tensor version."""
    x = ops.convert_to_tensor(x, name="x_before_deps")
    dependencies = [
        check_ops.assert_positive(x),
        check_ops.assert_less(x, constant_op.constant(
            1, self.dtype))] if self.validate_args else []
    return control_flow_ops.with_dependencies(dependencies, x)
