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
"""The Poisson distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

__all__ = [
  'Poisson',
]


class Poisson(distribution.Distribution):
  """Poisson distribution.

  The Poisson distribution is parameterized by `lam`, the rate parameter.

  The pmf of this distribution is:

  ```

  pmf(k) = e^(-lam) * lam^k / k!,  k >= 0
  ```

  """

  def __init__(self,
               lam,
               validate_args=True,
               allow_nan_stats=False,
               name="Poisson"):
    """Construct Poisson distributions.

    Args:
      lam: Floating point tensor, the rate parameter of the
        distribution(s). `lam` must be positive.
      validate_args: Whether to assert that `lam > 0` as well as inputs to
        pmf computations are non-negative integers. If validate_args is
        `False`, then `pmf` computations might return NaN, as well as
        can be evaluated at any real value.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: A name for this distribution.
    """
    with ops.name_scope(name, values=[lam]) as scope:
      self._name = scope
      with ops.control_dependencies(
          [check_ops.assert_positive(lam)] if validate_args else []):
        self._lam = array_ops.identity(lam, name="lam")
        self._validate_args = validate_args
        self._allow_nan_stats = allow_nan_stats

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._lam.dtype

  @property
  def lam(self):
    """Rate parameter."""
    return self._lam

  @property
  def validate_args(self):
    """Boolean describing behavior on invalid input."""
    return self._validate_args

  @property
  def allow_nan_stats(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._allow_nan_stats

  def batch_shape(self, name="batch_shape"):
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[self.lam]):
        return array_ops.shape(self.lam)

  def get_batch_shape(self):
    return self.lam.get_shape()

  def event_shape(self, name="event_shape"):
    with ops.name_scope(self.name):
      with ops.name_scope(name):
        return constant_op.constant([], dtype=dtypes.int32)

  def get_event_shape(self):
    return tensor_shape.scalar()

  def log_cdf(self, x, name="log_cdf"):
    """Log cumulative density function.

    Args:
      x: Non-negative floating point tensor with dtype `dtype` and whose shape
        can be broadcast with `self.lam`.
      name: A name for this operation.

    Returns:
      The Log CDF of the events.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[x]):
        return math_ops.log(self.cdf(x))

  def cdf(self, x, name="cdf"):
    """Cumulative density function.

    Args:
      x: Non-negative floating point tensor with dtype `dtype` and whose shape
        can be broadcast with `self.lam`.
      name: A name for this operation.

    Returns:
      The CDF of the events.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[self.lam, x]):
        x = self._check_x(x, check_integer=False)
        return math_ops.igammac(math_ops.floor(x + 1), self.lam)

  def prob(self, x, name="prob"):
    """Probability mass function.

    Args:
      x: Non-negative floating point tensor with dtype `dtype` and whose shape
        can be broadcast with `self.lam`. `x` is only legal if it is
        non-negative and its components are equal to integer values.
      name: A name for this operation.

    Returns:
      The probabilities of the events.
    """
    return super(Poisson, self).prob(x, name)

  def log_prob(self, x, name="log_prob"):
    """Log probability mass function.

    Args:
      x: Non-negative floating point tensor with dtype `dtype` and whose shape
        can be broadcast with `self.lam`. `x` is only legal if it is
        non-negative and its components are equal to integer values.
      name: A name for this operation (optional).

    Returns:
      The log-probabilities of the events.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[self.lam, x]):
        x = self._check_x(x, check_integer=True)
        return x * math_ops.log(self.lam) - self.lam - math_ops.lgamma(x + 1)

  def mean(self, name="mean"):
    """Mean of the distribution.

    Args:
      name: Name for the op.

    Returns:
      mean: `Tensor` of the same type and shape as `lam`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[self.lam]):
        return array_ops.identity(self.lam)

  def variance(self, name="variance"):
    """Variance of the distribution.

    Args:
      name: Name for the op.

    Returns:
      variance: `Tensor` of the same type and shape as `lam`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[self.lam]):
        return array_ops.identity(self.lam)

  def std(self, name="std"):
    """Standard deviation of the distribution.

    Args:
      name: Name for the op.

    Returns:
      std: `Tensor` of the same type and shape as `lam`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[self.lam]):
        return math_ops.sqrt(self.variance())

  def mode(self, name="mode"):
    """Mode of the distribution.

    Note that when `lam` is an integer, there are actually two modes.
    Namely, `lam` and `lam - 1` are both modes. Here we return
    only the larger of the two modes.

    Args:
      name: Name for the op.

    Returns:
      mode: `Tensor` of the same type and shape as `lam`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name, values=[self.lam]):
        return math_ops.floor(self.lam)

  @property
  def is_continuous(self):
    return False

  @property
  def is_reparameterized(self):
    return False

  def _check_x(self, x, check_integer=True):
    with ops.name_scope('check_x', values=[x]):
      x = ops.convert_to_tensor(x, name="x")
      if not self.validate_args:
        return x
      dependencies = [check_ops.assert_non_negative(x)]
      if check_integer:
        dependencies += [distribution_util.assert_integer_form(
            x, message="x has non-integer components.")]
      return control_flow_ops.with_dependencies(dependencies, x)
