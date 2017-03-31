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
"""The Uniform distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class Uniform(distribution.Distribution):
  """Uniform distribution with `low` and `high` parameters.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; a, b) = I[a <= x < b] / Z
  Z = b - a
  ```

  where:
  * `low = a`,
  * `high = b`,
  * `Z` is the normalizing constant, and,
  * `I[predicate]` is the [indicator function](
    https://en.wikipedia.org/wiki/Indicator_function) for `predicate`.

  The parameters `low` and `high` must be shaped in a way that supports
  broadcasting (e.g., `high - low` is a valid operation).

  #### Examples

  ```python
  # Without broadcasting:
  u1 = Uniform(low=3.0, high=4.0)  # a single uniform distribution [3, 4]
  u2 = Uniform(low=[1.0, 2.0],
               high=[3.0, 4.0])  # 2 distributions [1, 3], [2, 4]
  u3 = Uniform(low=[[1.0, 2.0],
                    [3.0, 4.0]],
               high=[[1.5, 2.5],
                     [3.5, 4.5]])  # 4 distributions
  ```

  ```python
  # With broadcasting:
  u1 = Uniform(low=3.0, high=[5.0, 6.0, 7.0])  # 3 distributions
  ```

  """

  def __init__(self,
               low=0.,
               high=1.,
               validate_args=False,
               allow_nan_stats=True,
               name="Uniform"):
    """Initialize a batch of Uniform distributions.

    Args:
      low: Floating point tensor, lower boundary of the output interval. Must
        have `low < high`.
      high: Floating point tensor, upper boundary of the output interval. Must
        have `low < high`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      InvalidArgumentError: if `low >= high` and `validate_args=False`.
    """
    parameters = locals()
    with ops.name_scope(name, values=[low, high]):
      with ops.control_dependencies([
          check_ops.assert_less(
              low, high, message="uniform not defined when low >= high.")
      ] if validate_args else []):
        self._low = array_ops.identity(low, name="low")
        self._high = array_ops.identity(high, name="high")
        check_ops.assert_same_float_dtype([self._low, self._high])
    super(Uniform, self).__init__(
        dtype=self._low.dtype,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._low,
                       self._high],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("low", "high"),
            ([ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def low(self):
    """Lower boundary of the output interval."""
    return self._low

  @property
  def high(self):
    """Upper boundary of the output interval."""
    return self._high

  def range(self, name="range"):
    """`high - low`."""
    with self._name_scope(name):
      return self.high - self.low

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.low),
        array_ops.shape(self.high))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.low.get_shape(),
        self.high.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    samples = random_ops.random_uniform(shape=shape,
                                        dtype=self.dtype,
                                        seed=seed)
    return self.low + self.range() * samples

  def _log_prob(self, x):
    return math_ops.log(self._prob(x))

  def _prob(self, x):
    broadcasted_x = x * array_ops.ones(self.batch_shape_tensor())
    return array_ops.where(
        math_ops.is_nan(broadcasted_x),
        broadcasted_x,
        array_ops.where(
            math_ops.logical_or(broadcasted_x < self.low,
                                broadcasted_x >= self.high),
            array_ops.zeros_like(broadcasted_x),
            array_ops.ones_like(broadcasted_x) / self.range()))

  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  def _cdf(self, x):
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(x), self.batch_shape_tensor())
    zeros = array_ops.zeros(broadcast_shape, dtype=self.dtype)
    ones = array_ops.ones(broadcast_shape, dtype=self.dtype)
    broadcasted_x = x * ones
    result_if_not_big = array_ops.where(
        x < self.low, zeros, (broadcasted_x - self.low) / self.range())
    return array_ops.where(x >= self.high, ones, result_if_not_big)

  def _entropy(self):
    return math_ops.log(self.range())

  def _mean(self):
    return (self.low + self.high) / 2.

  def _variance(self):
    return math_ops.square(self.range()) / 12.

  def _stddev(self):
    return self.range() / math.sqrt(12.)
