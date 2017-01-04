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
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class Uniform(distribution.Distribution):
  """Uniform distribution with `a` and `b` parameters.

  The PDF of this distribution is constant between [`a`, `b`], and 0 elsewhere.
  """

  def __init__(self,
               a=0.,
               b=1.,
               validate_args=False,
               allow_nan_stats=True,
               name="Uniform"):
    """Construct Uniform distributions with `a` and `b`.

    The parameters `a` and `b` must be shaped in a way that supports
    broadcasting (e.g. `b - a` is a valid operation).

    Here are examples without broadcasting:

    ```python
    # Without broadcasting
    u1 = Uniform(3.0, 4.0)  # a single uniform distribution [3, 4]
    u2 = Uniform([1.0, 2.0], [3.0, 4.0])  # 2 distributions [1, 3], [2, 4]
    u3 = Uniform([[1.0, 2.0],
                  [3.0, 4.0]],
                 [[1.5, 2.5],
                  [3.5, 4.5]])  # 4 distributions
    ```

    And with broadcasting:

    ```python
    u1 = Uniform(3.0, [5.0, 6.0, 7.0])  # 3 distributions
    ```

    Args:
      a: Floating point tensor, the minimum endpoint.
      b: Floating point tensor, the maximum endpoint. Must be > `a`.
      validate_args: `Boolean`, default `False`.  Whether to validate input with
        asserts. If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prefix Ops created by this distribution class.

    Raises:
      InvalidArgumentError: if `a >= b` and `validate_args=False`.
    """
    parameters = locals()
    parameters.pop("self")
    with ops.name_scope(name, values=[a, b]) as ns:
      with ops.control_dependencies([
          check_ops.assert_less(
              a, b, message="uniform not defined when a > b.")
      ] if validate_args else []):
        self._a = array_ops.identity(a, name="a")
        self._b = array_ops.identity(b, name="b")
        contrib_tensor_util.assert_same_float_dtype((self._a, self._b))
    super(Uniform, self).__init__(
        dtype=self._a.dtype,
        is_reparameterized=True,
        is_continuous=True,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._a, self._b],
        name=ns)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("a", "b"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  def range(self, name="range"):
    """`b - a`."""
    with self._name_scope(name):
      return self.b - self.a

  def _batch_shape(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self._a), array_ops.shape(self._b))

  def _get_batch_shape(self):
    return array_ops.broadcast_static_shape(
        self._a.get_shape(), self._b.get_shape())

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    shape = array_ops.concat_v2(([n], self.batch_shape()), 0)
    samples = random_ops.random_uniform(shape=shape,
                                        dtype=self.dtype,
                                        seed=seed)
    return (array_ops.expand_dims(self.a, 0) +
            array_ops.expand_dims(self.range(), 0) * samples)

  def _log_prob(self, x):
    return math_ops.log(self._prob(x))

  def _prob(self, x):
    broadcasted_x = x * array_ops.ones(self.batch_shape())
    return array_ops.where(
        math_ops.is_nan(broadcasted_x),
        broadcasted_x,
        array_ops.where(
            math_ops.logical_or(broadcasted_x < self.a,
                                broadcasted_x > self.b),
            array_ops.zeros_like(broadcasted_x),
            (1. / self.range()) * array_ops.ones_like(broadcasted_x)))

  def _log_cdf(self, x):
    return math_ops.log(self.cdf(x))

  def _cdf(self, x):
    broadcasted_x = x * array_ops.ones(self.batch_shape())
    zeros = array_ops.zeros_like(x + self.a + self.b, dtype=self.dtype)
    ones = array_ops.ones_like(x + self.a + self.b, dtype=self.dtype)
    result_if_not_big = array_ops.where(
        x < self.a, zeros, (broadcasted_x - self.a) / self.range())
    return array_ops.where(x >= self.b, ones, result_if_not_big)

  def _entropy(self):
    return math_ops.log(self.range())

  def _mean(self):
    return (self.a + self.b) / 2.

  def _variance(self):
    return math_ops.square(self.range()) / 12.

  def _std(self):
    return self.range() / math.sqrt(12.)
