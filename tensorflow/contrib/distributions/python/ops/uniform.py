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

from tensorflow.contrib.distributions.python.ops import distribution  # pylint: disable=line-too-long
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util  # pylint: disable=line-too-long
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class Uniform(distribution.Distribution):
  """Uniform distribution with `a` and `b` parameters.

  The PDF of this distribution is constant between [`a`, `b`], and 0 elsewhere.
  """

  def __init__(self,
               a=0.0,
               b=1.0,
               validate_args=True,
               allow_nan_stats=False,
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
      validate_args: Whether to assert that `a > b`. If `validate_args` is
        `False` and inputs are invalid, correct behavior is not guaranteed.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to prefix Ops created by this distribution class.

    Raises:
      InvalidArgumentError: if `a >= b` and `validate_args=True`.
    """
    self._allow_nan_stats = allow_nan_stats
    self._validate_args = validate_args
    with ops.op_scope([a, b], name):
      with ops.control_dependencies([check_ops.assert_less(
          a, b, message="uniform not defined when a > b.")] if validate_args
                                    else []):
        a = array_ops.identity(a, name="a")
        b = array_ops.identity(b, name="b")

    self._a = a
    self._b = b
    self._name = name
    self._batch_shape = self._ones().get_shape()
    self._event_shape = tensor_shape.TensorShape([])

    contrib_tensor_util.assert_same_float_dtype((a, b))

  @property
  def allow_nan_stats(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._allow_nan_stats

  @property
  def validate_args(self):
    """Boolean describing behavior on invalid input."""
    return self._validate_args

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self.a.dtype

  def batch_shape(self, name="batch_shape"):
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return array_ops.shape(self._ones())

  def get_batch_shape(self):
    return self._batch_shape

  def event_shape(self, name="event_shape"):
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return constant_op.constant([], dtype=dtypes.int32)

  def get_event_shape(self):
    return self._event_shape

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  def prob(self, x, name="prob"):
    """The PDF of observations in `x` under these Uniform distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `a` and `b`.
      name: The name to give this op.

    Returns:
      prob: tensor of dtype `dtype`, the prob values of `x`. If `x` is `nan`,
          will return `nan`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.a, self.b, x], name):
        x = ops.convert_to_tensor(x, name="x")
        if x.dtype != self.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s" %
                          (x.dtype, self.dtype))

        broadcasted_x = x * self._ones()
        return math_ops.select(
            math_ops.is_nan(broadcasted_x), broadcasted_x, math_ops.select(
                math_ops.logical_or(broadcasted_x < self.a,
                                    broadcasted_x > self.b),
                array_ops.zeros_like(broadcasted_x),
                (1.0 / self.range()) * array_ops.ones_like(broadcasted_x)))

  def log_prob(self, x, name="log_prob"):
    return super(Uniform, self).log_prob(x, name)

  def cdf(self, x, name="cdf"):
    """CDF of observations in `x` under these Uniform distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `a` and `b`.
      name: The name to give this op.

    Returns:
      cdf: tensor of dtype `dtype`, the CDFs of `x`. If `x` is `nan`, will
          return `nan`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.a, self.b, x], name):
        x = ops.convert_to_tensor(x, name="x")
        if x.dtype != self.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s" %
                          (x.dtype, self.dtype))

        broadcasted_x = x * self._ones()
        zeros = array_ops.zeros_like(x + self.a + self.b, dtype=self.dtype)
        ones = array_ops.ones_like(x + self.a + self.b, dtype=self.dtype)
        result_if_not_big = math_ops.select(
            x < self.a, zeros, (broadcasted_x - self.a) / self.range())
        return math_ops.select(x >= self.b, ones, result_if_not_big)

  def log_cdf(self, x, name="log_cdf"):
    with ops.name_scope(self.name):
      with ops.op_scope([self.a, self.b, x], name):
        x = ops.convert_to_tensor(x, name="x")
        return math_ops.log(self.cdf(x))

  def entropy(self, name="entropy"):
    """The entropy of Uniform distribution(s).

    Args:
      name: The name to give this op.

    Returns:
      entropy: tensor of dtype `dtype`, the entropy.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.a, self.b, self.range()], name):
        return math_ops.log(self.range())

  def sample_n(self, n, seed=None, name="sample_n"):
    """Sample `n` observations from the Uniform Distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
          with values of type `self.dtype`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.a, self.b, n], name):
        n = ops.convert_to_tensor(n, name="n")
        n_val = tensor_util.constant_value(n)

        shape = array_ops.concat(0, ([n], self.batch_shape()))
        samples = random_ops.random_uniform(shape=shape,
                                            dtype=self.dtype,
                                            seed=seed)

        # Provide some hints to shape inference
        inferred_shape = tensor_shape.vector(n_val).concatenate(
            self.get_batch_shape())
        samples.set_shape(inferred_shape)

        return (array_ops.expand_dims(self.a, 0) + array_ops.expand_dims(
            self.range(), 0) * samples)

  def mean(self, name="mean"):
    with ops.name_scope(self.name):
      with ops.op_scope([self._a, self._b], name):
        return (self.a + self.b) / 2

  def variance(self, name="variance"):
    with ops.name_scope(self.name):
      with ops.op_scope([self.range()], name):
        return math_ops.square(self.range()) / 12.

  def std(self, name="std"):
    with ops.name_scope(self.name):
      with ops.op_scope([self.range()], name):
        return self.range() / math_ops.sqrt(12.)

  def range(self, name="range"):
    """`b - a`."""
    with ops.name_scope(self.name):
      with ops.op_scope([self.a, self.b], name):
        return self.b - self.a

  @property
  def is_reparameterized(self):
    return True

  # TODO(rsepassi): Find a more efficient way of doing the broadcasting in_ones
  # and _zeros.
  def _ones(self):
    return array_ops.ones_like(self.a + self.b)

  def _zeros(self):
    return array_ops.zeros_like(self.a + self.b)

  @property
  def is_continuous(self):
    return True
