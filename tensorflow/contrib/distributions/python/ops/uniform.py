# Copyright 2016 Google Inc. All Rights Reserved.
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

from tensorflow.contrib.distributions.python.ops.distribution import ContinuousDistribution  # pylint: disable=line-too-long
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util  # pylint: disable=line-too-long
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class Uniform(ContinuousDistribution):
  """Uniform distribution with `a` and `b` parameters.

  The PDF of this distribution is constant between [`a`, `b`], and 0 elsewhere.
  """

  def __init__(self, a=0.0, b=1.0, name="Uniform"):
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
      a: `float` or `double` tensor, the minimum endpoint.
      b: `float` or `double` tensor, the maximum endpoint. Must be > `a`.
      name: The name to prefix Ops created by this distribution class.

    Raises:
      InvalidArgumentError: if `a >= b`.
    """
    with ops.op_scope([a, b], name):
      with ops.control_dependencies([check_ops.assert_less(a, b)]):
        a = ops.convert_to_tensor(a, name="a")
        b = ops.convert_to_tensor(b, name="b")
        if a.dtype != b.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s" %
                          (a.dtype, b.dtype))

    self._a = a
    self._b = b
    self._name = name
    self._batch_shape = self._ones().get_shape()
    self._event_shape = tensor_shape.TensorShape([])

    contrib_tensor_util.assert_same_float_dtype((a, b))

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self.a.dtype

  def batch_shape(self, name="batch_shape"):
    with ops.name_scope(self.name):
      return array_ops.shape(self._ones(), name=name)

  def get_batch_shape(self):
    return self._batch_shape

  def event_shape(self, name="event_shape"):
    with ops.name_scope(self.name):
      return constant_op.constant(1, name=name)

  def get_event_shape(self):
    return self._event_shape

  @property
  def a(self):
    return self._a

  @property
  def b(self):
    return self._b

  def pdf(self, x, name="pdf"):
    """The PDF of observations in `x` under these Uniform distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `a` and `b`.
      name: The name to give this op.

    Returns:
      pdf: tensor of dtype `dtype`, the pdf values of `x`. If `x` is `nan`, will
          return `nan`.
    """
    with ops.op_scope([self.a, self.b, x], self.name):
      with ops.name_scope(name):
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
                (1.0 / self.range) * array_ops.ones_like(broadcasted_x)))

  def log_pdf(self, x, name="log_pdf"):
    return super(Uniform, self).log_pdf(x, name)

  def cdf(self, x, name="cdf"):
    """CDF of observations in `x` under these Uniform distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `a` and `b`.
      name: The name to give this op.

    Returns:
      cdf: tensor of dtype `dtype`, the CDFs of `x`. If `x` is `nan`, will
          return `nan`.
    """
    with ops.op_scope([self.a, self.b, x], self.name):
      with ops.name_scope(name):
        x = ops.convert_to_tensor(x, name="x")
        if x.dtype != self.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s" %
                          (x.dtype, self.dtype))

    broadcasted_x = x * self._ones()
    return math_ops.select(broadcasted_x < self.a,
                           array_ops.zeros_like(broadcasted_x),
                           math_ops.select(broadcasted_x >= self.b,
                                           array_ops.ones_like(broadcasted_x),
                                           (broadcasted_x - self.a) /
                                           self.range))

  def log_cdf(self, x, name="log_cdf"):
    with ops.op_scope([self.a, self.b, x], self.name):
      with ops.name_scope(name):
        x = ops.convert_to_tensor(x, name="x")
        return math_ops.log(self.cdf(x))

  def entropy(self, name="entropy"):
    """The entropy of Uniform distribution(s).

    Args:
      name: The name to give this op.

    Returns:
      entropy: tensor of dtype `dtype`, the entropy.
    """
    with ops.op_scope([self.a, self.b], self.name):
      with ops.name_scope(name):
        return math_ops.log(self.range)

  def sample(self, n, seed=None, name="sample"):
    """Sample `n` observations from the Uniform Distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
          with values of type `self.dtype`.
    """
    with ops.op_scope([self.a, self.b, n], self.name):
      with ops.name_scope(name):
        n = ops.convert_to_tensor(n, name="n")
        n_val = tensor_util.constant_value(n)

        shape = array_ops.concat(0, [array_ops.pack([n]), self.batch_shape()])
        samples = random_ops.random_uniform(shape=shape,
                                            dtype=self.dtype,
                                            seed=seed)

        # Provide some hints to shape inference
        inferred_shape = tensor_shape.vector(n_val).concatenate(
            self.get_batch_shape())
        samples.set_shape(inferred_shape)

        return (array_ops.expand_dims(self.a, 0) + array_ops.expand_dims(
            self.range, 0) * samples)

  @property
  def mean(self):
    return (self.a + self.b) / 2

  @property
  def variance(self):
    return math_ops.square(self.range) / 12

  @property
  def range(self):
    """`b - a`."""
    return self.b - self.a

  # TODO(rsepassi): Find a more efficient way of doing the broadcasting in_ones
  # and _zeros.
  def _ones(self):
    return array_ops.ones_like(self.a + self.b)

  def _zeros(self):
    return array_ops.zeros_like(self.a + self.b)
