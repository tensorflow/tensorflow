# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for RandomFourierFeatureMapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.kernel_methods.python.mappers import dense_kernel_mapper
from tensorflow.contrib.kernel_methods.python.mappers.random_fourier_features import RandomFourierFeatureMapper
from tensorflow.python.framework import constant_op
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import googletest


def _inner_product(x, y):
  """Inner product between tensors x and y.

  The input tensors are assumed to be in ROW representation, that is, the method
  returns x * y^T.

  Args:
    x: input tensor in row format
    y: input tensor in row format

  Returns:
    the inner product of x, y
  """
  return math_ops.matmul(x, y, transpose_b=True)


def _compute_exact_rbf_kernel(x, y, stddev):
  """Computes exact RBF kernel given input tensors x and y and stddev."""
  diff = math_ops.subtract(x, y)
  diff_squared_norm = _inner_product(diff, diff)
  return math_ops.exp(-diff_squared_norm / (2 * stddev * stddev))


class RandomFourierFeatureMapperTest(TensorFlowTestCase):

  def testInvalidInputShape(self):
    x = constant_op.constant([[2.0, 1.0]])

    with self.test_session():
      rffm = RandomFourierFeatureMapper(3, 10)
      with self.assertRaisesWithPredicateMatch(
          dense_kernel_mapper.InvalidShapeError,
          r'Invalid dimension: expected 3 input features, got 2 instead.'):
        rffm.map(x)

  def testMappedShape(self):
    x1 = constant_op.constant([[2.0, 1.0, 0.0]])
    x2 = constant_op.constant([[1.0, -1.0, 2.0], [-1.0, 10.0, 1.0],
                               [4.0, -2.0, -1.0]])

    with self.test_session():
      rffm = RandomFourierFeatureMapper(3, 10, 1.0)
      mapped_x1 = rffm.map(x1)
      mapped_x2 = rffm.map(x2)
      self.assertEqual([1, 10], mapped_x1.get_shape())
      self.assertEqual([3, 10], mapped_x2.get_shape())

  def testSameOmegaReused(self):
    x = constant_op.constant([[2.0, 1.0, 0.0]])

    with self.test_session():
      rffm = RandomFourierFeatureMapper(3, 100)
      mapped_x = rffm.map(x)
      mapped_x_copy = rffm.map(x)
      # Two different evaluations of tensors output by map on the same input
      # are identical because the same paramaters are used for the mappings.
      self.assertAllClose(mapped_x.eval(), mapped_x_copy.eval(), atol=0.001)

  def testTwoMapperObjects(self):
    x = constant_op.constant([[2.0, 1.0, 0.0]])
    y = constant_op.constant([[1.0, -1.0, 2.0]])
    stddev = 3.0

    with self.test_session():
      # The mapped dimension is fairly small, so the kernel approximation is
      # very rough.
      rffm1 = RandomFourierFeatureMapper(3, 100, stddev)
      rffm2 = RandomFourierFeatureMapper(3, 100, stddev)
      mapped_x1 = rffm1.map(x)
      mapped_y1 = rffm1.map(y)
      mapped_x2 = rffm2.map(x)
      mapped_y2 = rffm2.map(y)

      approx_kernel_value1 = _inner_product(mapped_x1, mapped_y1)
      approx_kernel_value2 = _inner_product(mapped_x2, mapped_y2)
      self.assertAllClose(
          approx_kernel_value1.eval(), approx_kernel_value2.eval(), atol=0.01)

  def testBadKernelApproximation(self):
    x = constant_op.constant([[2.0, 1.0, 0.0]])
    y = constant_op.constant([[1.0, -1.0, 2.0]])
    stddev = 3.0

    with self.test_session():
      # The mapped dimension is fairly small, so the kernel approximation is
      # very rough.
      rffm = RandomFourierFeatureMapper(3, 100, stddev, seed=0)
      mapped_x = rffm.map(x)
      mapped_y = rffm.map(y)
      exact_kernel_value = _compute_exact_rbf_kernel(x, y, stddev)
      approx_kernel_value = _inner_product(mapped_x, mapped_y)
      self.assertAllClose(
          exact_kernel_value.eval(), approx_kernel_value.eval(), atol=0.2)

  def testGoodKernelApproximationAmortized(self):
    # Parameters.
    num_points = 20
    input_dim = 5
    mapped_dim = 5000
    stddev = 5.0

    # TODO(sibyl-vie3Poto): Reduce test's running time before moving to third_party. One
    # possible way to speed the test up is to compute both the approximate and
    # the exact kernel matrix directly using matrix operations instead of
    # computing the values for each pair of points separately.
    points_shape = [1, input_dim]
    points = [
        random_ops.random_uniform(shape=points_shape, maxval=1.0)
        for _ in xrange(num_points)
    ]

    normalized_points = [nn.l2_normalize(point, dim=1) for point in points]
    total_absolute_error = 0.0
    with self.test_session():
      rffm = RandomFourierFeatureMapper(input_dim, mapped_dim, stddev, seed=0)
      # Cache mappings so that they are not computed multiple times.
      cached_mappings = dict((point, rffm.map(point))
                             for point in normalized_points)
      for x in normalized_points:
        mapped_x = cached_mappings[x]
        for y in normalized_points:
          mapped_y = cached_mappings[y]
          exact_kernel_value = _compute_exact_rbf_kernel(x, y, stddev)
          approx_kernel_value = _inner_product(mapped_x, mapped_y)
          abs_error = math_ops.abs(exact_kernel_value - approx_kernel_value)
          total_absolute_error += abs_error
      self.assertAllClose(
          [[0.0]],
          total_absolute_error.eval() / (num_points * num_points),
          atol=0.02)


if __name__ == '__main__':
  googletest.main()
