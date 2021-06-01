# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for kernelized_utils.py."""

import functools

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.keras.utils import kernelized_utils
from tensorflow.python.platform import test


def _exact_gaussian(stddev):
  return functools.partial(
      kernelized_utils.exact_gaussian_kernel, stddev=stddev)


def _exact_laplacian(stddev):
  return functools.partial(
      kernelized_utils.exact_laplacian_kernel, stddev=stddev)


class KernelizedUtilsTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('gaussian', _exact_gaussian(stddev=10.0), [[1.0]]),
      ('laplacian', _exact_laplacian(stddev=50.0), [[1.0]]))
  def test_equal_vectors(self, exact_kernel_fn, expected_values):
    """Identical vectors give exactly the identity kernel value."""
    x = constant_op.constant([0.5, -0.5, -0.5, 0.5])
    y = constant_op.constant([0.5, -0.5, -0.5, 0.5])
    exact_kernel = exact_kernel_fn(x, y)
    shape = exact_kernel.shape.as_list()
    self.assertLen(shape, 2)
    # x and y are identical and therefore K(x, y) will be precisely equal to
    # the identity value of the kernel.
    self.assertAllClose(expected_values, exact_kernel, atol=1e-6)

  @parameterized.named_parameters(
      ('gaussian', _exact_gaussian(stddev=10.0), [[1.0]]),
      ('laplacian', _exact_laplacian(stddev=50.0), [[1.0]]))
  def test_almost_identical_vectors(self, exact_kernel_fn, expected_values):
    """Almost identical vectors give the identity kernel value."""
    x = constant_op.constant([1.0, 0.4, -2.1, -1.1])
    y = constant_op.constant([1.01, 0.39, -2.099, -1.101])
    exact_kernel = exact_kernel_fn(x, y)
    shape = exact_kernel.shape.as_list()
    self.assertLen(shape, 2)
    # x and y are almost identical and therefore K(x, y) will be almost equal to
    # the identity value of the kernel.
    self.assertAllClose(expected_values, exact_kernel, atol=1e-3)

  @parameterized.named_parameters(
      ('gaussian', _exact_gaussian(stddev=1.0), [[0.99], [0.977]]),
      ('laplacian', _exact_laplacian(stddev=5.0), [[0.96], [0.94]]))
  def test_similar_matrices(self, exact_kernel_fn, expected_values):
    """Pairwise "close" vectors give high kernel values (similarity scores)."""
    x = constant_op.constant([1.0, 3.4, -2.1, 0.9, 3.3, -2.0], shape=[2, 3])
    y = constant_op.constant([1.1, 3.35, -2.05])
    exact_kernel = exact_kernel_fn(x, y)
    shape = exact_kernel.shape.as_list()
    self.assertLen(shape, 2)
    # The 2 rows of x are close to y. The pairwise kernel values (similarity
    # scores) are somewhat close to the identity value of the kernel.
    self.assertAllClose(expected_values, exact_kernel, atol=1e-2)

  @parameterized.named_parameters(
      ('gaussian', _exact_gaussian(stddev=2.0), [[.997, .279], [.251, 1.],
                                                 [.164, 0.019]]),
      ('laplacian', _exact_laplacian(stddev=2.0), [[.904, .128], [.116, 1.],
                                                   [.07, 0.027]]))
  def test_matrices_varying_similarity(self, exact_kernel_fn, expected_values):
    """Test matrices with row vectors of varying pairwise similarity."""
    x = constant_op.constant([1.0, 2., -2., 0.9, 3.3, -1.0], shape=[3, 2])
    y = constant_op.constant([1.1, 2.1, -2., 0.9], shape=[2, 2])
    exact_kernel = exact_kernel_fn(x, y)

    shape = exact_kernel.shape.as_list()
    self.assertLen(shape, 2)
    self.assertAllClose(expected_values, exact_kernel, atol=1e-2)

  @parameterized.named_parameters(
      ('gaussian', _exact_gaussian(stddev=1.0), [[0.0]]),
      ('laplacian', _exact_laplacian(stddev=1.0), [[0.0]]))
  def test_completely_dissimilar_vectors(self, exact_kernel_fn,
                                         expected_values):
    """Very dissimilar vectors give very low similarity scores."""
    x = constant_op.constant([1.0, 3.4, -2.1, -5.1])
    y = constant_op.constant([0.5, 2.1, 1.0, 3.0])
    exact_kernel = exact_kernel_fn(x, y)
    shape = exact_kernel.shape.as_list()
    self.assertLen(shape, 2)
    # x and y are very "far" from each other and so the corresponding kernel
    # value will be very low.
    self.assertAllClose(expected_values, exact_kernel, atol=1e-2)


if __name__ == '__main__':
  test.main()
