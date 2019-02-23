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
"""Functional tests for Matrix functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.opt.python.training import matrix_functions
from tensorflow.python.platform import test

TOLERANCE = 1e-3


def np_power(mat_g, alpha):
  """Computes mat_g^alpha for a square symmetric matrix mat_g."""

  mat_u, diag_d, mat_v = np.linalg.svd(mat_g)
  diag_d = np.power(diag_d, alpha)
  return np.dot(np.dot(mat_u, np.diag(diag_d)), mat_v)


class MatrixFunctionTests(test.TestCase):

  def testMatrixSquareRootFunction(self):
    """Tests for matrix square roots."""

    size = 20
    mat_a = np.random.rand(size, size)
    mat = np.dot(mat_a, mat_a.T)
    expected_mat = np_power(mat, 0.5)
    mat_root = matrix_functions.matrix_square_root(mat, size)
    self.assertAllCloseAccordingToType(
        expected_mat, mat_root, atol=TOLERANCE, rtol=TOLERANCE)

  def testMatrixInversePthRootFunction(self):
    """Tests for matrix inverse pth roots."""

    size = 20
    mat_a = np.random.rand(size, size)
    mat = np.dot(mat_a, mat_a.T)
    expected_mat = np_power(mat, -0.125)
    mat_root = matrix_functions.matrix_inverse_pth_root(mat, size, -0.125)
    self.assertAllCloseAccordingToType(
        expected_mat, mat_root, atol=TOLERANCE, rtol=TOLERANCE)


if __name__ == '__main__':
  test.main()
