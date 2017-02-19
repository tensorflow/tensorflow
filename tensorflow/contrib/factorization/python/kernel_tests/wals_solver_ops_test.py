# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for wals_solver_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.factorization.python.ops import gen_factorization_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import test


def SparseBlock3x3():
  ind = np.array(
      [[0, 0], [0, 2], [1, 1], [2, 0], [2, 1], [3, 2]]).astype(np.int64)
  val = np.array([0.1, 0.2, 1.1, 2.0, 2.1, 3.2]).astype(np.float32)
  shape = np.array([4, 3]).astype(np.int64)
  return sparse_tensor.SparseTensor(ind, val, shape)


class WalsSolverOpsTest(test.TestCase):

  def setUp(self):
    self._column_factors = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]).astype(np.float32)
    self._row_factors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                                  [0.7, 0.8, 0.9],
                                  [1.1, 1.2, 1.3]]).astype(np.float32)
    self._column_weights = np.array([0.1, 0.2, 0.3]).astype(np.float32)
    self._row_weights = np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32)
    self._unobserved_weights = 0.1

  def testWalsSolverLhs(self):
    sparse_block = SparseBlock3x3()
    with self.test_session():
      [lhs_tensor,
       rhs_matrix] = gen_factorization_ops.wals_compute_partial_lhs_and_rhs(
           self._column_factors, self._column_weights, self._unobserved_weights,
           self._row_weights, sparse_block.indices, sparse_block.values,
           sparse_block.dense_shape[0], False)
      self.assertAllClose(lhs_tensor.eval(), [[
          [0.014800, 0.017000, 0.019200],
          [0.017000, 0.019600, 0.022200],
          [0.019200, 0.022200, 0.025200],
      ], [
          [0.0064000, 0.0080000, 0.0096000],
          [0.0080000, 0.0100000, 0.0120000],
          [0.0096000, 0.0120000, 0.0144000],
      ], [
          [0.0099000, 0.0126000, 0.0153000],
          [0.0126000, 0.0162000, 0.0198000],
          [0.0153000, 0.0198000, 0.0243000],
      ], [
          [0.058800, 0.067200, 0.075600],
          [0.067200, 0.076800, 0.086400],
          [0.075600, 0.086400, 0.097200],
      ]])
      self.assertAllClose(rhs_matrix.eval(), [[0.019300, 0.023000, 0.026700],
                                              [0.061600, 0.077000, 0.092400],
                                              [0.160400, 0.220000, 0.279600],
                                              [0.492800, 0.563200, 0.633600]])


if __name__ == "__main__":
  test.main()
