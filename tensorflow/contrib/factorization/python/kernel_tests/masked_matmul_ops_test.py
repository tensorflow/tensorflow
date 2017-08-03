# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for masked_matmul_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-bad-todo, g-import-not-at-top
import numpy as np

from tensorflow.contrib.factorization.python.ops import gen_factorization_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def MakeMask():
  inds = [[0, 0], [0, 2], [1, 1], [2, 0], [2, 3]] * 100
  indices = np.array(inds).astype(np.int64)
  shape = np.array([5, 4]).astype(np.int64)
  return (indices, shape)


class MaskedProductOpsTest(test.TestCase):

  def setUp(self):
    a = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.1, 1.2, 1.3],
        [1.4, 1.5, 1.6],
    ]
    b = [
        [0.1, 0.4, 0.7, 1.1],
        [0.2, 0.5, 0.8, 1.2],
        [0.3, 0.6, 0.9, 1.3],
    ]
    self._dot_products = np.array([0.14, 0.5, 0.77, 0.5, 2.9] * 100)
    self._a = np.array(a).astype(np.float32)
    self._b = np.array(b).astype(np.float32)
    self._mask_ind, self._mask_shape = MakeMask()

  def _runTestMaskedProduct(self, transpose_a, transpose_b):
    with ops.Graph().as_default(), self.test_session() as sess:
      a = self._a if not transpose_a else array_ops.transpose(self._a)
      b = self._b if not transpose_b else array_ops.transpose(self._b)

      def AssertClose(sp_x, sp_y):
        x_inds, x_vals, y_inds, y_vals = sess.run(
            [sp_x.indices, sp_x.values,
             sp_y.indices, sp_y.values])
        self.assertAllClose(x_inds, y_inds)
        self.assertAllClose(x_vals, y_vals)

      values = gen_factorization_ops.masked_matmul(
          a, b, self._mask_ind, transpose_a, transpose_b)
      result = sparse_tensor.SparseTensor(
          self._mask_ind, values, self._mask_shape)
      true_result = sparse_tensor.SparseTensor(
          self._mask_ind, self._dot_products, self._mask_shape)
      AssertClose(result, true_result)

  def _runTestEmptyMaskedProduct(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      empty_mask = constant_op.constant(0, shape=[0, 2], dtype=dtypes.int64)
      values = gen_factorization_ops.masked_matmul(
          self._a, self._b, empty_mask, False, False)
      self.assertEqual(len(values.eval(session=sess)), 0)

  def testMaskedProduct(self):
    self._runTestMaskedProduct(False, False)

  def testMaskedProductTransposeA(self):
    self._runTestMaskedProduct(True, False)

  def testMaskedProductTransposeB(self):
    self._runTestMaskedProduct(False, True)

  def testMaskedProductTransposeAAndB(self):
    self._runTestMaskedProduct(True, True)

  def testEmptyMaskedProduct(self):
    self._runTestEmptyMaskedProduct()


if __name__ == "__main__":
  test.main()
