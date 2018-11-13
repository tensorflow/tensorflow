# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SparseReorder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import sparse_ops
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SparseSliceOpTest(test.TestCase):

  def _SparseTensor_4x6(self, val_dtype=np.int64):
    # [0 |  |2 |  |4 |5 ]
    # [  |11|  |13|14|  ]
    # [20|  |  |23|  |25]
    # [30|  |32|33|  |35]
    ind = np.array([[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1,
                                                                     4], [2, 0],
                    [2, 3], [2, 5], [3, 0], [3, 2], [3, 3], [3, 5]]).astype(
                        np.int64)
    val = np.array([0, 2, 4, 5, 11, 13, 14, 20, 23, 25, 30, 32, 33, 35]).astype(
        val_dtype)
    shape = np.array([4, 6]).astype(np.int64)
    return sparse_tensor.SparseTensor(ind, val, shape)

  def _SparseTensor_5x7(self):
    # [0 |  |2 |  |4 |5 |  ]
    # [  |11|  |13|14|  |16]
    # [20|  |  |23|  |25|  ]
    # [30|  |32|33|  |35|  ]
    # [  |41|  |  |44|  |46]
    ind = np.array([[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4],
                    [1, 6], [2, 0], [2, 3], [2, 5], [3, 0], [3, 2], [3, 3],
                    [3, 5], [4, 1], [4, 4], [4, 6]]).astype(np.int64)
    val = np.array(
        [0, 2, 4, 5, 11, 13, 14, 16, 20, 23, 25, 30, 32, 33, 35, 41, 44,
         46]).astype(np.int64)
    shape = np.array([5, 7]).astype(np.int64)
    return sparse_tensor.SparseTensor(ind, val, shape)

  def _SparseTensorValue_3x4x2(self):
    #  slice(:,:, 0)
    #  ['a0'|    |'b0'|    ]
    #  [    |'c0'|    |'d0']
    #  [    |    |'e0'|    ]
    #  slice(:,:, 1)
    #  ['a1'|    |'b1'|    ]
    #  [    |'c1'|    |'d1']
    #  [    |    |'e1'|    ]
    ind = np.array([[0, 0, 0], [0, 0, 1], [0, 2, 0], [0, 2, 1], [1, 1, 0],
                    [1, 1, 1], [1, 3, 0], [1, 3, 1], [2, 2, 0], [2, 2,
                                                                 1]]).astype(
                                                                     np.int64)
    val = np.array(['a0', 'a1', 'b0', 'b1', 'c0', 'c1', 'd0', 'd1', 'e0', 'e1'])
    shape = np.array([3, 4, 2]).astype(np.int64)
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def _SparseTensor_3x4x2(self):
    return sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_3x4x2())

  def testSliceMatrixRows(self):
    with self.session(use_gpu=False):
      sp_input = self._SparseTensor_4x6()
      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [2, 6])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [2, 0], [3, 7])
      self.assertAllEqual(
          sp_tensor0.indices.eval(),
          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4]])
      self.assertAllEqual(sp_tensor0.values.eval(), [0, 2, 4, 5, 11, 13, 14])
      self.assertAllEqual(sp_tensor0.dense_shape.eval(), [2, 6])
      self.assertAllEqual(
          sp_tensor1.indices.eval(),
          [[0, 0], [0, 3], [0, 5], [1, 0], [1, 2], [1, 3], [1, 5]])
      self.assertAllEqual(sp_tensor1.values.eval(),
                          [20, 23, 25, 30, 32, 33, 35])
      self.assertAllEqual(sp_tensor1.dense_shape.eval(), [2, 6])

  def testSliceMatrixUnevenCols(self):
    with self.session(use_gpu=False):
      sp_input = self._SparseTensor_5x7()
      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [5, 3])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [0, 3], [5, 2])
      sp_tensor2 = sparse_ops.sparse_slice(sp_input, [0, 5], [5, 2])

      self.assertAllEqual(
          sp_tensor0.indices.eval(),
          [[0, 0], [0, 2], [1, 1], [2, 0], [3, 0], [3, 2], [4, 1]])
      self.assertAllEqual(sp_tensor0.values.eval(), [0, 2, 11, 20, 30, 32, 41])
      self.assertAllEqual(sp_tensor0.dense_shape.eval(), [5, 3])
      self.assertAllEqual(sp_tensor1.indices.eval(),
                          [[0, 1], [1, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensor1.values.eval(), [4, 13, 14, 23, 33, 44])
      self.assertAllEqual(sp_tensor1.dense_shape.eval(), [5, 2])
      self.assertAllEqual(sp_tensor2.indices.eval(),
                          [[0, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensor2.values.eval(), [5, 16, 25, 35, 46])
      self.assertAllEqual(sp_tensor2.dense_shape.eval(), [5, 2])

      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [5, 2])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [0, 2], [5, 2])
      sp_tensor2 = sparse_ops.sparse_slice(sp_input, [0, 4], [5, 2])
      sp_tensor3 = sparse_ops.sparse_slice(sp_input, [0, 6], [5, 2])
      self.assertAllEqual(sp_tensor0.indices.eval(),
                          [[0, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensor0.values.eval(), [0, 11, 20, 30, 41])
      self.assertAllEqual(sp_tensor0.dense_shape.eval(), [5, 2])
      self.assertAllEqual(sp_tensor1.indices.eval(),
                          [[0, 0], [1, 1], [2, 1], [3, 0], [3, 1]])
      self.assertAllEqual(sp_tensor1.values.eval(), [2, 13, 23, 32, 33])
      self.assertAllEqual(sp_tensor1.dense_shape.eval(), [5, 2])
      self.assertAllEqual(sp_tensor2.indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [2, 1], [3, 1], [4, 0]])
      self.assertAllEqual(sp_tensor2.values.eval(), [4, 5, 14, 25, 35, 44])
      self.assertAllEqual(sp_tensor2.dense_shape.eval(), [5, 2])
      self.assertAllEqual(sp_tensor3.indices.eval(), [[1, 0], [4, 0]])
      self.assertAllEqual(sp_tensor3.values.eval(), [16, 46])
      self.assertAllEqual(sp_tensor3.dense_shape.eval(), [5, 1])

  def testSliceMatrixUnevenRows(self):
    with self.session(use_gpu=False):
      sp_input = self._SparseTensor_5x7()
      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [3, 7])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [3, 0], [3, 7])
      self.assertAllEqual(sp_tensor0.indices.eval(),
                          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3],
                           [1, 4], [1, 6], [2, 0], [2, 3], [2, 5]])
      self.assertAllEqual(sp_tensor0.values.eval(),
                          [0, 2, 4, 5, 11, 13, 14, 16, 20, 23, 25])
      self.assertAllEqual(sp_tensor0.dense_shape.eval(), [3, 7])
      self.assertAllEqual(
          sp_tensor1.indices.eval(),
          [[0, 0], [0, 2], [0, 3], [0, 5], [1, 1], [1, 4], [1, 6]])
      self.assertAllEqual(sp_tensor1.values.eval(),
                          [30, 32, 33, 35, 41, 44, 46])
      self.assertAllEqual(sp_tensor1.dense_shape.eval(), [2, 7])

      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [2, 7])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [2, 0], [2, 7])
      sp_tensor2 = sparse_ops.sparse_slice(sp_input, [4, 0], [2, 7])
      self.assertAllEqual(
          sp_tensor0.indices.eval(),
          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4], [1, 6]])
      self.assertAllEqual(sp_tensor0.values.eval(),
                          [0, 2, 4, 5, 11, 13, 14, 16])
      self.assertAllEqual(sp_tensor0.dense_shape.eval(), [2, 7])

      self.assertAllEqual(sp_tensor1.values.eval(),
                          [20, 23, 25, 30, 32, 33, 35])
      self.assertAllEqual(sp_tensor1.dense_shape.eval(), [2, 7])
      self.assertAllEqual(sp_tensor2.indices.eval(), [[0, 1], [0, 4], [0, 6]])
      self.assertAllEqual(sp_tensor2.values.eval(), [41, 44, 46])
      self.assertAllEqual(sp_tensor2.dense_shape.eval(), [1, 7])
    return

  def testSliceAllRows(self):
    with self.session(use_gpu=False):
      sp_input = self._SparseTensor_4x6()
      sp_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [1, 6])
      sp_tensor1 = sparse_ops.sparse_slice(sp_input, [1, 0], [1, 6])
      sp_tensor2 = sparse_ops.sparse_slice(sp_input, [2, 0], [1, 7])
      sp_tensor3 = sparse_ops.sparse_slice(sp_input, [3, 0], [2, 7])
      self.assertAllEqual(sp_tensor0.indices.eval(),
                          [[0, 0], [0, 2], [0, 4], [0, 5]])
      self.assertAllEqual(sp_tensor0.values.eval(), [0, 2, 4, 5])
      self.assertAllEqual(sp_tensor0.dense_shape.eval(), [1, 6])
      self.assertAllEqual(sp_tensor1.indices.eval(), [[0, 1], [0, 3], [0, 4]])
      self.assertAllEqual(sp_tensor1.values.eval(), [11, 13, 14])
      self.assertAllEqual(sp_tensor1.dense_shape.eval(), [1, 6])
      self.assertAllEqual(sp_tensor2.indices.eval(), [[0, 0], [0, 3], [0, 5]])
      self.assertAllEqual(sp_tensor2.values.eval(), [20, 23, 25])
      self.assertAllEqual(sp_tensor2.dense_shape.eval(), [1, 6])
      self.assertAllEqual(sp_tensor3.indices.eval(),
                          [[0, 0], [0, 2], [0, 3], [0, 5]])
      self.assertAllEqual(sp_tensor3.values.eval(), [30, 32, 33, 35])
      self.assertAllEqual(sp_tensor3.dense_shape.eval(), [1, 6])

  def testSliceColumns(self):
    with self.session(use_gpu=False):
      sp_input = self._SparseTensor_4x6()
      sparse_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [4, 2])
      sparse_tensor1 = sparse_ops.sparse_slice(sp_input, [0, 2], [5, 2])
      sparse_tensor2 = sparse_ops.sparse_slice(sp_input, [0, 4], [5, 3])

      self.assertAllEqual(sparse_tensor0.indices.eval(),
                          [[0, 0], [1, 1], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor0.values.eval(), [0, 11, 20, 30])
      self.assertAllEqual(sparse_tensor0.dense_shape.eval(), [4, 2])
      self.assertAllEqual(sparse_tensor1.indices.eval(),
                          [[0, 0], [1, 1], [2, 1], [3, 0], [3, 1]])
      self.assertAllEqual(sparse_tensor1.values.eval(), [2, 13, 23, 32, 33])
      self.assertAllEqual(sparse_tensor1.dense_shape.eval(), [4, 2])
      self.assertAllEqual(sparse_tensor2.indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [2, 1], [3, 1]])
      self.assertAllEqual(sparse_tensor2.values.eval(), [4, 5, 14, 25, 35])
      self.assertAllEqual(sparse_tensor2.dense_shape.eval(), [4, 2])

  def testSliceAllColumns(self):
    with self.session(use_gpu=False):
      sp_input = self._SparseTensor_4x6()
      sparse_tensor0 = sparse_ops.sparse_slice(sp_input, [0, 0], [4, 1])
      sparse_tensor1 = sparse_ops.sparse_slice(sp_input, [0, 1], [4, 1])
      sparse_tensor2 = sparse_ops.sparse_slice(sp_input, [0, 2], [4, 1])
      sparse_tensor3 = sparse_ops.sparse_slice(sp_input, [0, 3], [4, 1])
      sparse_tensor4 = sparse_ops.sparse_slice(sp_input, [0, 4], [5, 1])
      sparse_tensor5 = sparse_ops.sparse_slice(sp_input, [0, 5], [6, 3])
      self.assertAllEqual(sparse_tensor0.indices.eval(),
                          [[0, 0], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor0.values.eval(), [0, 20, 30])
      self.assertAllEqual(sparse_tensor0.dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensor1.indices.eval(), [[1, 0]])
      self.assertAllEqual(sparse_tensor1.values.eval(), [11])
      self.assertAllEqual(sparse_tensor1.dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensor2.indices.eval(), [[0, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor2.values.eval(), [2, 32])
      self.assertAllEqual(sparse_tensor2.dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensor3.indices.eval(),
                          [[1, 0], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor3.dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensor3.values.eval(), [13, 23, 33])
      self.assertAllEqual(sparse_tensor4.indices.eval(), [[0, 0], [1, 0]])
      self.assertAllEqual(sparse_tensor4.values.eval(), [4, 14])
      self.assertAllEqual(sparse_tensor4.dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensor5.indices.eval(),
                          [[0, 0], [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensor5.values.eval(), [5, 25, 35])
      self.assertAllEqual(sparse_tensor5.dense_shape.eval(), [4, 1])

  def testGradients(self):
    sp_input = self._SparseTensor_4x6(val_dtype=np.float32)
    start_and_size = [([0, 0], [4, 2]),
                      ([0, 2], [5, 2]),
                      ([0, 4], [5, 3])]

    with self.session(use_gpu=False):
      for start, size in start_and_size:
        sp_output = sparse_ops.sparse_slice(sp_input, start, size)
        nnz_in = len(sp_input.values.eval())
        nnz_out = len(sp_output.values.eval())

        err = gradient_checker.compute_gradient_error(
            [sp_input.values], [(nnz_in,)], sp_output.values, (nnz_out,))
        self.assertLess(err, 1e-3)


if __name__ == '__main__':
  test.main()
