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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SparseSplitOpTest(test.TestCase):

  def _SparseTensor_4x6(self):
    # [0 |  |2 |  |4 |5 ]
    # [  |11|  |13|14|  ]
    # [20|  |  |23|  |25]
    # [30|  |32|33|  |35]
    ind = np.array([[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3], [1, 4],
                    [2, 0], [2, 3], [2, 5], [3, 0], [3, 2], [3, 3],
                    [3, 5]]).astype(np.int64)
    val = np.array(
        [0, 2, 4, 5, 11, 13, 14, 20, 23, 25, 30, 32, 33, 35]).astype(np.int64)
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
                    [1, 1, 1], [1, 3, 0], [1, 3, 1], [2, 2, 0],
                    [2, 2, 1]]).astype(np.int64)
    val = np.array(['a0', 'a1', 'b0', 'b1', 'c0', 'c1', 'd0', 'd1', 'e0', 'e1'])
    shape = np.array([3, 4, 2]).astype(np.int64)
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def _SparseTensor_3x4x2(self):
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_3x4x2(
    ))

  @test_util.run_deprecated_v1
  def testSplitMatrixRows(self):
    with self.session(use_gpu=False):
      sp_tensors = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_4x6(), num_split=2, axis=0)
      self.assertAllEqual(len(sp_tensors), 2)
      self.assertAllEqual(sp_tensors[0].indices.eval(), [[0, 0], [0, 2], [0, 4],
                                                         [0, 5], [1, 1], [1, 3],
                                                         [1, 4]])
      self.assertAllEqual(sp_tensors[0].values.eval(), [0, 2, 4, 5, 11, 13, 14])
      self.assertAllEqual(sp_tensors[0].dense_shape.eval(), [2, 6])
      self.assertAllEqual(sp_tensors[1].indices.eval(), [[0, 0], [0, 3], [0, 5],
                                                         [1, 0], [1, 2], [1, 3],
                                                         [1, 5]])
      self.assertAllEqual(sp_tensors[1].values.eval(),
                          [20, 23, 25, 30, 32, 33, 35])
      self.assertAllEqual(sp_tensors[1].dense_shape.eval(), [2, 6])

  @test_util.run_deprecated_v1
  def testSplitMatrixUnevenCols(self):
    with self.session(use_gpu=False):
      sp_tensors_3 = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_5x7(), num_split=3, axis=1)
      self.assertAllEqual(len(sp_tensors_3), 3)
      self.assertAllEqual(sp_tensors_3[0].indices.eval(),
                          [[0, 0], [0, 2], [1, 1], [2, 0], [3, 0], [3, 2],
                           [4, 1]])
      self.assertAllEqual(sp_tensors_3[0].values.eval(),
                          [0, 2, 11, 20, 30, 32, 41])
      self.assertAllEqual(sp_tensors_3[0].dense_shape.eval(), [5, 3])
      self.assertAllEqual(sp_tensors_3[1].indices.eval(),
                          [[0, 1], [1, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensors_3[1].values.eval(),
                          [4, 13, 14, 23, 33, 44])
      self.assertAllEqual(sp_tensors_3[1].dense_shape.eval(), [5, 2])
      self.assertAllEqual(sp_tensors_3[2].indices.eval(),
                          [[0, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensors_3[2].values.eval(), [5, 16, 25, 35, 46])
      self.assertAllEqual(sp_tensors_3[2].dense_shape.eval(), [5, 2])
      sp_tensors_4 = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_5x7(), num_split=4, axis=1)
      self.assertAllEqual(len(sp_tensors_4), 4)
      self.assertAllEqual(sp_tensors_4[0].indices.eval(),
                          [[0, 0], [1, 1], [2, 0], [3, 0], [4, 1]])
      self.assertAllEqual(sp_tensors_4[0].values.eval(), [0, 11, 20, 30, 41])
      self.assertAllEqual(sp_tensors_4[0].dense_shape.eval(), [5, 2])
      self.assertAllEqual(sp_tensors_4[1].indices.eval(),
                          [[0, 0], [1, 1], [2, 1], [3, 0], [3, 1]])
      self.assertAllEqual(sp_tensors_4[1].values.eval(), [2, 13, 23, 32, 33])
      self.assertAllEqual(sp_tensors_4[1].dense_shape.eval(), [5, 2])
      self.assertAllEqual(sp_tensors_4[2].indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [2, 1], [3, 1], [4, 0]])
      self.assertAllEqual(sp_tensors_4[2].values.eval(), [4, 5, 14, 25, 35, 44])
      self.assertAllEqual(sp_tensors_4[2].dense_shape.eval(), [5, 2])
      self.assertAllEqual(sp_tensors_4[3].indices.eval(), [[1, 0], [4, 0]])
      self.assertAllEqual(sp_tensors_4[3].values.eval(), [16, 46])
      self.assertAllEqual(sp_tensors_4[3].dense_shape.eval(), [5, 1])

  @test_util.run_deprecated_v1
  def testSplitMatrixUnevenRows(self):
    with self.session(use_gpu=False):
      sp_tensors_2 = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_5x7(), num_split=2, axis=0)
      self.assertAllEqual(sp_tensors_2[0].indices.eval(),
                          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3],
                           [1, 4], [1, 6], [2, 0], [2, 3], [2, 5]])
      self.assertAllEqual(sp_tensors_2[0].values.eval(),
                          [0, 2, 4, 5, 11, 13, 14, 16, 20, 23, 25])
      self.assertAllEqual(sp_tensors_2[0].dense_shape.eval(), [3, 7])
      self.assertAllEqual(sp_tensors_2[1].indices.eval(),
                          [[0, 0], [0, 2], [0, 3], [0, 5], [1, 1], [1, 4],
                           [1, 6]])
      self.assertAllEqual(sp_tensors_2[1].values.eval(),
                          [30, 32, 33, 35, 41, 44, 46])
      self.assertAllEqual(sp_tensors_2[1].dense_shape.eval(), [2, 7])
      self.assertAllEqual(len(sp_tensors_2), 2)
      sp_tensors_3 = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_5x7(), num_split=3, axis=0)
      self.assertAllEqual(len(sp_tensors_3), 3)
      self.assertAllEqual(sp_tensors_3[0].indices.eval(),
                          [[0, 0], [0, 2], [0, 4], [0, 5], [1, 1], [1, 3],
                           [1, 4], [1, 6]])
      self.assertAllEqual(sp_tensors_3[0].values.eval(),
                          [0, 2, 4, 5, 11, 13, 14, 16])
      self.assertAllEqual(sp_tensors_3[0].dense_shape.eval(), [2, 7])

      self.assertAllEqual(sp_tensors_3[1].values.eval(),
                          [20, 23, 25, 30, 32, 33, 35])
      self.assertAllEqual(sp_tensors_3[1].dense_shape.eval(), [2, 7])
      self.assertAllEqual(sp_tensors_3[2].indices.eval(), [[0, 1], [0, 4],
                                                           [0, 6]])
      self.assertAllEqual(sp_tensors_3[2].values.eval(), [41, 44, 46])
      self.assertAllEqual(sp_tensors_3[2].dense_shape.eval(), [1, 7])
    return

  @test_util.run_deprecated_v1
  def testSplitAllRows(self):
    with self.session(use_gpu=False):
      sp_tensors = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_4x6(), num_split=4, axis=0)
      self.assertAllEqual(len(sp_tensors), 4)
      self.assertAllEqual(sp_tensors[0].indices.eval(), [[0, 0], [0, 2], [0, 4],
                                                         [0, 5]])
      self.assertAllEqual(sp_tensors[0].values.eval(), [0, 2, 4, 5])
      self.assertAllEqual(sp_tensors[0].dense_shape.eval(), [1, 6])
      self.assertAllEqual(sp_tensors[1].indices.eval(), [[0, 1], [0, 3], [0,
                                                                          4]])
      self.assertAllEqual(sp_tensors[1].values.eval(), [11, 13, 14])
      self.assertAllEqual(sp_tensors[1].dense_shape.eval(), [1, 6])
      self.assertAllEqual(sp_tensors[2].indices.eval(), [[0, 0], [0, 3], [0,
                                                                          5]])
      self.assertAllEqual(sp_tensors[2].values.eval(), [20, 23, 25])
      self.assertAllEqual(sp_tensors[2].dense_shape.eval(), [1, 6])
      self.assertAllEqual(sp_tensors[3].indices.eval(), [[0, 0], [0, 2], [0, 3],
                                                         [0, 5]])
      self.assertAllEqual(sp_tensors[3].values.eval(), [30, 32, 33, 35])
      self.assertAllEqual(sp_tensors[3].dense_shape.eval(), [1, 6])

  @test_util.run_deprecated_v1
  def testSplitColumns(self):
    with self.session(use_gpu=False):
      sparse_tensors = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_4x6(), num_split=3, axis=1)
      self.assertAllEqual(len(sparse_tensors), 3)
      self.assertAllEqual(sparse_tensors[0].indices.eval(), [[0, 0], [1, 1],
                                                             [2, 0], [3, 0]])
      self.assertAllEqual(sparse_tensors[0].values.eval(), [0, 11, 20, 30])
      self.assertAllEqual(sparse_tensors[0].dense_shape.eval(), [4, 2])
      self.assertAllEqual(sparse_tensors[1].indices.eval(),
                          [[0, 0], [1, 1], [2, 1], [3, 0], [3, 1]])
      self.assertAllEqual(sparse_tensors[1].values.eval(), [2, 13, 23, 32, 33])
      self.assertAllEqual(sparse_tensors[1].dense_shape.eval(), [4, 2])
      self.assertAllEqual(sparse_tensors[2].indices.eval(),
                          [[0, 0], [0, 1], [1, 0], [2, 1], [3, 1]])
      self.assertAllEqual(sparse_tensors[2].values.eval(), [4, 5, 14, 25, 35])
      self.assertAllEqual(sparse_tensors[2].dense_shape.eval(), [4, 2])

  @test_util.run_deprecated_v1
  def testSplitAllColumns(self):
    with self.session(use_gpu=False):
      sparse_tensors = sparse_ops.sparse_split(
          sp_input=self._SparseTensor_4x6(), num_split=6, axis=1)
      self.assertAllEqual(len(sparse_tensors), 6)
      self.assertAllEqual(sparse_tensors[0].indices.eval(), [[0, 0], [2, 0],
                                                             [3, 0]])
      self.assertAllEqual(sparse_tensors[0].values.eval(), [0, 20, 30])
      self.assertAllEqual(sparse_tensors[0].dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensors[1].indices.eval(), [[1, 0]])
      self.assertAllEqual(sparse_tensors[1].values.eval(), [11])
      self.assertAllEqual(sparse_tensors[1].dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensors[2].indices.eval(), [[0, 0], [3, 0]])
      self.assertAllEqual(sparse_tensors[2].values.eval(), [2, 32])
      self.assertAllEqual(sparse_tensors[2].dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensors[3].indices.eval(), [[1, 0], [2, 0],
                                                             [3, 0]])
      self.assertAllEqual(sparse_tensors[3].dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensors[3].values.eval(), [13, 23, 33])
      self.assertAllEqual(sparse_tensors[4].indices.eval(), [[0, 0], [1, 0]])
      self.assertAllEqual(sparse_tensors[4].values.eval(), [4, 14])
      self.assertAllEqual(sparse_tensors[4].dense_shape.eval(), [4, 1])
      self.assertAllEqual(sparse_tensors[5].indices.eval(), [[0, 0], [2, 0],
                                                             [3, 0]])
      self.assertAllEqual(sparse_tensors[5].values.eval(), [5, 25, 35])
      self.assertAllEqual(sparse_tensors[5].dense_shape.eval(), [4, 1])

  @test_util.run_deprecated_v1
  def testSliceConcat(self):
    for sp_input in (self._SparseTensorValue_3x4x2(),
                     self._SparseTensor_3x4x2()):
      with self.cached_session(use_gpu=False):
        sparse_tensors = sparse_ops.sparse_split(
            sp_input=sp_input, num_split=2, axis=1)
        concat_tensor = sparse_ops.sparse_concat(1, sparse_tensors)
        expected_output = self._SparseTensor_3x4x2()
        self.assertAllEqual(concat_tensor.indices.eval(),
                            expected_output.indices.eval())

  def testArgumentErrors(self):
    with self.assertRaisesRegex(ValueError, 'Keyword arguments are required'):
      sparse_ops.sparse_split(3, 2, 1)
    with self.assertRaisesRegex(ValueError, 'sp_input is required'):
      sparse_ops.sparse_split()
    with self.assertRaisesRegex(ValueError, 'num_split is required'):
      sparse_ops.sparse_split(sp_input=1)
    with self.assertRaisesRegex(ValueError, 'axis is required'):
      sparse_ops.sparse_split(num_split=2, sp_input=1)


if __name__ == '__main__':
  test.main()
