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
"""Tests for SparseConcat."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SparseConcatTest(test.TestCase):

  def _SparseTensor_UnknownShape(self,
                                 ind_shape=None,
                                 val_shape=None,
                                 shape_shape=None):
    return sparse_tensor.SparseTensor(
        array_ops.placeholder(
            dtypes.int64, shape=ind_shape),
        array_ops.placeholder(
            dtypes.float32, shape=val_shape),
        array_ops.placeholder(
            dtypes.int64, shape=shape_shape))

  def _SparseTensorValue_3x3(self):
    # [    1]
    # [2    ]
    # [3   4]
    ind = np.array([[0, 2], [1, 0], [2, 0], [2, 2]])
    val = np.array([1, 2, 3, 4])
    shape = np.array([3, 3])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.float32), np.array(shape, np.int64))

  def _SparseTensor_3x3(self):
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_3x3())

  def _SparseTensorValue_3x5(self):
    # [         ]
    # [  1      ]
    # [2     1 0]
    ind = np.array([[1, 1], [2, 0], [2, 3], [2, 4]])
    val = np.array([1, 2, 1, 0])
    shape = np.array([3, 5])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.float32), np.array(shape, np.int64))

  def _SparseTensor_3x5(self):
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_3x5())

  def _SparseTensor_3x2(self):
    # [   ]
    # [1  ]
    # [2  ]
    ind = np.array([[1, 0], [2, 0]])
    val = np.array([1, 2])
    shape = np.array([3, 2])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x3(self):
    # [  1  ]
    # [1   2]
    ind = np.array([[0, 1], [1, 0], [1, 2]])
    val = np.array([1, 1, 2])
    shape = np.array([2, 3])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x3x4(self):
    ind = np.array([
        [0, 0, 1],
        [0, 1, 0], [0, 1, 2],
        [1, 0, 3],
        [1, 1, 1], [1, 1, 3],
        [1, 2, 2]])
    val = np.array([1, 10, 12, 103, 111, 113, 122])
    shape = np.array([2, 3, 4])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.float32),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_String3x3(self):
    # [    a]
    # [b    ]
    # [c   d]
    ind = np.array([[0, 2], [1, 0], [2, 0], [2, 2]])
    val = np.array(["a", "b", "c", "d"])
    shape = np.array([3, 3])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.string),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_String3x5(self):
    # [         ]
    # [  e      ]
    # [f     g h]
    ind = np.array([[1, 1], [2, 0], [2, 3], [2, 4]])
    val = np.array(["e", "f", "g", "h"])
    shape = np.array([3, 5])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.string),
        constant_op.constant(shape, dtypes.int64))

  def testConcat1(self):
    with self.session(use_gpu=False) as sess:
      # concat(A):
      # [    1]
      # [2    ]
      # [3   4]
      for sp_a in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
        # Note that we ignore concat_dim in this case since we short-circuit the
        # single-input case in python.
        for concat_dim in (-2000, 1, 2000):
          sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a])

          self.assertEqual(sp_concat.indices.get_shape(), [4, 2])
          self.assertEqual(sp_concat.values.get_shape(), [4])
          self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

          concat_out = self.evaluate(sp_concat)

          self.assertAllEqual(concat_out.indices,
                              [[0, 2], [1, 0], [2, 0], [2, 2]])
          self.assertAllEqual(concat_out.values, [1, 2, 3, 4])
          self.assertAllEqual(concat_out.dense_shape, [3, 3])

  def testConcat2(self):
    with self.session(use_gpu=False) as sess:
      # concat(A, B):
      # [    1          ]
      # [2       1      ]
      # [3   4 2     1 0]
      for sp_a in (self._SparseTensorValue_3x3(), self._SparseTensor_3x3()):
        for sp_b in (self._SparseTensorValue_3x5(), self._SparseTensor_3x5()):
          for concat_dim in (-1, 1):
            sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b])

            self.assertEqual(sp_concat.indices.get_shape(), [8, 2])
            self.assertEqual(sp_concat.values.get_shape(), [8])
            self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

            concat_out = self.evaluate(sp_concat)

            self.assertAllEqual(concat_out.indices, [[0, 2], [1, 0], [1, 4],
                                                     [2, 0], [2, 2], [2, 3],
                                                     [2, 6], [2, 7]])
            self.assertAllEqual(concat_out.values, [1, 2, 1, 3, 4, 2, 1, 0])
            self.assertAllEqual(concat_out.dense_shape, [3, 8])

  def testConcatDim0(self):
    with self.session(use_gpu=False) as sess:
      # concat(A, D):
      # [    1]
      # [2    ]
      # [3   4]
      # [  1  ]
      # [1   2]
      sp_a = self._SparseTensor_3x3()
      sp_d = self._SparseTensor_2x3()

      for concat_dim in (-2, 0):
        sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_d])

        self.assertEqual(sp_concat.indices.get_shape(), [7, 2])
        self.assertEqual(sp_concat.values.get_shape(), [7])
        self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

        concat_out = self.evaluate(sp_concat)

        self.assertAllEqual(
            concat_out.indices,
            [[0, 2], [1, 0], [2, 0], [2, 2], [3, 1], [4, 0], [4, 2]])
        self.assertAllEqual(concat_out.values, np.array([1, 2, 3, 4, 1, 1, 2]))
        self.assertAllEqual(concat_out.dense_shape, np.array([5, 3]))

  def testConcat3(self):
    with self.session(use_gpu=False) as sess:
      # concat(A, B, C):
      # [    1              ]
      # [2       1       1  ]
      # [3   4 2     1 0 2  ]
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x5()
      sp_c = self._SparseTensor_3x2()

      for concat_dim in (-1, 1):
        sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b, sp_c])

        self.assertEqual(sp_concat.indices.get_shape(), [10, 2])
        self.assertEqual(sp_concat.values.get_shape(), [10])
        self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

        concat_out = self.evaluate(sp_concat)

        self.assertAllEqual(concat_out.indices, [[0, 2], [1, 0], [1, 4], [1, 8],
                                                 [2, 0], [2, 2], [2, 3], [2, 6],
                                                 [2, 7], [2, 8]])
        self.assertAllEqual(concat_out.values, [1, 2, 1, 1, 3, 4, 2, 1, 0, 2])
        self.assertAllEqual(concat_out.dense_shape, [3, 10])

  def testConcatNonNumeric(self):
    with self.session(use_gpu=False) as sess:
      # concat(A, B):
      # [    a          ]
      # [b       e      ]
      # [c   d f     g h]
      sp_a = self._SparseTensor_String3x3()
      sp_b = self._SparseTensor_String3x5()

      for concat_dim in (-1, 1):
        sp_concat = sparse_ops.sparse_concat(concat_dim, [sp_a, sp_b])

        self.assertEqual(sp_concat.indices.get_shape(), [8, 2])
        self.assertEqual(sp_concat.values.get_shape(), [8])
        self.assertEqual(sp_concat.dense_shape.get_shape(), [2])

        concat_out = self.evaluate(sp_concat)

        self.assertAllEqual(
            concat_out.indices,
            [[0, 2], [1, 0], [1, 4], [2, 0], [2, 2], [2, 3], [2, 6], [2, 7]])
        self.assertAllEqual(concat_out.values,
                            [b"a", b"b", b"e", b"c", b"d", b"f", b"g", b"h"])
        self.assertAllEqual(concat_out.dense_shape, [3, 8])

  def testMismatchedRank(self):
    with self.session(use_gpu=False):
      sp_a = self._SparseTensor_3x3()
      sp_e = self._SparseTensor_2x3x4()

      # Rank mismatches can be caught at shape-inference time
      for concat_dim in (-1, 1):
        with self.assertRaises(ValueError):
          sparse_ops.sparse_concat(concat_dim, [sp_a, sp_e])

  def testMismatchedRankExpandNonconcatDim(self):
    with self.session(use_gpu=False):
      sp_a = self._SparseTensor_3x3()
      sp_e = self._SparseTensor_2x3x4()

      # Rank mismatches should be caught at shape-inference time, even for
      # expand_nonconcat_dim=True.
      for concat_dim in (-1, 1):
        with self.assertRaises(ValueError):
          sparse_ops.sparse_concat(
              concat_dim, [sp_a, sp_e], expand_nonconcat_dim=True)

  def testMismatchedShapes(self):
    with self.session(use_gpu=False) as sess:
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x5()
      sp_c = self._SparseTensor_3x2()
      sp_d = self._SparseTensor_2x3()
      for concat_dim in (-1, 1):
        sp_concat = sparse_ops.sparse_concat(concat_dim,
                                             [sp_a, sp_b, sp_c, sp_d])

        # Shape mismatches can only be caught when the op is run
        with self.assertRaisesOpError("Input shapes must match"):
          self.evaluate(sp_concat)

  def testMismatchedShapesExpandNonconcatDim(self):
    with self.session(use_gpu=False) as sess:
      sp_a = self._SparseTensor_3x3()
      sp_b = self._SparseTensor_3x5()
      sp_c = self._SparseTensor_3x2()
      sp_d = self._SparseTensor_2x3()
      for concat_dim0 in (-2, 0):
        for concat_dim1 in (-1, 1):
          sp_concat_dim0 = sparse_ops.sparse_concat(
              concat_dim0, [sp_a, sp_b, sp_c, sp_d], expand_nonconcat_dim=True)
          sp_concat_dim1 = sparse_ops.sparse_concat(
              concat_dim1, [sp_a, sp_b, sp_c, sp_d], expand_nonconcat_dim=True)

          sp_concat_dim0_out = self.evaluate(sp_concat_dim0)
          sp_concat_dim1_out = self.evaluate(sp_concat_dim1)

          self.assertAllEqual(sp_concat_dim0_out.indices,
                              [[0, 2], [1, 0], [2, 0], [2, 2], [4, 1], [5, 0],
                               [5, 3], [5, 4], [7, 0], [8, 0], [9, 1], [10, 0],
                               [10, 2]])
          self.assertAllEqual(sp_concat_dim0_out.values,
                              [1, 2, 3, 4, 1, 2, 1, 0, 1, 2, 1, 1, 2])
          self.assertAllEqual(sp_concat_dim0_out.dense_shape, [11, 5])

          self.assertAllEqual(sp_concat_dim1_out.indices,
                              [[0, 2], [0, 11], [1, 0], [1, 4], [1, 8], [1, 10],
                               [1, 12], [2, 0], [2, 2], [2, 3], [2, 6], [2, 7],
                               [2, 8]])
          self.assertAllEqual(sp_concat_dim1_out.values,
                              [1, 1, 2, 1, 1, 1, 2, 3, 4, 2, 1, 0, 2])
          self.assertAllEqual(sp_concat_dim1_out.dense_shape, [3, 13])

  def testShapeInferenceUnknownShapes(self):
    with self.session(use_gpu=False):
      sp_inputs = [
          self._SparseTensor_UnknownShape(),
          self._SparseTensor_UnknownShape(val_shape=[3]),
          self._SparseTensor_UnknownShape(ind_shape=[1, 3]),
          self._SparseTensor_UnknownShape(shape_shape=[3])
      ]

      for concat_dim in (-2, 0):
        sp_concat = sparse_ops.sparse_concat(concat_dim, sp_inputs)

        self.assertEqual(sp_concat.indices.get_shape().as_list(), [None, 3])
        self.assertEqual(sp_concat.values.get_shape().as_list(), [None])
        self.assertEqual(sp_concat.dense_shape.get_shape(), [3])


if __name__ == "__main__":
  test.main()
