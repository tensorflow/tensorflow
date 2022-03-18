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
"""Tests for RaggedTensor.from_sparse."""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorToSparseOpTest(test_util.TensorFlowTestCase):

  def testDocStringExample(self):
    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [0, 2], [1, 0], [3, 0]],
        values=[1, 2, 3, 4, 5],
        dense_shape=[4, 3])
    rt = RaggedTensor.from_sparse(st)

    self.assertAllEqual(rt, [[1, 2, 3], [4], [], [5]])

  def testEmpty(self):
    st = sparse_tensor.SparseTensor(
        indices=array_ops.zeros([0, 2], dtype=dtypes.int64),
        values=[],
        dense_shape=[4, 3])
    rt = RaggedTensor.from_sparse(st)

    self.assertAllEqual(rt, [[], [], [], []])

  def testBadSparseTensorRank(self):
    st1 = sparse_tensor.SparseTensor(indices=[[0]], values=[0], dense_shape=[3])
    self.assertRaisesRegex(ValueError, r'rank\(st_input\) must be at least 2',
                           RaggedTensor.from_sparse, st1)

    if not context.executing_eagerly():
      st3 = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=[0],
          dense_shape=array_ops.placeholder(dtypes.int64))
      self.assertRaisesRegex(ValueError, r'rank\(st_input\) must be at least 2',
                             RaggedTensor.from_sparse, st3)

  def testGoodPartialSparseTensorRank(self):
    if not context.executing_eagerly():
      st1 = sparse_tensor.SparseTensor(
          indices=[[0, 0]],
          values=[0],
          dense_shape=array_ops.placeholder(dtypes.int64))
      st2 = sparse_tensor.SparseTensor(
          indices=array_ops.placeholder(dtypes.int64),
          values=[0],
          dense_shape=[4, 3])

      # Shouldn't throw ValueError
      RaggedTensor.from_sparse(st1)
      RaggedTensor.from_sparse(st2)

  def testNonRaggedSparseTensor(self):
    # "index_suffix" means the value of the innermost dimension of the index
    # (i.e., indices[i][-1]).
    # See comments in _assert_sparse_indices_are_ragged_right() for more
    # details/background.

    # index_suffix of first index is not zero.
    st1 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [0, 2], [2, 0]], values=[1, 2, 3], dense_shape=[3, 3])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'.*SparseTensor is not right-ragged'):
      self.evaluate(RaggedTensor.from_sparse(st1))
    # index_suffix of an index that starts a new row is not zero.
    st2 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [2, 1]], values=[1, 2, 3], dense_shape=[3, 3])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'.*SparseTensor is not right-ragged'):
      self.evaluate(RaggedTensor.from_sparse(st2))
    # index_suffix of an index that continues a row skips a cell.
    st3 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [0, 1], [0, 3]], values=[1, 2, 3], dense_shape=[3, 3])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r'.*SparseTensor is not right-ragged'):
      self.evaluate(RaggedTensor.from_sparse(st3))

  def testGoodMultiDimensionalSparse(self):
    st1 = sparse_tensor.SparseTensor(
      indices=[
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 1, 0],
        [1, 0, 0]
      ],
      values=[1, 2, 3, 4, 5],
      dense_shape=[2, 1, 3]
    )
    
    st2 = sparse_tensor.SparseTensor(
      indices=[
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 2],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
      ],
      values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      dense_shape=[2, 2, 1, 3]
    )
    res1 = RaggedTensor.from_sparse(st1)
    res2 = RaggedTensor.from_sparse(st2)

    self.assertAllEqual(res1.to_sparse().indices, st1.indices)
    self.assertAllEqual(res2.to_sparse().indices, st2.indices)
    self.assertAllEqual(res1.flat_values, st1.values)
    self.assertAllEqual(res2.flat_values, st2.values)


if __name__ == '__main__':
  googletest.main()
