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
"""Tests for ragged.from_sparse."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


class RaggedTensorToSparseOpTest(test_util.TensorFlowTestCase):

  def testDocStringExample(self):
    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [0, 2], [1, 0], [3, 0]],
        values=[1, 2, 3, 4, 5],
        dense_shape=[4, 3])
    rt = ragged.from_sparse(st)

    with self.test_session():
      self.assertEqual(rt.eval().tolist(), [[1, 2, 3], [4], [], [5]])

  def testEmpty(self):
    st = sparse_tensor.SparseTensor(
        indices=array_ops.zeros([0, 2], dtype=dtypes.int64),
        values=[],
        dense_shape=[4, 3])
    rt = ragged.from_sparse(st)

    with self.test_session():
      self.assertEqual(rt.eval().tolist(), [[], [], [], []])

  def testBadSparseTensorRank(self):
    st1 = sparse_tensor.SparseTensor(indices=[[0]], values=[0], dense_shape=[3])
    st2 = sparse_tensor.SparseTensor(
        indices=[[0, 0, 0]], values=[0], dense_shape=[3, 3, 3])
    st3 = sparse_tensor.SparseTensor(
        indices=array_ops.placeholder(dtypes.int64),
        values=[0],
        dense_shape=array_ops.placeholder(dtypes.int64))
    self.assertRaisesRegexp(ValueError, r'rank\(st_input\) must be 2',
                            ragged.from_sparse, st1)
    self.assertRaisesRegexp(ValueError, r'rank\(st_input\) must be 2',
                            ragged.from_sparse, st2)
    self.assertRaisesRegexp(ValueError, r'rank\(st_input\) must be 2',
                            ragged.from_sparse, st3)

  def testNonRaggedSparseTensor(self):
    # "index_suffix" means the value of the innermost dimension of the index
    # (i.e., indices[i][-1]).
    # See comments in _assert_sparse_indices_are_ragged_right() for more
    # details/background.

    # index_suffix of first index is not zero.
    st1 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [0, 2], [2, 0]], values=[1, 2, 3], dense_shape=[3, 3])
    # index_suffix of an index that starts a new row is not zero.
    st2 = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [2, 1]], values=[1, 2, 3], dense_shape=[3, 3])
    # index_suffix of an index that continues a row skips a cell.
    st3 = sparse_tensor.SparseTensor(
        indices=[[0, 1], [0, 1], [0, 3]], values=[1, 2, 3], dense_shape=[3, 3])
    rt1 = ragged.from_sparse(st1)
    rt2 = ragged.from_sparse(st2)
    rt3 = ragged.from_sparse(st3)
    with self.test_session():
      self.assertRaisesRegexp(errors.InvalidArgumentError,
                              r'.*SparseTensor is not right-ragged', rt1.eval)
      self.assertRaisesRegexp(errors.InvalidArgumentError,
                              r'.*SparseTensor is not right-ragged', rt2.eval)
      self.assertRaisesRegexp(errors.InvalidArgumentError,
                              r'.*SparseTensor is not right-ragged', rt3.eval)


if __name__ == '__main__':
  googletest.main()
