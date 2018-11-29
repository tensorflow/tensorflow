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
"""Tests for ragged.to_sparse op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ragged
from tensorflow.python.platform import googletest


class RaggedTensorToSparseOpTest(test_util.TensorFlowTestCase):

  def testDocStringExample(self):
    rt = ragged.constant([[1, 2, 3], [4], [], [5, 6]])
    st = ragged.to_sparse(rt)
    expected = ('SparseTensorValue(indices='
                'array([[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [3, 1]]), '
                'values=array([1, 2, 3, 4, 5, 6], dtype=int32), '
                'dense_shape=array([4, 3]))')
    with self.test_session():
      self.assertEqual(' '.join(repr(st.eval()).split()), expected)

  def test2DRaggedTensorWithOneRaggedDimension(self):
    rt = ragged.constant([['a', 'b'], ['c', 'd', 'e'], ['f'], [], ['g']])
    with self.test_session():
      st = ragged.to_sparse(rt).eval()
      self.assertAllEqual(
          st.indices, [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0], [4, 0]])
      self.assertAllEqual(st.values, b'a b c d e f g'.split())
      self.assertAllEqual(st.dense_shape, [5, 3])

  def test3DRaggedTensorWithOneRaggedDimension(self):
    rt = ragged.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]],
                          [[11, 12]], [], [[13, 14]]],
                         ragged_rank=1)
    with self.test_session():
      st = ragged.to_sparse(rt).eval()
      self.assertAllEqual(
          st.indices, [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0],
                       [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1],
                       [2, 0, 0], [2, 0, 1], [4, 0, 0], [4, 0, 1]])
      self.assertAllEqual(st.values,
                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
      self.assertAllEqual(st.dense_shape, [5, 3, 2])

  def test4DRaggedTensorWithOneRaggedDimension(self):
    rt = ragged.constant(
        [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [], [[[9, 10], [11, 12]]]],
        ragged_rank=1)
    with self.test_session():
      st = ragged.to_sparse(rt).eval()
      self.assertAllEqual(st.values, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
      self.assertAllEqual(
          st.indices,
          [
              [0, 0, 0, 0],  # index for value=1
              [0, 0, 0, 1],  # index for value=2
              [0, 0, 1, 0],  # index for value=3
              [0, 0, 1, 1],  # index for value=4
              [0, 1, 0, 0],  # index for value=5
              [0, 1, 0, 1],  # index for value=6
              [0, 1, 1, 0],  # index for value=7
              [0, 1, 1, 1],  # index for value=8
              [2, 0, 0, 0],  # index for value=9
              [2, 0, 0, 1],  # index for value=10
              [2, 0, 1, 0],  # index for value=11
              [2, 0, 1, 1],  # index for value=12
          ])
      self.assertAllEqual(st.dense_shape, [3, 2, 2, 2])

  def test4DRaggedTensorWithTwoRaggedDimensions(self):
    rt = ragged.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8], [9, 10]]],
                          [[[11, 12]], [], [[13, 14]]], []],
                         ragged_rank=2)
    with self.test_session():
      st = ragged.to_sparse(rt).eval()
      self.assertAllEqual(
          st.indices,
          [
              [0, 0, 0, 0],  # index for value=1
              [0, 0, 0, 1],  # index for value=2
              [0, 0, 1, 0],  # index for value=3
              [0, 0, 1, 1],  # index for value=4
              [0, 1, 0, 0],  # index for value=5
              [0, 1, 0, 1],  # index for value=6
              [0, 1, 1, 0],  # index for value=7
              [0, 1, 1, 1],  # index for value=8
              [0, 1, 2, 0],  # index for value=9
              [0, 1, 2, 1],  # index for value=10
              [1, 0, 0, 0],  # index for value=11
              [1, 0, 0, 1],  # index for value=12
              [1, 2, 0, 0],  # index for value=13
              [1, 2, 0, 1],  # index for value=14
          ])
      self.assertAllEqual(st.values,
                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
      self.assertAllEqual(st.dense_shape, [3, 3, 3, 2])

  def testShape(self):
    rt = ragged.constant([[1, 2], [3, 4, 5], [6], [], [7]])
    st = ragged.to_sparse(rt)
    self.assertEqual(st.indices.shape.as_list(), [7, 2])
    self.assertEqual(st.values.shape.as_list(), [7])
    self.assertEqual(st.dense_shape.shape.as_list(), [2])

    rt = ragged.constant([[[1, 2]], [], [[3, 4]], []], ragged_rank=1)
    st = ragged.to_sparse(rt)
    self.assertEqual(st.indices.shape.as_list(), [4, 3])
    self.assertEqual(st.values.shape.as_list(), [4])
    self.assertEqual(st.dense_shape.shape.as_list(), [3])

    rt = ragged.constant([[[1], [2, 3, 4, 5, 6, 7]], [[]]])
    st = ragged.to_sparse(rt)
    self.assertEqual(st.indices.shape.as_list(), [7, 3])
    self.assertEqual(st.values.shape.as_list(), [7])
    self.assertEqual(st.dense_shape.shape.as_list(), [3])

  def testKernelErrors(self):
    # An empty vector, defined using a placeholder to ensure that we can't
    # determine that it's invalid at graph-construction time.
    empty_vector = array_ops.placeholder_with_default(
        array_ops.zeros([0], dtypes.int64), shape=None)

    bad_rt1 = ragged.from_row_splits(row_splits=[2, 3], values=[1, 2, 3])
    with self.test_session():
      bad_split0_error = r'First value of ragged splits must be 0.*'
      self.assertRaisesRegexp(errors.InvalidArgumentError, bad_split0_error,
                              ragged.to_sparse(bad_rt1).eval)

    bad_rt2 = ragged.from_row_splits(row_splits=[0, 5], values=empty_vector)
    bad_rt3 = ragged.from_row_splits(
        row_splits=[0, 1],
        values=ragged.from_row_splits(row_splits=[0, 5], values=empty_vector))
    with self.test_session():
      split_mismatch1_error = r'Final value of ragged splits must match.*'
      for rt in [bad_rt2, bad_rt3]:
        self.assertRaisesRegexp(errors.InvalidArgumentError,
                                split_mismatch1_error,
                                ragged.to_sparse(rt).eval)

    bad_rt4 = ragged.from_row_splits(
        row_splits=[0, 5],
        values=ragged.from_row_splits(row_splits=[0], values=empty_vector))
    with self.test_session():
      split_mismatch2_error = r'Final value of ragged splits must match.*'
      self.assertRaisesRegexp(errors.InvalidArgumentError,
                              split_mismatch2_error,
                              ragged.to_sparse(bad_rt4).eval)

    bad_rt5 = ragged.from_row_splits(row_splits=empty_vector, values=[])
    with self.test_session():
      empty_splits_error = (r'ragged splits may not be empty.*')
      self.assertRaisesRegexp(errors.InvalidArgumentError, empty_splits_error,
                              ragged.to_sparse(bad_rt5).eval)

  def testGradient(self):
    # rt1.shape == rt2.shape == [2, (D2), (D3), 2].
    rt1 = ragged.constant([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]],
                          ragged_rank=2)
    rt2 = ragged.constant([[[[9.0, 8.0], [7.0, 6.0]], [[5.0, 4.0]]]],
                          ragged_rank=2)
    rt = ragged.map_inner_values(math_ops.add, rt1, rt2 * 2.0)
    st = ragged.to_sparse(rt)

    g1, g2 = gradients_impl.gradients(st.values, [rt1.inner_values,
                                                  rt2.inner_values])
    print(g1, g2)
    with self.test_session():
      self.assertEqual(g1.eval().tolist(), [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
      self.assertEqual(g2.eval().tolist(), [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])


if __name__ == '__main__':
  googletest.main()
