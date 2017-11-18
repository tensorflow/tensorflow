# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utilities working with arbitrarily nested structures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import test


class SparseTest(test.TestCase):

  def testAnySparse(self):
    test_cases = (
        ((), False),
        ((None), False),
        ((dtypes.string), False),
        ((None, -1, dtypes.string), False),
        ((sparse.SparseType(dtypes.string)), True),
        ((None, sparse.SparseType(dtypes.string)), True),
        ((sparse.SparseType(dtypes.string), dtypes.string), True),
        ((((sparse.SparseType(dtypes.string)))), True)
    )
    for test_case in test_cases:
      self.assertEqual(sparse.any_sparse(test_case[0]), test_case[1])

  def assertSparseValuesEqual(self, a, b):
    if not isinstance(a, sparse_tensor.SparseTensor):
      self.assertFalse(isinstance(b, sparse_tensor.SparseTensor))
      self.assertEqual(a, b)
      return
    self.assertTrue(isinstance(b, sparse_tensor.SparseTensor))
    with self.test_session():
      self.assertAllEqual(a.eval().indices, b.eval().indices)
      self.assertAllEqual(a.eval().values, b.eval().values)
      self.assertAllEqual(a.eval().dense_shape, b.eval().dense_shape)

  def testSerializeDeserialize(self):
    test_cases = (
        (),
        sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
        sparse_tensor.SparseTensor(
            indices=[[3, 4]], values=[-1], dense_shape=[4, 5]),
        sparse_tensor.SparseTensor(
            indices=[[0, 0], [3, 4]], values=[1, -1], dense_shape=[4, 5]),
        (sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1])),
        (sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]), ()),
        ((), sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1])),
    )
    for expected in test_cases:
      actual = sparse.deserialize_sparse_tensors(
          sparse.serialize_sparse_tensors(expected),
          sparse.get_sparse_types(expected))
      nest.assert_same_structure(expected, actual)
      for a, e in zip(nest.flatten(actual), nest.flatten(expected)):
        self.assertSparseValuesEqual(a, e)

  def testGetSparseTypes(self):
    s = sparse_tensor.SparseTensor(
        indices=[[0, 0]], values=[1], dense_shape=[1, 1])
    t = sparse.SparseType(dtypes.int32)
    test_cases = (
        ((), ()),
        (s, t),
        ((s), (t)),
        ((s, ()), (t, ())),
        (((), s), ((), t)),
    )
    for test_case in test_cases:
      self.assertEqual(sparse.get_sparse_types(test_case[0]), test_case[1])

  def testWrapSparseTypes(self):
    c = constant_op.constant([1])
    d = dtypes.int32
    s = sparse_tensor.SparseTensor(
        indices=[[0, 0]], values=[1], dense_shape=[1, 1])
    t = sparse.SparseType(dtypes.int32)
    test_cases = (
        ((), ()),
        (s, t),
        (c, d),
        ((s), (t)),
        ((c), (d)),
        ((s, ()), (t, ())),
        (((), s), ((), t)),
        ((c, ()), (d, ())),
        (((), c), ((), d)),
        ((s, (), c), (t, (), d)),
        (((), s, ()), ((), t, ())),
        (((), c, ()), ((), d, ())),
    )
    for test_case in test_cases:
      self.assertEqual(
          sparse.wrap_sparse_types(test_case[0], sparse.get_sparse_types(
              test_case[0])), test_case[1])

  def testUnwrapSparseTypes(self):
    d = dtypes.string
    t = sparse.SparseType(dtypes.int32)
    test_cases = (
        ((), ()),
        (t, d),
        (d, d),
        ((t), (d)),
        ((d), (d)),
        ((t, ()), (d, ())),
        (((), t), ((), d)),
        ((d, ()), (d, ())),
        (((), d), ((), d)),
        ((t, (), d), (d, (), d)),
        (((), t, ()), ((), d, ())),
        (((), d, ()), ((), d, ())),
    )
    for test_case in test_cases:
      self.assertEqual(sparse.unwrap_sparse_types(test_case[0]), test_case[1])


if __name__ == "__main__":
  test.main()
