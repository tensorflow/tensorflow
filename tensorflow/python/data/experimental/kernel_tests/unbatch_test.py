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
"""Tests for `tf.data.experimental.unbatch()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class UnbatchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @test_util.run_deprecated_v1
  def testUnbatchWithUnknownRankInput(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    dataset = dataset_ops.Dataset.from_tensors(placeholder).apply(
        batching.unbatch())
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_elem = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(iterator.initializer, feed_dict={placeholder: [0, 1, 2, 3]})
      for i in range(4):
        self.assertEqual(i, self.evaluate(next_elem))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_elem)

  @test_util.run_deprecated_v1
  def testUnbatchScalarDataset(self):
    data = tuple([math_ops.range(10) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = (dtypes.int32,) * 3
    data = data.batch(2)
    self.assertEqual(expected_types, data.output_types)
    data = data.apply(batching.unbatch())
    self.assertEqual(expected_types, data.output_types)

    iterator = dataset_ops.make_one_shot_iterator(data)
    op = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual((i,) * 3, self.evaluate(op))

      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(op)

  @test_util.run_deprecated_v1
  def testUnbatchDatasetWithStrings(self):
    data = tuple([math_ops.range(10) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    data = data.map(lambda x, y, z: (x, string_ops.as_string(y), z))
    expected_types = (dtypes.int32, dtypes.string, dtypes.int32)
    data = data.batch(2)
    self.assertEqual(expected_types, data.output_types)
    data = data.apply(batching.unbatch())
    self.assertEqual(expected_types, data.output_types)

    iterator = dataset_ops.make_one_shot_iterator(data)
    op = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual((i, compat.as_bytes(str(i)), i), self.evaluate(op))

      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(op)

  @test_util.run_deprecated_v1
  def testUnbatchDatasetWithSparseTensor(self):
    st = sparse_tensor.SparseTensorValue(
        indices=[[i, i] for i in range(10)],
        values=list(range(10)),
        dense_shape=[10, 10])
    data = dataset_ops.Dataset.from_tensors(st)
    data = data.apply(batching.unbatch())
    data = data.batch(5)
    data = data.apply(batching.unbatch())
    iterator = dataset_ops.make_one_shot_iterator(data)
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        st_row = self.evaluate(next_element)
        self.assertEqual([i], st_row.indices)
        self.assertEqual([i], st_row.values)
        self.assertEqual([10], st_row.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @test_util.run_deprecated_v1
  def testUnbatchDatasetWithDenseAndSparseTensor(self):
    st = sparse_tensor.SparseTensorValue(
        indices=[[i, i] for i in range(10)],
        values=list(range(10)),
        dense_shape=[10, 10])
    data = dataset_ops.Dataset.from_tensors((list(range(10)), st))
    data = data.apply(batching.unbatch())
    data = data.batch(5)
    data = data.apply(batching.unbatch())
    iterator = dataset_ops.make_one_shot_iterator(data)
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        dense_elem, st_row = self.evaluate(next_element)
        self.assertEqual(i, dense_elem)
        self.assertEqual([i], st_row.indices)
        self.assertEqual([i], st_row.values)
        self.assertEqual([10], st_row.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @test_util.run_deprecated_v1
  def testUnbatchSingleElementTupleDataset(self):
    data = tuple([(math_ops.range(10),) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = ((dtypes.int32,),) * 3
    data = data.batch(2)
    self.assertEqual(expected_types, data.output_types)
    data = data.apply(batching.unbatch())
    self.assertEqual(expected_types, data.output_types)

    iterator = dataset_ops.make_one_shot_iterator(data)
    op = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual(((i,),) * 3, self.evaluate(op))

      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(op)

  @test_util.run_deprecated_v1
  def testUnbatchMultiElementTupleDataset(self):
    data = tuple([(math_ops.range(10 * i, 10 * i + 10),
                   array_ops.fill([10], "hi")) for i in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = ((dtypes.int32, dtypes.string),) * 3
    data = data.batch(2)
    self.assertAllEqual(expected_types, data.output_types)
    data = data.apply(batching.unbatch())
    self.assertAllEqual(expected_types, data.output_types)

    iterator = dataset_ops.make_one_shot_iterator(data)
    op = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual(((i, b"hi"), (10 + i, b"hi"), (20 + i, b"hi")),
                         self.evaluate(op))

      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(op)

  @test_util.run_deprecated_v1
  def testUnbatchEmpty(self):
    data = dataset_ops.Dataset.from_tensors(
        (constant_op.constant([]), constant_op.constant([], shape=[0, 4]),
         constant_op.constant([], shape=[0, 4, 0])))
    data = data.apply(batching.unbatch())
    iterator = dataset_ops.make_one_shot_iterator(data)
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  def testUnbatchStaticShapeMismatch(self):
    data = dataset_ops.Dataset.from_tensors((np.arange(7), np.arange(8),
                                             np.arange(9)))
    with self.assertRaises(ValueError):
      data.apply(batching.unbatch())

  @test_util.run_deprecated_v1
  def testUnbatchDynamicShapeMismatch(self):
    ph1 = array_ops.placeholder(dtypes.int32, shape=[None])
    ph2 = array_ops.placeholder(dtypes.int32, shape=None)
    data = dataset_ops.Dataset.from_tensors((ph1, ph2))
    data = data.apply(batching.unbatch())
    iterator = dataset_ops.make_initializable_iterator(data)
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      # Mismatch in the 0th dimension.
      sess.run(
          iterator.initializer,
          feed_dict={
              ph1: np.arange(7).astype(np.int32),
              ph2: np.arange(8).astype(np.int32)
          })
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(next_element)

      # No 0th dimension (i.e. scalar value) for one component.
      sess.run(
          iterator.initializer,
          feed_dict={
              ph1: np.arange(7).astype(np.int32),
              ph2: 7
          })
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(next_element)


if __name__ == "__main__":
  test.main()
