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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.data.python.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class DatasetConstructorTest(test.TestCase):

  def testTensorDataset(self):
    """Test an dataset that represents a single tuple of tensors."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))

    iterator = (dataset_ops.Dataset.from_tensors(components)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape for c in components],
                     [t.shape for t in get_next])

    with self.test_session() as sess:
      sess.run(init_op)
      results = sess.run(get_next)
      for component, result_component in zip(components, results):
        self.assertAllEqual(component, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testTensorSliceDataset(self):
    """Test an dataset that represents the slices from a tuple of tensors."""
    components = (
        np.tile(np.array([[1], [2], [3], [4]]), 20), np.tile(
            np.array([[12], [13], [14], [15]]), 22),
        np.array([37.0, 38.0, 39.0, 40.0])
    )

    iterator = (dataset_ops.Dataset.from_tensor_slices(components)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape[1:] for c in components],
                     [t.shape for t in get_next])

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(4):
        results = sess.run(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component[i], result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testTensorSliceDatasetWithDict(self):
    components = {"foo": [1, 2, 3], "bar": [[4.0], [5.0], [6.0]]}
    iterator = (dataset_ops.Dataset.from_tensor_slices(components)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual(dtypes.int32, iterator.output_types["foo"])
    self.assertEqual(dtypes.float32, iterator.output_types["bar"])
    self.assertEqual((), iterator.output_shapes["foo"])
    self.assertEqual((1,), iterator.output_shapes["bar"])

    with self.test_session() as sess:
      sess.run(init_op)
      for i in range(3):
        results = sess.run(get_next)
        self.assertEqual(components["foo"][i], results["foo"])
        self.assertEqual(components["bar"][i], results["bar"])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testSparseTensorSliceDataset(self):
    """Test a dataset based on slices of a `tf.SparseTensor`."""
    st = array_ops.sparse_placeholder(dtypes.float64)
    iterator = (dataset_ops.Dataset.from_sparse_tensor_slices(st)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = sparse_tensor.SparseTensor(*iterator.get_next())

    with self.test_session() as sess:
      slices = [[1., 2., 3.], [1.], [1.], [1., 2.], [], [1., 2.], [], [], []]

      # Test with sparse tensor in the appropriate order.
      indices = np.array(
          [[i, j] for i in range(len(slices)) for j in range(len(slices[i]))])
      values = np.array([val for s in slices for val in s])
      dense_shape = np.array([len(slices), max(len(s) for s in slices) + 1])
      sparse_feed = sparse_tensor.SparseTensorValue(indices, values,
                                                    dense_shape)
      sess.run(init_op, feed_dict={st: sparse_feed})
      for i, s in enumerate(slices):
        results = sess.run(get_next)
        self.assertAllEqual(s, results.values)
        expected_indices = np.array(
            [[j] for j in range(len(slices[i]))]).reshape([-1, 1])
        self.assertAllEqual(expected_indices, results.indices)
        self.assertAllEqual(dense_shape[1:], results.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Test with sparse tensor in the reverse order, which is not
      # currently supported.
      reverse_order_indices = indices[::-1, :]
      reverse_order_values = values[::-1]
      sparse_feed = sparse_tensor.SparseTensorValue(
          reverse_order_indices, reverse_order_values, dense_shape)
      with self.assertRaises(errors.UnimplementedError):
        sess.run(init_op, feed_dict={st: sparse_feed})

      # Test with an empty sparse tensor.
      empty_indices = np.empty((0, 4), dtype=np.int64)
      empty_values = np.empty((0,), dtype=np.float64)
      empty_dense_shape = [0, 4, 37, 9]
      sparse_feed = sparse_tensor.SparseTensorValue(empty_indices, empty_values,
                                                    empty_dense_shape)
      sess.run(init_op, feed_dict={st: sparse_feed})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  # pylint: disable=g-long-lambda,unnecessary-lambda
  def testNestedStructure(self):
    components = (np.array([1, 2, 3]), (np.array([4., 5.]), np.array([6., 7.])),
                  np.array([8, 9, 10]))

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.shuffle(10, 10)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.repeat(-1)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.filter(lambda x, y, z: True)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.take(5)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([3], ([2], [2]), [3]), dataset.output_shapes)

    dataset = dataset.map(lambda x, y, z: ((x, z), (y[0], y[1])))
    self.assertEquals(((dtypes.int64, dtypes.int64),
                       (dtypes.float64, dtypes.float64)), dataset.output_types)
    self.assertEquals((([3], [3]), ([2], [2])), dataset.output_shapes)

    dataset = dataset.flat_map(
        lambda x, y: dataset_ops.Dataset.from_tensors(((x[0], x[1]),
                                                       (y[0], y[1])))
    )
    self.assertEquals(((dtypes.int64, dtypes.int64),
                       (dtypes.float64, dtypes.float64)), dataset.output_types)
    self.assertEquals((([3], [3]), ([2], [2])), dataset.output_shapes)

    dataset = dataset.batch(32)
    self.assertEquals(((dtypes.int64, dtypes.int64),
                       (dtypes.float64, dtypes.float64)), dataset.output_types)
    self.assertEquals((([None, 3], [None, 3]), ([None, 2], [None, 2])),
                      nest.pack_sequence_as(dataset.output_shapes, [
                          s.as_list()
                          for s in nest.flatten(dataset.output_shapes)
                      ]))

    iterator = dataset.make_one_shot_iterator()
    (w, x), (y, z) = iterator.get_next()
    self.assertEquals(dtypes.int64, w.dtype)
    self.assertEquals(dtypes.int64, x.dtype)
    self.assertEquals(dtypes.float64, y.dtype)
    self.assertEquals(dtypes.float64, z.dtype)
    self.assertEquals([None, 3], w.shape.as_list())
    self.assertEquals([None, 3], x.shape.as_list())
    self.assertEquals([None, 2], y.shape.as_list())
    self.assertEquals([None, 2], z.shape.as_list())

    iterator = dataset.make_initializable_iterator()
    (w, x), (y, z) = iterator.get_next()
    self.assertEquals(dtypes.int64, w.dtype)
    self.assertEquals(dtypes.int64, x.dtype)
    self.assertEquals(dtypes.float64, y.dtype)
    self.assertEquals(dtypes.float64, z.dtype)
    self.assertEquals([None, 3], w.shape.as_list())
    self.assertEquals([None, 3], x.shape.as_list())
    self.assertEquals([None, 2], y.shape.as_list())
    self.assertEquals([None, 2], z.shape.as_list())

    # Define a separate set of components with matching leading
    # dimension for the from-slices constructor.
    components_for_slices = (np.array([1, 2, 3]), (np.array(
        [4., 5., 6.]), np.array([7., 8., 9.])), np.array([10, 11, 12]))

    dataset = dataset_ops.Dataset.from_tensor_slices(components_for_slices)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([], ([], []), []), dataset.output_shapes)

  def testNonSequenceNestedStructure(self):
    components = np.array([1, 2, 3])

    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEquals(dtypes.int64, dataset.output_types)
    self.assertEquals([3], dataset.output_shapes)

    dataset = dataset.filter(
        lambda x: math_ops.reduce_all(math_ops.equal(x, components)))
    self.assertEquals(dtypes.int64, dataset.output_types)
    self.assertEquals([3], dataset.output_shapes)

    dataset = dataset.map(lambda x: array_ops.stack([x, x]))
    self.assertEquals(dtypes.int64, dataset.output_types)
    self.assertEquals([2, 3], dataset.output_shapes)

    dataset = dataset.flat_map(
        lambda x: dataset_ops.Dataset.from_tensor_slices(x))
    self.assertEquals(dtypes.int64, dataset.output_types)
    self.assertEquals([3], dataset.output_shapes)

    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()
    self.assertEquals(dtypes.int64, get_next.dtype)
    self.assertEquals([3], get_next.shape)


if __name__ == "__main__":
  test.main()
