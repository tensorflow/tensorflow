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

import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class DatasetConstructorTest(test_base.DatasetTestBase):

  def testFromTensors(self):
    """Test a dataset that represents a single tuple of tensors."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))

    iterator = (dataset_ops.Dataset.from_tensors(components)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([c.shape for c in components],
                     [t.shape for t in get_next])

    with self.cached_session() as sess:
      sess.run(init_op)
      results = sess.run(get_next)
      for component, result_component in zip(components, results):
        self.assertAllEqual(component, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromTensorsSparse(self):
    """Test a dataset that represents a single tuple of tensors."""
    components = (sparse_tensor.SparseTensorValue(
        indices=np.array([[0]]),
        values=np.array([0]),
        dense_shape=np.array([1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1]]),
                      values=np.array([-1, 1]),
                      dense_shape=np.array([2, 2])))

    iterator = (
        dataset_ops.Dataset.from_tensors(components)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual(
        [tensor_shape.TensorShape(c.dense_shape) for c in components],
        [shape for shape in iterator.output_shapes])

    with self.cached_session() as sess:
      sess.run(init_op)
      results = sess.run(get_next)
      for component, result_component in zip(components, results):
        self.assertSparseValuesEqual(component, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromTensorsMixed(self):
    """Test an dataset that represents a single tuple of tensors."""
    components = (np.array(1), np.array([1, 2, 3]), np.array(37.0),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0]]),
                      values=np.array([0]),
                      dense_shape=np.array([1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1]]),
                      values=np.array([-1, 1]),
                      dense_shape=np.array([2, 2])))

    iterator = (
        dataset_ops.Dataset.from_tensors(components)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([
        tensor_shape.TensorShape(c.dense_shape)
        if sparse_tensor.is_sparse(c) else c.shape for c in components
    ], [shape for shape in iterator.output_shapes])

    with self.cached_session() as sess:
      sess.run(init_op)
      results = sess.run(get_next)
      for component, result_component in zip(components, results):
        if sparse_tensor.is_sparse(component):
          self.assertSparseValuesEqual(component, result_component)
        else:
          self.assertAllEqual(component, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromTensorSlices(self):
    """Test a dataset that represents the slices from a tuple of tensors."""
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

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(4):
        results = sess.run(get_next)
        for component, result_component in zip(components, results):
          self.assertAllEqual(component[i], result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromTensorSlicesSparse(self):
    """Test a dataset that represents the slices from a tuple of tensors."""
    components = (sparse_tensor.SparseTensorValue(
        indices=np.array([[0, 0], [1, 0], [2, 0]]),
        values=np.array([0, 0, 0]),
        dense_shape=np.array([3, 1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1], [2, 2]]),
                      values=np.array([1, 2, 3]),
                      dense_shape=np.array([3, 3])))

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual(
        [tensor_shape.TensorShape(c.dense_shape[1:]) for c in components],
        [shape for shape in iterator.output_shapes])

    with self.cached_session() as sess:
      sess.run(init_op)
      expected = [
          (sparse_tensor.SparseTensorValue(
              indices=np.array([[0]]),
              values=np.array([0]),
              dense_shape=np.array([1])),
           sparse_tensor.SparseTensorValue(
               indices=np.array([[0]]),
               values=np.array([1]),
               dense_shape=np.array([3]))),
          (sparse_tensor.SparseTensorValue(
              indices=np.array([[0]]),
              values=np.array([0]),
              dense_shape=np.array([1])),
           sparse_tensor.SparseTensorValue(
               indices=np.array([[1]]),
               values=np.array([2]),
               dense_shape=np.array([3]))),
          (sparse_tensor.SparseTensorValue(
              indices=np.array([[0]]),
              values=np.array([0]),
              dense_shape=np.array([1])),
           sparse_tensor.SparseTensorValue(
               indices=np.array([[2]]),
               values=np.array([3]),
               dense_shape=np.array([3]))),
      ]
      for i in range(3):
        results = sess.run(get_next)
        for component, result_component in zip(expected[i], results):
          self.assertSparseValuesEqual(component, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromTensorSlicesMixed(self):
    """Test a dataset that represents the slices from a tuple of tensors."""
    components = (np.tile(np.array([[1], [2], [3]]), 20),
                  np.tile(np.array([[12], [13], [14]]), 22),
                  np.array([37.0, 38.0, 39.0]),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 0], [2, 0]]),
                      values=np.array([0, 0, 0]),
                      dense_shape=np.array([3, 1])),
                  sparse_tensor.SparseTensorValue(
                      indices=np.array([[0, 0], [1, 1], [2, 2]]),
                      values=np.array([1, 2, 3]),
                      dense_shape=np.array([3, 3])))

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([
        tensor_shape.TensorShape(c.dense_shape[1:])
        if sparse_tensor.is_sparse(c) else c.shape[1:] for c in components
    ], [shape for shape in iterator.output_shapes])

    with self.cached_session() as sess:
      sess.run(init_op)
      expected = [
          (sparse_tensor.SparseTensorValue(
              indices=np.array([[0]]),
              values=np.array([0]),
              dense_shape=np.array([1])),
           sparse_tensor.SparseTensorValue(
               indices=np.array([[0]]),
               values=np.array([1]),
               dense_shape=np.array([3]))),
          (sparse_tensor.SparseTensorValue(
              indices=np.array([[0]]),
              values=np.array([0]),
              dense_shape=np.array([1])),
           sparse_tensor.SparseTensorValue(
               indices=np.array([[1]]),
               values=np.array([2]),
               dense_shape=np.array([3]))),
          (sparse_tensor.SparseTensorValue(
              indices=np.array([[0]]),
              values=np.array([0]),
              dense_shape=np.array([1])),
           sparse_tensor.SparseTensorValue(
               indices=np.array([[2]]),
               values=np.array([3]),
               dense_shape=np.array([3]))),
      ]
      for i in range(3):
        results = sess.run(get_next)
        for component, result_component in zip(
            (list(zip(*components[:3]))[i] + expected[i]), results):
          if sparse_tensor.is_sparse(component):
            self.assertSparseValuesEqual(component, result_component)
          else:
            self.assertAllEqual(component, result_component)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromTensorSlicesWithDict(self):
    components = {"foo": [1, 2, 3], "bar": [[4.0], [5.0], [6.0]]}
    iterator = (dataset_ops.Dataset.from_tensor_slices(components)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual(dtypes.int32, iterator.output_types["foo"])
    self.assertEqual(dtypes.float32, iterator.output_types["bar"])
    self.assertEqual((), iterator.output_shapes["foo"])
    self.assertEqual((1,), iterator.output_shapes["bar"])

    with self.cached_session() as sess:
      sess.run(init_op)
      for i in range(3):
        results = sess.run(get_next)
        self.assertEqual(components["foo"][i], results["foo"])
        self.assertEqual(components["bar"][i], results["bar"])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testFromSparseTensorSlices(self):
    """Test a dataset based on slices of a `tf.SparseTensor`."""
    st = array_ops.sparse_placeholder(dtypes.float64)
    iterator = (dataset_ops.Dataset.from_sparse_tensor_slices(st)
                .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = sparse_tensor.SparseTensor(*iterator.get_next())

    with self.cached_session() as sess:
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
    components = (np.array([1, 2, 3], dtype=np.int64),
                  (np.array([4., 5.]), np.array([6., 7.])),
                  np.array([8, 9, 10], dtype=np.int64))

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
    components_for_slices = (np.array([1, 2, 3], dtype=np.int64),
                             (np.array([4., 5., 6.]),
                              np.array([7., 8., 9.])),
                             np.array([10, 11, 12], dtype=np.int64))

    dataset = dataset_ops.Dataset.from_tensor_slices(components_for_slices)
    self.assertEquals((dtypes.int64, (dtypes.float64, dtypes.float64),
                       dtypes.int64), dataset.output_types)
    self.assertEquals(([], ([], []), []), dataset.output_shapes)

  def testNestedDict(self):
    components = {"a": {"aa": 1, "ab": [2.0, 2.0]}, "b": [3, 3, 3]}
    dataset = dataset_ops.Dataset.from_tensors(components)
    self.assertEquals(dtypes.int32, dataset.output_types["a"]["aa"])
    self.assertEquals(dtypes.float32, dataset.output_types["a"]["ab"])
    self.assertEquals(dtypes.int32, dataset.output_types["b"])
    self.assertEquals([], dataset.output_shapes["a"]["aa"])
    self.assertEquals([2], dataset.output_shapes["a"]["ab"])
    self.assertEquals([3], dataset.output_shapes["b"])

  def testNonSequenceNestedStructure(self):
    components = np.array([1, 2, 3], dtype=np.int64)

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

  def testSplitPipelineFailsWithPlacementError(self):
    with session.Session(
        target="",
        config=config_pb2.ConfigProto(device_count={"CPU": 2})) as sess:

      dataset = dataset_ops.Dataset.from_tensors(0)

      # Define a pipeline that attempts to use variables on two
      # different devices.
      #
      # Initialize the variables before creating to iterator, to avoid the
      # placement algorithm overriding the DT_RESOURCE colocation constraints.
      with ops.device("/cpu:0"):
        var_0 = resource_variable_ops.ResourceVariable(initial_value=0)
        dataset = dataset.map(lambda x: x + var_0.read_value())
      sess.run(var_0.initializer)

      with ops.device("/cpu:1"):
        var_1 = resource_variable_ops.ResourceVariable(initial_value=0)
        dataset = dataset.map(lambda x: x + var_1.read_value())
      sess.run(var_1.initializer)

      iterator = dataset.make_initializable_iterator()
      sess.run(iterator.initializer)

      with self.assertRaisesRegexp(
          errors.FailedPreconditionError,
          "Error while reading resource variable Variable"):
        sess.run(iterator.get_next())


class DatasetConstructorBenchmark(test.Benchmark):

  def benchmarkSliceRepeatBatch(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data)
        .repeat(num_epochs + 1).batch(batch_size))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with session.Session() as sess:
      sess.run(iterator.initializer)
      # Run one whole epoch to burn in the computation.
      for _ in range(input_size // batch_size):
        sess.run(next_element)
      deltas = []
      try:
        while True:
          start = time.time()
          sess.run(next_element)
          deltas.append(time.time() - start)
      except errors.OutOfRangeError:
        pass

    median_wall_time = np.median(deltas)
    print("Slice/repeat/batch with sess.run() input size: %d batch size: %d "
          "Median wall time per element: %f" % (input_size, batch_size,
                                                median_wall_time))
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name="benchmark_slice_repeat_batch_input_%d_batch_%d" % (input_size,
                                                                 batch_size))

  def benchmarkSliceRepeatBatchCallable(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data)
        .repeat(num_epochs + 1).batch(batch_size))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with session.Session() as sess:
      sess.run(iterator.initializer)
      get_next_element = sess.make_callable(next_element)
      # Run one whole epoch to burn in the computation.
      for _ in range(input_size // batch_size):
        get_next_element()
      deltas = []
      try:
        while True:
          start = time.time()
          get_next_element()
          deltas.append(time.time() - start)
      except errors.OutOfRangeError:
        pass

    median_wall_time = np.median(deltas)
    print(
        "Slice/repeat/batch with callable input size: %d batch size: %d Median"
        " wall time per element: %f" % (input_size, batch_size,
                                        median_wall_time))
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name="benchmark_slice_repeat_batch_callable_input_%d_batch_%d" %
        (input_size, batch_size))

  def benchmarkReshapeSliceRepeatCallable(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data.reshape(100, 100))
        .repeat(num_epochs + 1))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with session.Session() as sess:
      sess.run(iterator.initializer)
      get_next_element = sess.make_callable(next_element)
      # Run one whole epoch to burn in the computation.
      for _ in range(input_size // batch_size):
        get_next_element()
      deltas = []
      try:
        while True:
          start = time.time()
          get_next_element()
          deltas.append(time.time() - start)
      except errors.OutOfRangeError:
        pass

    median_wall_time = np.median(deltas)
    print("Reshape/slice/repeat with callable input size: %d batch size: %d "
          "Median wall time per element: %f" % (input_size, batch_size,
                                                median_wall_time))
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name="benchmark_reshape_slice_repeat_callable_input_%d_batch_%d" %
        (input_size, batch_size))

  def benchmarkSliceBatchCacheRepeatCallable(self):
    input_size = 10000
    batch_size = 100
    num_epochs = 100

    input_data = np.random.randn(input_size)

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(input_data).batch(batch_size)
        .cache().repeat(num_epochs + 1))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with session.Session() as sess:
      sess.run(iterator.initializer)
      get_next_element = sess.make_callable(next_element)
      # Run one whole epoch to burn in the computation.
      for _ in range(input_size // batch_size):
        get_next_element()
      deltas = []
      try:
        while True:
          start = time.time()
          get_next_element()
          deltas.append(time.time() - start)
      except errors.OutOfRangeError:
        pass

    median_wall_time = np.median(deltas)
    print(
        "Slice/batch/cache/repeat with callable input size: %d batch size: %d "
        "Median wall time per element: %f"
        % (input_size, batch_size, median_wall_time))
    self.report_benchmark(
        iters=len(deltas),
        wall_time=median_wall_time,
        name="benchmark_slice_batch_cache_repeat_callable_input_%d_batch_%d" %
        (input_size, batch_size))


if __name__ == "__main__":
  test.main()
