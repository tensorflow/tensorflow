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

import math
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class BatchDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  def testDenseToSparseBatchDataset(self):
    components = np.random.randint(12, size=(100,)).astype(np.int32)
    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components)
        .map(lambda x: array_ops.fill([x], x)).apply(
            batching.dense_to_sparse_batch(4,
                                           [12])).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)

      for start in range(0, len(components), 4):
        results = self.evaluate(get_next)
        self.assertAllEqual([[i, j]
                             for i, c in enumerate(components[start:start + 4])
                             for j in range(c)], results.indices)
        self.assertAllEqual(
            [c for c in components[start:start + 4] for _ in range(c)],
            results.values)
        self.assertAllEqual([min(4,
                                 len(components) - start), 12],
                            results.dense_shape)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testDenseToSparseBatchDatasetWithUnknownShape(self):
    components = np.random.randint(5, size=(40,)).astype(np.int32)
    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components).map(
            lambda x: array_ops.fill([x, x], x)).apply(
                batching.dense_to_sparse_batch(
                    4, [5, None])).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)

      for start in range(0, len(components), 4):
        results = self.evaluate(get_next)
        self.assertAllEqual([[i, j, z]
                             for i, c in enumerate(components[start:start + 4])
                             for j in range(c)
                             for z in range(c)], results.indices)
        self.assertAllEqual([
            c for c in components[start:start + 4] for _ in range(c)
            for _ in range(c)
        ], results.values)
        self.assertAllEqual([
            min(4,
                len(components) - start), 5,
            np.max(components[start:start + 4])
        ], results.dense_shape)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testDenseToSparseBatchDatasetWithInvalidShape(self):
    input_tensor = array_ops.constant([[1]])
    with self.assertRaisesRegexp(ValueError, "Dimension -2 must be >= 0"):
      dataset_ops.Dataset.from_tensors(input_tensor).apply(
          batching.dense_to_sparse_batch(4,
                                         [-2])).make_initializable_iterator()

  def testDenseToSparseBatchDatasetShapeErrors(self):
    input_tensor = array_ops.placeholder(dtypes.int32)
    iterator = (
        dataset_ops.Dataset.from_tensors(input_tensor).apply(
            batching.dense_to_sparse_batch(4,
                                           [12])).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      # Initialize with an input tensor of incompatible rank.
      sess.run(init_op, feed_dict={input_tensor: [[1]]})
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "incompatible with the row shape"):
        sess.run(get_next)

      # Initialize with an input tensor that is larger than `row_shape`.
      sess.run(init_op, feed_dict={input_tensor: range(13)})
      with self.assertRaisesRegexp(errors.DataLossError,
                                   "larger than the row shape"):
        sess.run(get_next)

  def testUnbatchWithUnknownRankInput(self):
    placeholder = array_ops.placeholder(dtypes.int32)
    dataset = dataset_ops.Dataset.from_tensors(placeholder).apply(
        batching.unbatch())
    iterator = dataset.make_initializable_iterator()
    next_elem = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(iterator.initializer, feed_dict={placeholder: [0, 1, 2, 3]})
      for i in range(4):
        self.assertEqual(i, self.evaluate(next_elem))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_elem)

  def testUnbatchScalarDataset(self):
    data = tuple([math_ops.range(10) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = (dtypes.int32,) * 3
    data = data.batch(2)
    self.assertEqual(expected_types, data.output_types)
    data = data.apply(batching.unbatch())
    self.assertEqual(expected_types, data.output_types)

    iterator = data.make_one_shot_iterator()
    op = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual((i,) * 3, self.evaluate(op))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(op)

  def testUnbatchDatasetWithStrings(self):
    data = tuple([math_ops.range(10) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    data = data.map(lambda x, y, z: (x, string_ops.as_string(y), z))
    expected_types = (dtypes.int32, dtypes.string, dtypes.int32)
    data = data.batch(2)
    self.assertEqual(expected_types, data.output_types)
    data = data.apply(batching.unbatch())
    self.assertEqual(expected_types, data.output_types)

    iterator = data.make_one_shot_iterator()
    op = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual((i, compat.as_bytes(str(i)), i), self.evaluate(op))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(op)

  def testUnbatchDatasetWithSparseTensor(self):
    st = sparse_tensor.SparseTensorValue(
        indices=[[i, i] for i in range(10)],
        values=list(range(10)),
        dense_shape=[10, 10])
    data = dataset_ops.Dataset.from_tensors(st)
    data = data.apply(batching.unbatch())
    data = data.batch(5)
    data = data.apply(batching.unbatch())
    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        st_row = self.evaluate(next_element)
        self.assertEqual([i], st_row.indices)
        self.assertEqual([i], st_row.values)
        self.assertEqual([10], st_row.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testUnbatchDatasetWithDenseAndSparseTensor(self):
    st = sparse_tensor.SparseTensorValue(
        indices=[[i, i] for i in range(10)],
        values=list(range(10)),
        dense_shape=[10, 10])
    data = dataset_ops.Dataset.from_tensors((list(range(10)), st))
    data = data.apply(batching.unbatch())
    data = data.batch(5)
    data = data.apply(batching.unbatch())
    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        dense_elem, st_row = self.evaluate(next_element)
        self.assertEqual(i, dense_elem)
        self.assertEqual([i], st_row.indices)
        self.assertEqual([i], st_row.values)
        self.assertEqual([10], st_row.dense_shape)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testUnbatchSingleElementTupleDataset(self):
    data = tuple([(math_ops.range(10),) for _ in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = ((dtypes.int32,),) * 3
    data = data.batch(2)
    self.assertEqual(expected_types, data.output_types)
    data = data.apply(batching.unbatch())
    self.assertEqual(expected_types, data.output_types)

    iterator = data.make_one_shot_iterator()
    op = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual(((i,),) * 3, self.evaluate(op))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(op)

  def testUnbatchMultiElementTupleDataset(self):
    data = tuple([(math_ops.range(10 * i, 10 * i + 10),
                   array_ops.fill([10], "hi")) for i in range(3)])
    data = dataset_ops.Dataset.from_tensor_slices(data)
    expected_types = ((dtypes.int32, dtypes.string),) * 3
    data = data.batch(2)
    self.assertAllEqual(expected_types, data.output_types)
    data = data.apply(batching.unbatch())
    self.assertAllEqual(expected_types, data.output_types)

    iterator = data.make_one_shot_iterator()
    op = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual(((i, b"hi"), (10 + i, b"hi"), (20 + i, b"hi")),
                         sess.run(op))

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(op)

  def testUnbatchEmpty(self):
    data = dataset_ops.Dataset.from_tensors(
        (constant_op.constant([]), constant_op.constant([], shape=[0, 4]),
         constant_op.constant([], shape=[0, 4, 0])))
    data = data.apply(batching.unbatch())
    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testUnbatchStaticShapeMismatch(self):
    data = dataset_ops.Dataset.from_tensors((np.arange(7), np.arange(8),
                                             np.arange(9)))
    with self.assertRaises(ValueError):
      data.apply(batching.unbatch())

  def testUnbatchDynamicShapeMismatch(self):
    ph1 = array_ops.placeholder(dtypes.int32, shape=[None])
    ph2 = array_ops.placeholder(dtypes.int32, shape=None)
    data = dataset_ops.Dataset.from_tensors((ph1, ph2))
    data = data.apply(batching.unbatch())
    iterator = data.make_initializable_iterator()
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
        sess.run(next_element)

      # No 0th dimension (i.e. scalar value) for one component.
      sess.run(
          iterator.initializer,
          feed_dict={
              ph1: np.arange(7).astype(np.int32),
              ph2: 7
          })
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(next_element)

  @parameterized.named_parameters(
      ("Default", None, None),
      ("SequentialCalls", 1, None),
      ("ParallelCalls", 2, None),
      ("ParallelBatches", None, 10),
  )
  def testMapAndBatch(self, num_parallel_calls, num_parallel_batches):
    """Test a dataset that maps a TF function across its input elements."""
    # The pipeline is TensorSliceDataset ->
    # RepeatDataset(count) -> MapAndBatchDataset(square_3, batch_size).
    components = (np.arange(7),
                  np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis],
                  np.array(37.0) * np.arange(7))

    count = array_ops.placeholder(dtypes.int64, shape=[])
    batch_size = array_ops.placeholder(dtypes.int64, shape=[])

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    iterator = (
        dataset_ops.Dataset.from_tensor_slices(components).repeat(count).apply(
            batching.map_and_batch(
                map_func=_map_fn,
                batch_size=batch_size,
                num_parallel_calls=num_parallel_calls,
                num_parallel_batches=num_parallel_batches))
        .make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()

    self.assertEqual([[None] + list(c.shape[1:]) for c in components],
                     [t.shape.as_list() for t in get_next])

    with self.cached_session() as sess:
      # Batch of a finite input, where the batch_size divides the
      # total number of elements.
      sess.run(init_op, feed_dict={count: 28, batch_size: 14})
      num_batches = (28 * 7) // 14
      for i in range(num_batches):
        result = self.evaluate(get_next)
        for component, result_component in zip(components, result):
          for j in range(14):
            self.assertAllEqual(component[(i * 14 + j) % 7]**2,
                                result_component[j])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Batch of a finite input, where the batch_size does not
      # divide the total number of elements.
      sess.run(init_op, feed_dict={count: 14, batch_size: 8})

      # We expect (num_batches - 1) full-sized batches.
      num_batches = int(math.ceil((14 * 7) / 8))
      for i in range(num_batches - 1):
        result = self.evaluate(get_next)
        for component, result_component in zip(components, result):
          for j in range(8):
            self.assertAllEqual(component[(i * 8 + j) % 7]**2,
                                result_component[j])
      result = self.evaluate(get_next)
      for component, result_component in zip(components, result):
        for j in range((14 * 7) % 8):
          self.assertAllEqual(component[((num_batches - 1) * 8 + j) % 7]**2,
                              result_component[j])
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Batch of an empty input should fail straight away.
      sess.run(init_op, feed_dict={count: 0, batch_size: 8})
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

      # Empty batch should be an initialization time error.
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(init_op, feed_dict={count: 14, batch_size: 0})

  @parameterized.named_parameters(
      ("Even", False),
      ("Uneven", True),
  )
  def testMapAndBatchPartialBatch(self, drop_remainder):
    iterator = (
        dataset_ops.Dataset.range(10).apply(
            batching.map_and_batch(
                lambda x: array_ops.reshape(x * x, [1]),
                batch_size=4,
                drop_remainder=drop_remainder)).make_one_shot_iterator())
    if drop_remainder:
      self.assertEqual([4, 1], iterator.output_shapes.as_list())
    else:
      self.assertEqual([None, 1], iterator.output_shapes.as_list())
    next_element = iterator.get_next()
    with self.cached_session() as sess:
      self.assertAllEqual([[0], [1], [4], [9]], self.evaluate(next_element))
      self.assertAllEqual([[16], [25], [36], [49]], self.evaluate(next_element))
      if not drop_remainder:
        self.assertAllEqual([[64], [81]], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testMapAndBatchYieldsPartialBatch(self):
    iterator = (
        dataset_ops.Dataset.range(10).apply(
            batching.map_and_batch(lambda x: array_ops.reshape(x * x, [1]),
                                   4)).make_one_shot_iterator())
    self.assertEqual([None, 1], iterator.output_shapes.as_list())
    next_element = iterator.get_next()
    with self.cached_session() as sess:
      self.assertAllEqual([[0], [1], [4], [9]], self.evaluate(next_element))
      self.assertAllEqual([[16], [25], [36], [49]], self.evaluate(next_element))
      self.assertAllEqual([[64], [81]], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testMapAndBatchParallelGetNext(self):
    iterator = (
        dataset_ops.Dataset.range(50000).apply(
            batching.map_and_batch(lambda x: x,
                                   batch_size=100)).make_one_shot_iterator())
    elements = []
    for _ in range(100):
      elements.append(iterator.get_next())
    with self.cached_session() as sess:
      for i in range(5):
        got = self.evaluate(elements)
        got.sort(key=lambda x: x[0])
        expected = []
        for j in range(100):
          expected.append(range(i * 10000 + j * 100, i * 10000 + (j + 1) * 100))
        self.assertAllEqual(got, expected)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(elements)

  def testMapAndBatchParallelGetNextDropRemainder(self):
    iterator = (
        dataset_ops.Dataset.range(49999).apply(
            batching.map_and_batch(
                lambda x: x, batch_size=100,
                drop_remainder=True)).make_one_shot_iterator())
    elements = []
    for _ in range(100):
      elements.append(iterator.get_next())
    with self.cached_session() as sess:
      for i in range(4):
        got = self.evaluate(elements)
        got.sort(key=lambda x: x[0])
        expected = []
        for j in range(100):
          expected.append(range(i * 10000 + j * 100, i * 10000 + (j + 1) * 100))
        self.assertAllEqual(got, expected)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(elements)

  def testMapAndBatchSparse(self):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    iterator = dataset_ops.Dataset.range(10).apply(
        batching.map_and_batch(_sparse, 5)).make_initializable_iterator()
    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.evaluate(init_op)
      for i in range(2):
        actual = self.evaluate(get_next)
        expected = sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            values=[i * 5, i * 5 + 1, i * 5 + 2, i * 5 + 3, i * 5 + 4],
            dense_shape=[5, 1])
        self.assertTrue(sparse_tensor.is_sparse(actual))
        self.assertSparseValuesEqual(actual, expected)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  def testMapAndBatchFails(self):
    """Test a dataset that maps a TF function across its input elements."""
    dataset = dataset_ops.Dataset.from_tensors(
        array_ops.check_numerics(
            constant_op.constant(1.0) / constant_op.constant(0.0), "oops"))
    batch_size = array_ops.placeholder(dtypes.int64, shape=[])
    iterator = (
        dataset.apply(batching.map_and_batch(
            lambda x: x, batch_size)).make_initializable_iterator())
    init_op = iterator.initializer
    with self.cached_session() as sess:
      with self.assertRaisesRegexp(errors.InvalidArgumentError, "oops"):
        sess.run(init_op, feed_dict={batch_size: 14})

  def testMapAndBatchShapeMismatch(self):
    """Test a dataset that maps a TF function across its input elements."""

    def generator():
      yield [1]
      yield [2]
      yield [3]
      yield [[4, 5, 6]]

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int32)
    batch_size = 4
    iterator = (
        dataset.apply(batching.map_and_batch(
            lambda x: x, batch_size)).make_initializable_iterator())
    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.cached_session() as sess:
      self.evaluate(init_op)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "number of elements does not match"):
        sess.run(get_next)

  def testMapAndBatchImplicitDispose(self):
    # Tests whether a map and batch dataset will be cleaned up correctly when
    # the pipeline does not run it until exhaustion.
    # The pipeline is TensorSliceDataset -> RepeatDataset(1000) ->
    # MapAndBatchDataset(f=square_3, batch_size=100).
    components = (np.arange(1000),
                  np.array([[1, 2, 3]]) * np.arange(1000)[:, np.newaxis],
                  np.array(37.0) * np.arange(1000))

    def _map_fn(x, y, z):
      return math_ops.square(x), math_ops.square(y), math_ops.square(z)

    dataset = dataset_ops.Dataset.from_tensor_slices(components).repeat(
        1000).apply(batching.map_and_batch(_map_fn, batch_size=100))
    dataset = dataset.prefetch(5)
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      for _ in range(3):
        sess.run(get_next)

  @parameterized.named_parameters(
      ("1", 0),
      ("2", 5),
      ("3", 10),
      ("4", 90),
      ("5", 95),
      ("6", 99),
  )
  def testMapAndBatchOutOfRangeError(self, threshold):

    def raising_py_fn(i):
      if i >= threshold:
        raise StopIteration()
      else:
        return i

    iterator = (
        dataset_ops.Dataset.range(100).apply(
            batching.map_and_batch(
                lambda x: script_ops.py_func(raising_py_fn, [x], dtypes.int64),
                batch_size=10)).make_one_shot_iterator())
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(threshold // 10):
        self.assertAllEqual([i * 10 + j for j in range(10)],
                            self.evaluate(get_next))
      if threshold % 10 != 0:
        self.assertAllEqual(
            [threshold // 10 * 10 + j for j in range(threshold % 10)],
            sess.run(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(get_next)

  @parameterized.named_parameters(
      ("1", False, dtypes.bool),
      ("2", -42, dtypes.int8),
      ("3", -42, dtypes.int16),
      ("4", -42, dtypes.int32),
      ("5", -42, dtypes.int64),
      ("6", 42, dtypes.uint8),
      ("7", 42, dtypes.uint16),
      ("8", 42.0, dtypes.float16),
      ("9", 42.0, dtypes.float32),
      ("10", 42.0, dtypes.float64),
      ("11", b"hello", dtypes.string),
  )
  def testMapAndBatchTypes(self, element, dtype):

    def gen():
      yield element

    dataset = dataset_ops.Dataset.from_generator(gen, dtype).repeat(100).apply(
        batching.map_and_batch(lambda x: x, batch_size=10))

    get_next = dataset.make_one_shot_iterator().get_next()

    with self.cached_session() as sess:
      for _ in range(10):
        self.assertAllEqual([element for _ in range(10)],
                            self.evaluate(get_next))


class UnbatchDatasetBenchmark(test.Benchmark):

  def benchmarkNativeUnbatch(self):
    batch_sizes = [1, 2, 5, 10, 20, 50]
    elems_per_trial = 10000
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.from_tensors("element").repeat(None)
      batch_size_placeholder = array_ops.placeholder(dtypes.int64, shape=[])
      dataset = dataset.batch(batch_size_placeholder)
      dataset = dataset.apply(batching.unbatch())
      dataset = dataset.skip(elems_per_trial)
      iterator = dataset.make_initializable_iterator()
      next_element = iterator.get_next()

      with session.Session() as sess:
        for batch_size in batch_sizes:
          deltas = []
          for _ in range(5):
            sess.run(
                iterator.initializer,
                feed_dict={batch_size_placeholder: batch_size})
            start = time.time()
            sess.run(next_element.op)
            end = time.time()
            deltas.append((end - start) / elems_per_trial)

          median_wall_time = np.median(deltas)
          print("Unbatch (native) batch size: %d Median wall time per element:"
                " %f microseconds" % (batch_size, median_wall_time * 1e6))
          self.report_benchmark(
              iters=10000,
              wall_time=median_wall_time,
              name="benchmark_unbatch_dataset_native_batch_size_%d" %
              batch_size)

  # Include a benchmark of the previous `unbatch()` implementation that uses
  # a composition of more primitive ops. Eventually we'd hope to generate code
  # that is as good in both cases.
  def benchmarkOldUnbatchImplementation(self):
    batch_sizes = [1, 2, 5, 10, 20, 50]
    elems_per_trial = 10000
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.from_tensors("element").repeat(None)
      batch_size_placeholder = array_ops.placeholder(dtypes.int64, shape=[])
      dataset = dataset.batch(batch_size_placeholder)
      dataset = dataset.flat_map(dataset_ops.Dataset.from_tensor_slices)
      dataset = dataset.skip(elems_per_trial)
      iterator = dataset.make_initializable_iterator()
      next_element = iterator.get_next()

      with session.Session() as sess:
        for batch_size in batch_sizes:
          deltas = []
          for _ in range(5):
            sess.run(
                iterator.initializer,
                feed_dict={batch_size_placeholder: batch_size})
            start = time.time()
            sess.run(next_element.op)
            end = time.time()
            deltas.append((end - start) / elems_per_trial)

          median_wall_time = np.median(deltas)
          print("Unbatch (unfused) batch size: %d Median wall time per element:"
                " %f microseconds" % (batch_size, median_wall_time * 1e6))
          self.report_benchmark(
              iters=10000,
              wall_time=median_wall_time,
              name="benchmark_unbatch_dataset_unfused_batch_size_%d" %
              batch_size)


if __name__ == "__main__":
  test.main()
