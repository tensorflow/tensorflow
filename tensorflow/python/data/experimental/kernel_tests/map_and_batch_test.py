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
"""Tests for `tf.data.experimental.map_and_batch()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

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
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


class MapAndBatchTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Default", None, None, False),
      ("SequentialCalls", 1, None, False),
      ("ParallelCalls", 2, None, False),
      ("ParallelBatches", None, 10, False),
      ("DefaultNUMA", None, None, True),
      ("SequentialCallsNUMA", 1, None, True),
      ("ParallelCallsNUMA", 2, None, True),
      ("ParallelBatchesNUMA", None, 10, True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatch(self, num_parallel_calls, num_parallel_batches,
                      numa_aware):
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

    dataset = (
        dataset_ops.Dataset.from_tensor_slices(components).repeat(count).apply(
            batching.map_and_batch(
                map_func=_map_fn,
                batch_size=batch_size,
                num_parallel_calls=num_parallel_calls,
                num_parallel_batches=num_parallel_batches)))

    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)

    iterator = dataset_ops.make_initializable_iterator(dataset)
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
        self.evaluate(get_next)

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
        self.evaluate(get_next)

      # Batch of an empty input should fail straight away.
      sess.run(init_op, feed_dict={count: 0, batch_size: 8})
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next)

      # Empty batch should be an initialization time error.
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(init_op, feed_dict={count: 14, batch_size: 0})

  @parameterized.named_parameters(
      ("Even", False, False),
      ("Uneven", True, False),
      ("EvenNUMA", False, True),
      ("UnevenNUMA", True, True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatchPartialBatch(self, drop_remainder, numa_aware):
    dataset = (
        dataset_ops.Dataset.range(10).apply(
            batching.map_and_batch(
                lambda x: array_ops.reshape(x * x, [1]),
                batch_size=4,
                drop_remainder=drop_remainder)))

    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)

    if drop_remainder:
      self.assertEqual([4, 1], iterator.output_shapes.as_list())
    else:
      self.assertEqual([None, 1], iterator.output_shapes.as_list())
    next_element = iterator.get_next()
    with self.cached_session():
      self.assertAllEqual([[0], [1], [4], [9]], self.evaluate(next_element))
      self.assertAllEqual([[16], [25], [36], [49]], self.evaluate(next_element))
      if not drop_remainder:
        self.assertAllEqual([[64], [81]], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @parameterized.named_parameters(
      ("Normal", False),
      ("NUMA", True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatchYieldsPartialBatch(self, numa_aware):
    dataset = (
        dataset_ops.Dataset.range(10).apply(
            batching.map_and_batch(lambda x: array_ops.reshape(x * x, [1]), 4)))

    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)

    iterator = dataset_ops.make_one_shot_iterator(dataset)
    self.assertEqual([None, 1], iterator.output_shapes.as_list())
    next_element = iterator.get_next()
    with self.cached_session():
      self.assertAllEqual([[0], [1], [4], [9]], self.evaluate(next_element))
      self.assertAllEqual([[16], [25], [36], [49]], self.evaluate(next_element))
      self.assertAllEqual([[64], [81]], self.evaluate(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

  @parameterized.named_parameters(
      ("Normal", False),
      ("NUMA", True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatchParallelGetNext(self, numa_aware):
    dataset = dataset_ops.Dataset.range(50000).apply(
        batching.map_and_batch(lambda x: x, batch_size=100))
    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)

    elements = []
    for _ in range(100):
      elements.append(iterator.get_next())
    with self.cached_session():
      for i in range(5):
        got = self.evaluate(elements)
        got.sort(key=lambda x: x[0])
        expected = []
        for j in range(100):
          expected.append(range(i * 10000 + j * 100, i * 10000 + (j + 1) * 100))
        self.assertAllEqual(got, expected)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(elements)

  @parameterized.named_parameters(
      ("Normal", False),
      ("NUMA", True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatchParallelGetNextDropRemainder(self, numa_aware):
    dataset = dataset_ops.Dataset.range(49999).apply(
        batching.map_and_batch(
            lambda x: x, batch_size=100, drop_remainder=True))

    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)

    elements = []
    for _ in range(100):
      elements.append(iterator.get_next())
    with self.cached_session():
      for i in range(4):
        got = self.evaluate(elements)
        got.sort(key=lambda x: x[0])
        expected = []
        for j in range(100):
          expected.append(range(i * 10000 + j * 100, i * 10000 + (j + 1) * 100))
        self.assertAllEqual(got, expected)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(elements)

  @parameterized.named_parameters(
      ("Normal", False),
      ("NUMA", True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatchSparse(self, numa_aware):

    def _sparse(i):
      return sparse_tensor.SparseTensorValue(
          indices=[[0]], values=(i * [1]), dense_shape=[1])

    dataset = dataset_ops.Dataset.range(10).apply(
        batching.map_and_batch(_sparse, 5))
    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)

    init_op = iterator.initializer
    get_next = iterator.get_next()

    with self.cached_session():
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
        self.evaluate(get_next)

  @parameterized.named_parameters(
      ("Normal", False),
      ("NUMA", True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatchFails(self, numa_aware):
    """Test a dataset that maps a TF function across its input elements."""
    dataset = dataset_ops.Dataset.from_tensors(
        array_ops.check_numerics(
            constant_op.constant(1.0) / constant_op.constant(0.0), "oops"))
    batch_size = array_ops.placeholder(dtypes.int64, shape=[])
    dataset = dataset.apply(batching.map_and_batch(lambda x: x, batch_size))
    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)

    init_op = iterator.initializer
    with self.cached_session() as sess:
      with self.assertRaisesRegexp(errors.InvalidArgumentError, "oops"):
        sess.run(init_op, feed_dict={batch_size: 14})

  @parameterized.named_parameters(
      ("Normal", False),
      ("NUMA", True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatchShapeMismatch(self, numa_aware):
    """Test a dataset that maps a TF function across its input elements."""

    def generator():
      yield [1]
      yield [2]
      yield [3]
      yield [[4, 5, 6]]

    dataset = dataset_ops.Dataset.from_generator(
        generator, output_types=dtypes.int32)
    batch_size = 4
    dataset = dataset.apply(batching.map_and_batch(lambda x: x, batch_size))
    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)
    iterator = dataset_ops.make_initializable_iterator(dataset)

    init_op = iterator.initializer
    get_next = iterator.get_next()
    with self.cached_session():
      self.evaluate(init_op)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "number of elements does not match"):
        self.evaluate(get_next)

  @parameterized.named_parameters(
      ("Normal", False),
      ("NUMA", True),
  )
  def testMapAndBatchImplicitDispose(self, numa_aware):
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
    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    with self.cached_session():
      for _ in range(3):
        self.evaluate(get_next)

  @parameterized.named_parameters(
      ("1", 0, False),
      ("2", 5, False),
      ("3", 10, False),
      ("4", 90, False),
      ("5", 95, False),
      ("6", 99, False),
      ("1NUMA", 0, True),
      ("2NUMA", 5, True),
      ("3NUMA", 10, True),
      ("4NUMA", 90, True),
      ("5NUMA", 95, True),
      ("6NUMA", 99, True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatchMapError(self, threshold, numa_aware):

    def raising_py_fn(i):
      if i >= threshold:
        raise StopIteration()
      else:
        return i

    dataset = dataset_ops.Dataset.range(100).apply(
        batching.map_and_batch(
            lambda x: script_ops.py_func(raising_py_fn, [x], dtypes.int64),
            batch_size=10))
    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    with self.cached_session():
      for i in range(threshold // 10):
        self.assertAllEqual([i * 10 + j for j in range(10)],
                            self.evaluate(get_next))
      if numa_aware:
        if threshold % 10 != 0:
          self.assertAllEqual(
              [threshold // 10 * 10 + j for j in range(threshold % 10)],
              self.evaluate(get_next))
      else:
        for i in range(threshold // 10, 10):
          with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(get_next)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next)

  @parameterized.named_parameters(
      ("1", False, dtypes.bool, False),
      ("2", -42, dtypes.int8, False),
      ("3", -42, dtypes.int16, False),
      ("4", -42, dtypes.int32, False),
      ("5", -42, dtypes.int64, False),
      ("6", 42, dtypes.uint8, False),
      ("7", 42, dtypes.uint16, False),
      ("8", 42.0, dtypes.float16, False),
      ("9", 42.0, dtypes.float32, False),
      ("10", 42.0, dtypes.float64, False),
      ("11", b"hello", dtypes.string, False),
      ("1NUMA", False, dtypes.bool, True),
      ("2NUMA", -42, dtypes.int8, True),
      ("3NUMA", -42, dtypes.int16, True),
      ("4NUMA", -42, dtypes.int32, True),
      ("5NUMA", -42, dtypes.int64, True),
      ("6NUMA", 42, dtypes.uint8, True),
      ("7NUMA", 42, dtypes.uint16, True),
      ("8NUMA", 42.0, dtypes.float16, True),
      ("9NUMA", 42.0, dtypes.float32, True),
      ("10NUMA", 42.0, dtypes.float64, True),
      ("11NUMA", b"hello", dtypes.string, True),
  )
  def testMapAndBatchTypes(self, element, dtype, numa_aware):

    def gen():
      yield element

    dataset = dataset_ops.Dataset.from_generator(gen, dtype).repeat(100).apply(
        batching.map_and_batch(lambda x: x, batch_size=10))

    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)

    get_next = dataset_ops.make_one_shot_iterator(dataset).get_next()

    with self.cached_session():
      for _ in range(10):
        self.assertAllEqual([element for _ in range(10)],
                            self.evaluate(get_next))

  @parameterized.named_parameters(
      ("Identity", None, lambda x: x, None),
      ("Replicate", None, lambda x: (x, x), None),
      ("Swap", (None, None), lambda x, y: (y, x), None),
      ("Project", (None, None), lambda x, y: x, None),
  )
  @test_util.run_deprecated_v1
  def testShortCircuit(self, structure, map_fn, num_parallel_calls):
    dataset = self.structuredDataset(structure).repeat().apply(
        batching.map_and_batch(map_fn, batch_size=10))
    get_next = dataset_ops.make_one_shot_iterator(dataset).get_next()

    with self.cached_session() as sess:
      if isinstance(structure, tuple):
        expected = map_fn(
            *sess.run(self.structuredElement(structure, shape=[10])))
      else:
        expected = map_fn(
            sess.run(self.structuredElement(structure, shape=[10])))
      self.assertAllEqual(expected, self.evaluate(get_next))

  @test_util.run_deprecated_v1
  def testShortCircuitCapturedInput(self):
    captured_t = array_ops.placeholder(dtypes.int64, shape=[])
    dataset = self.structuredDataset(None).repeat().apply(
        batching.map_and_batch(lambda x: captured_t, batch_size=10))
    iterator = dataset_ops.make_initializable_iterator(dataset)
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(iterator.initializer, feed_dict={captured_t: 42})
      self.assertAllEqual([42] * 10, self.evaluate(get_next))

  @parameterized.named_parameters(
      ("Normal", False),
      ("NUMA", True),
  )
  @test_util.run_deprecated_v1
  def testMapAndBatchControlFlow(self, numa_aware):

    def map_fn(x):
      previous_cond_v2_value = control_flow_ops.ENABLE_COND_V2
      control_flow_ops.ENABLE_COND_V2 = True
      return_value = control_flow_ops.cond(x < 50, lambda: x + 1, lambda: x * x)
      control_flow_ops.ENABLE_COND_V2 = previous_cond_v2_value
      return return_value

    dataset = dataset_ops.Dataset.range(100).apply(
        batching.map_and_batch(map_fn, batch_size=10))
    if numa_aware:
      options = dataset_ops.Options()
      options.experimental_numa_aware = True
      dataset = dataset.with_options(options)
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()
    with self.cached_session():
      for i in range(10):
        if i < 5:
          self.assertAllEqual([i * 10 + j + 1 for j in range(10)],
                              self.evaluate(get_next))
        else:
          self.assertAllEqual(
              [((i * 10) + j) * ((i * 10) + j) for j in range(10)],
              self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next)


if __name__ == "__main__":
  test.main()
