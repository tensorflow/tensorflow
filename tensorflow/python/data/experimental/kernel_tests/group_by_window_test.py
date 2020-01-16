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
"""Tests for `tf.data.experimental.group_by_window()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


# NOTE(mrry): These tests are based on the tests in bucket_ops_test.py.
# Currently, they use a constant batch size, though should be made to use a
# different batch size per key.
class GroupByWindowTest(test_base.DatasetTestBase, parameterized.TestCase):

  def _dynamicPad(self, bucket, window, window_size):
    # TODO(mrry): To match `tf.contrib.training.bucket()`, implement a
    # generic form of padded_batch that pads every component
    # dynamically and does not rely on static shape information about
    # the arguments.
    return dataset_ops.Dataset.zip(
        (dataset_ops.Dataset.from_tensors(bucket),
         window.padded_batch(
             32, (tensor_shape.TensorShape([]), tensor_shape.TensorShape(
                 [None]), tensor_shape.TensorShape([3])))))

  @combinations.generate(test_base.default_test_combinations())
  def testSingleBucket(self):

    def _map_fn(v):
      return (v, array_ops.fill([v], v),
              array_ops.fill([3], string_ops.as_string(v)))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(
        math_ops.range(32)).map(_map_fn)

    bucketed_dataset = input_dataset.apply(
        grouping.group_by_window(
            lambda x, y, z: 0,
            lambda k, bucket: self._dynamicPad(k, bucket, 32), 32))
    get_next = self.getNext(bucketed_dataset)

    which_bucket, bucketed_values = self.evaluate(get_next())

    self.assertEqual(0, which_bucket)

    expected_scalar_int = np.arange(32, dtype=np.int64)
    expected_unk_int64 = np.zeros((32, 31)).astype(np.int64)
    for i in range(32):
      expected_unk_int64[i, :i] = i
    expected_vec3_str = np.vstack(3 * [np.arange(32).astype(bytes)]).T

    self.assertAllEqual(expected_scalar_int, bucketed_values[0])
    self.assertAllEqual(expected_unk_int64, bucketed_values[1])
    self.assertAllEqual(expected_vec3_str, bucketed_values[2])

  @combinations.generate(test_base.default_test_combinations())
  def testEvenOddBuckets(self):

    def _map_fn(v):
      return (v, array_ops.fill([v], v),
              array_ops.fill([3], string_ops.as_string(v)))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(
        math_ops.range(64)).map(_map_fn)

    bucketed_dataset = input_dataset.apply(
        grouping.group_by_window(
            lambda x, y, z: math_ops.cast(x % 2, dtypes.int64),
            lambda k, bucket: self._dynamicPad(k, bucket, 32), 32))

    get_next = self.getNext(bucketed_dataset)

    # Get two minibatches (one containing even values, one containing odds)
    which_bucket_even, bucketed_values_even = self.evaluate(get_next())
    which_bucket_odd, bucketed_values_odd = self.evaluate(get_next())

    # Count number of bucket_tensors.
    self.assertEqual(3, len(bucketed_values_even))
    self.assertEqual(3, len(bucketed_values_odd))

    # Ensure bucket 0 was used for all minibatch entries.
    self.assertAllEqual(0, which_bucket_even)
    self.assertAllEqual(1, which_bucket_odd)

    # Test the first bucket outputted, the events starting at 0
    expected_scalar_int = np.arange(0, 32 * 2, 2, dtype=np.int64)
    expected_unk_int64 = np.zeros((32, 31 * 2)).astype(np.int64)
    for i in range(0, 32):
      expected_unk_int64[i, :2 * i] = 2 * i
      expected_vec3_str = np.vstack(
          3 * [np.arange(0, 32 * 2, 2).astype(bytes)]).T

    self.assertAllEqual(expected_scalar_int, bucketed_values_even[0])
    self.assertAllEqual(expected_unk_int64, bucketed_values_even[1])
    self.assertAllEqual(expected_vec3_str, bucketed_values_even[2])

    # Test the second bucket outputted, the odds starting at 1
    expected_scalar_int = np.arange(1, 32 * 2 + 1, 2, dtype=np.int64)
    expected_unk_int64 = np.zeros((32, 31 * 2 + 1)).astype(np.int64)
    for i in range(0, 32):
      expected_unk_int64[i, :2 * i + 1] = 2 * i + 1
      expected_vec3_str = np.vstack(
          3 * [np.arange(1, 32 * 2 + 1, 2).astype(bytes)]).T

    self.assertAllEqual(expected_scalar_int, bucketed_values_odd[0])
    self.assertAllEqual(expected_unk_int64, bucketed_values_odd[1])
    self.assertAllEqual(expected_vec3_str, bucketed_values_odd[2])

  @combinations.generate(test_base.default_test_combinations())
  def testEvenOddBucketsFilterOutAllOdd(self):

    def _map_fn(v):
      return {
          "x": v,
          "y": array_ops.fill([v], v),
          "z": array_ops.fill([3], string_ops.as_string(v))
      }

    def _dynamic_pad_fn(bucket, window, _):
      return dataset_ops.Dataset.zip(
          (dataset_ops.Dataset.from_tensors(bucket),
           window.padded_batch(
               32, {
                   "x": tensor_shape.TensorShape([]),
                   "y": tensor_shape.TensorShape([None]),
                   "z": tensor_shape.TensorShape([3])
               })))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(math_ops.range(
        128)).map(_map_fn).filter(lambda d: math_ops.equal(d["x"] % 2, 0))

    bucketed_dataset = input_dataset.apply(
        grouping.group_by_window(
            lambda d: math_ops.cast(d["x"] % 2, dtypes.int64),
            lambda k, bucket: _dynamic_pad_fn(k, bucket, 32), 32))

    get_next = self.getNext(bucketed_dataset)

    # Get two minibatches ([0, 2, ...] and [64, 66, ...])
    which_bucket0, bucketed_values_even0 = self.evaluate(get_next())
    which_bucket1, bucketed_values_even1 = self.evaluate(get_next())

    # Ensure that bucket 1 was completely filtered out
    self.assertAllEqual(0, which_bucket0)
    self.assertAllEqual(0, which_bucket1)
    self.assertAllEqual(
        np.arange(0, 64, 2, dtype=np.int64), bucketed_values_even0["x"])
    self.assertAllEqual(
        np.arange(64, 128, 2, dtype=np.int64), bucketed_values_even1["x"])

  @combinations.generate(test_base.default_test_combinations())
  def testDynamicWindowSize(self):
    components = np.arange(100).astype(np.int64)

    # Key fn: even/odd
    # Reduce fn: batches of 5
    # Window size fn: even=5, odd=10

    def window_size_func(key):
      window_sizes = constant_op.constant([5, 10], dtype=dtypes.int64)
      return window_sizes[key]

    dataset = dataset_ops.Dataset.from_tensor_slices(components).apply(
        grouping.group_by_window(lambda x: x % 2, lambda _, xs: xs.batch(20),
                                 None, window_size_func))

    get_next = self.getNext(dataset)
    with self.assertRaises(errors.OutOfRangeError):
      batches = 0
      while True:
        result = self.evaluate(get_next())
        is_even = all(x % 2 == 0 for x in result)
        is_odd = all(x % 2 == 1 for x in result)
        self.assertTrue(is_even or is_odd)
        expected_batch_size = 5 if is_even else 10
        self.assertEqual(expected_batch_size, result.shape[0])
        batches += 1

    self.assertEqual(batches, 15)

  @combinations.generate(test_base.default_test_combinations())
  def testSimple(self):
    components = np.random.randint(100, size=(200,)).astype(np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(
        components).map(lambda x: x * x).apply(
            grouping.group_by_window(lambda x: x % 2, lambda _, xs: xs.batch(4),
                                     4))
    get_next = self.getNext(dataset)
    counts = []
    with self.assertRaises(errors.OutOfRangeError):
      while True:
        result = self.evaluate(get_next())
        self.assertTrue(
            all(x % 2 == 0 for x in result) or all(x % 2 == 1) for x in result)
        counts.append(result.shape[0])

    self.assertEqual(len(components), sum(counts))
    num_full_batches = len([c for c in counts if c == 4])
    self.assertGreaterEqual(num_full_batches, 24)
    self.assertTrue(all(c == 4 for c in counts[:num_full_batches]))

  @combinations.generate(test_base.default_test_combinations())
  def testImmediateOutput(self):
    components = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0], dtype=np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).repeat(
        -1).apply(
            grouping.group_by_window(lambda x: x % 3, lambda _, xs: xs.batch(4),
                                     4))
    get_next = self.getNext(dataset)
    # The input is infinite, so this test demonstrates that:
    # 1. We produce output without having to consume the entire input,
    # 2. Different buckets can produce output at different rates, and
    # 3. For deterministic input, the output is deterministic.
    for _ in range(3):
      self.assertAllEqual([0, 0, 0, 0], self.evaluate(get_next()))
      self.assertAllEqual([1, 1, 1, 1], self.evaluate(get_next()))
      self.assertAllEqual([2, 2, 2, 2], self.evaluate(get_next()))
      self.assertAllEqual([0, 0, 0, 0], self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testSmallGroups(self):
    components = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components).apply(
        grouping.group_by_window(lambda x: x % 2, lambda _, xs: xs.batch(4), 4))
    get_next = self.getNext(dataset)
    self.assertAllEqual([0, 0, 0, 0], self.evaluate(get_next()))
    self.assertAllEqual([1, 1, 1, 1], self.evaluate(get_next()))
    # The small outputs at the end are deterministically produced in key
    # order.
    self.assertAllEqual([0, 0, 0], self.evaluate(get_next()))
    self.assertAllEqual([1], self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testEmpty(self):
    dataset = dataset_ops.Dataset.range(4).apply(
        grouping.group_by_window(lambda _: 0, lambda _, xs: xs, 0))

    get_next = self.getNext(dataset)
    with self.assertRaisesRegexp(
        errors.InvalidArgumentError,
        "Window size must be greater than zero, but got 0."):
      print(self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testReduceFuncError(self):
    components = np.random.randint(100, size=(200,)).astype(np.int64)

    def reduce_func(_, xs):
      # Introduce an incorrect padded shape that cannot (currently) be
      # detected at graph construction time.
      return xs.padded_batch(
          4,
          padded_shapes=(tensor_shape.TensorShape([]),
                         constant_op.constant([5], dtype=dtypes.int64) * -1))

    dataset = dataset_ops.Dataset.from_tensor_slices(
        components).map(lambda x: (x, ops.convert_to_tensor([x * x]))).apply(
            grouping.group_by_window(lambda x, _: x % 2, reduce_func, 32))
    get_next = self.getNext(dataset)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testConsumeWindowDatasetMoreThanOnce(self):
    components = np.random.randint(50, size=(200,)).astype(np.int64)

    def reduce_func(key, window):
      # Apply two different kinds of padding to the input: tight
      # padding, and quantized (to a multiple of 10) padding.
      return dataset_ops.Dataset.zip((
          window.padded_batch(
              4, padded_shapes=tensor_shape.TensorShape([None])),
          window.padded_batch(
              4, padded_shapes=ops.convert_to_tensor([(key + 1) * 10])),
      ))

    dataset = dataset_ops.Dataset.from_tensor_slices(
        components
    ).map(lambda x: array_ops.fill([math_ops.cast(x, dtypes.int32)], x)).apply(
        grouping.group_by_window(
            lambda x: math_ops.cast(array_ops.shape(x)[0] // 10, dtypes.int64),
            reduce_func, 4))

    get_next = self.getNext(dataset)
    counts = []
    with self.assertRaises(errors.OutOfRangeError):
      while True:
        tight_result, multiple_of_10_result = self.evaluate(get_next())
        self.assertEqual(0, multiple_of_10_result.shape[1] % 10)
        self.assertAllEqual(tight_result,
                            multiple_of_10_result[:, :tight_result.shape[1]])
        counts.append(tight_result.shape[0])
    self.assertEqual(len(components), sum(counts))

  @combinations.generate(test_base.default_test_combinations())
  def testShortCircuit(self):

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(
        grouping.group_by_window(lambda x: x, lambda _, window: window.batch(1),
                                 1))
    self.assertDatasetProduces(
        dataset, expected_output=[[i] for i in range(10)])


if __name__ == "__main__":
  test.main()
