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
from absl.testing import parameterized
import numpy as np

from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python.data.kernel_tests import checkpoint_test_base
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
      return (v, array_ops.fill([v],
                                v), array_ops.fill([3],
                                                   string_ops.as_string(v)))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(
        math_ops.range(32)).map(_map_fn)

    bucketed_dataset = input_dataset.group_by_window(
        key_func=lambda x, y, z: 0,
        reduce_func=lambda k, bucket: self._dynamicPad(k, bucket, 32),
        window_size=32)
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
      return (v, array_ops.fill([v],
                                v), array_ops.fill([3],
                                                   string_ops.as_string(v)))

    input_dataset = dataset_ops.Dataset.from_tensor_slices(
        math_ops.range(64)).map(_map_fn)

    bucketed_dataset = input_dataset.group_by_window(
        key_func=lambda x, y, z: math_ops.cast(x % 2, dtypes.int64),
        reduce_func=lambda k, bucket: self._dynamicPad(k, bucket, 32),
        window_size=32)

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
      expected_vec3_str = np.vstack(3 *
                                    [np.arange(0, 32 * 2, 2).astype(bytes)]).T

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

    bucketed_dataset = input_dataset.group_by_window(
        key_func=lambda d: math_ops.cast(d["x"] % 2, dtypes.int64),
        reduce_func=lambda k, bucket: _dynamic_pad_fn(k, bucket, 32),
        window_size=32)

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

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.group_by_window(
        key_func=lambda x: x % 2,
        reduce_func=lambda _, xs: xs.batch(20),
        window_size=None,
        window_size_func=window_size_func)

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
    dataset = dataset_ops.Dataset.from_tensor_slices(components).map(
        lambda x: x * x)
    dataset = dataset.group_by_window(
        key_func=lambda x: x % 2,
        reduce_func=lambda _, xs: xs.batch(4),
        window_size=4)
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
    components = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0],
                          dtype=np.int64)
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.repeat(-1)
    dataset = dataset.group_by_window(
        key_func=lambda x: x % 3,
        reduce_func=lambda _, xs: xs.batch(4),
        window_size=4)
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
    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.group_by_window(
        key_func=lambda x: x % 2,
        reduce_func=lambda _, xs: xs.batch(4),
        window_size=4)
    get_next = self.getNext(dataset)
    self.assertAllEqual([0, 0, 0, 0], self.evaluate(get_next()))
    self.assertAllEqual([1, 1, 1, 1], self.evaluate(get_next()))
    # The small outputs at the end are deterministically produced in key
    # order.
    self.assertAllEqual([0, 0, 0], self.evaluate(get_next()))
    self.assertAllEqual([1], self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testEmpty(self):
    dataset = dataset_ops.Dataset.range(4).group_by_window(
        key_func=lambda _: 0, reduce_func=lambda _, xs: xs, window_size=0)
    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Window size must be greater than zero, but got 0."):
      print(self.evaluate(get_next()))

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

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.map(
        lambda x: array_ops.fill([math_ops.cast(x, dtypes.int32)], x))
    # pylint: disable=g-long-lambda
    dataset = dataset.group_by_window(
        key_func=lambda x: math_ops.cast(
            array_ops.shape(x)[0] // 10, dtypes.int64),
        reduce_func=reduce_func,
        window_size=4)

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

    dataset = dataset_ops.Dataset.range(10).group_by_window(
        key_func=lambda x: x,
        reduce_func=lambda _, window: window.batch(1),
        window_size=1)
    self.assertDatasetProduces(
        dataset, expected_output=[[i] for i in range(10)])

  @combinations.generate(test_base.default_test_combinations())
  def testGroupByWindowWithAutotune(self):
    dataset = dataset_ops.Dataset.range(1000).group_by_window(
        key_func=lambda x: x // 10,
        reduce_func=lambda key, window: dataset_ops.Dataset.from_tensors(key),
        window_size=4)
    dataset = dataset.map(lambda x: x + 1, num_parallel_calls=-1)
    get_next = self.getNext(dataset)
    self.evaluate(get_next())

  @combinations.generate(test_base.default_test_combinations())
  def testGroupByWindowCardinality(self):
    dataset = dataset_ops.Dataset.range(1).repeat().group_by_window(
        key_func=lambda x: x % 2,
        reduce_func=lambda key, window: dataset_ops.Dataset.from_tensors(key),
        window_size=4)
    self.assertEqual(self.evaluate(dataset.cardinality()), dataset_ops.INFINITE)

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    dataset = dataset_ops.Dataset.from_tensors(np.int64(42)).group_by_window(
        key_func=lambda x: x,
        reduce_func=lambda key, window: window.batch(4),
        window_size=4,
        name="group_by_window")
    self.assertDatasetProduces(dataset, [[42]])


class GroupByWindowCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                                  parameterized.TestCase):

  def _build_dataset(self, components):
    dataset = dataset_ops.Dataset.from_tensor_slices(components).repeat(-1)
    dataset = dataset.group_by_window(
        key_func=lambda x: x % 3,
        reduce_func=lambda _, xs: xs.batch(4),
        window_size=4)
    return dataset

  @combinations.generate(test_base.default_test_combinations())
  def test(self):
    components = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0],
                          dtype=np.int64)
    self.verify_unused_iterator(
        lambda: self._build_dataset(components),
        num_outputs=12,
        verify_exhausted=False)
    self.verify_multiple_breaks(
        lambda: self._build_dataset(components),
        num_outputs=12,
        verify_exhausted=False)
    self.verify_reset_restored_iterator(
        lambda: self._build_dataset(components),
        num_outputs=12,
        verify_exhausted=False)


class GroupByWindowErrorMessageTest(
    test_base.DatasetTestBase, parameterized.TestCase
):

  @combinations.generate(test_base.default_test_combinations())
  def testReduceFuncError(self):
    components = np.random.randint(100, size=(200,)).astype(np.int64)

    def my_reduce_func(_, window_dataset):
      # Introduce an incorrect padded shape that cannot (currently) be
      # detected at graph construction time.
      return window_dataset.padded_batch(
          4,
          padded_shapes=(
              tensor_shape.TensorShape([]),
              constant_op.constant([5], dtype=dtypes.int64) * -1,
          ),
      )

    dataset = dataset_ops.Dataset.from_tensor_slices(components)
    dataset = dataset.map(lambda x: (x, ops.convert_to_tensor([x * x])))
    dataset = dataset.group_by_window(
        key_func=lambda x, _: x % 2, reduce_func=my_reduce_func, window_size=32
    )
    get_next = self.getNext(dataset)
    with self.assertRaises(errors.InternalError) as error:
      self.evaluate(get_next())

    msg = str(error.exception)
    self.assertIn(error_codes_pb2.Code.Name(errors.INVALID_ARGUMENT), msg)
    self.assertIn(
        my_reduce_func.__name__,
        msg,
        "{} should show up in the error message".format(
            my_reduce_func.__name__
        ),
    )

  @combinations.generate(test_base.default_test_combinations())
  def testPropagateUserDefinedFunctionErrorMessage(self):
    dataset = dataset_ops.Dataset.from_tensor_slices([0])

    def a_cool_user_defined_reduce_func(unused_key, window_dataset):
      it = iter(window_dataset)
      l = [next(it) for _ in range(2)]  # This causes OutOfRange error
      return dataset_ops.Dataset.from_tensor_slices(l)

    dataset = dataset.group_by_window(
        key_func=lambda x: 0,
        window_size=2,
        reduce_func=a_cool_user_defined_reduce_func,
    )

    get_next = self.getNext(dataset)
    with self.assertRaisesRegex(
        errors.InternalError,
        ".*{}.*".format(a_cool_user_defined_reduce_func.__name__),
        msg=(
            "The name of user-defined-function should show up in the error"
            " message"
        ),
    ):
      # Loop over the dataset
      with self.assertRaises(errors.OutOfRangeError):
        while True:
          self.evaluate(get_next())


if __name__ == "__main__":
  test.main()
