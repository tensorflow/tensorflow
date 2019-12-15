# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the private `_RebatchDataset` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.experimental.ops import scan_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


def _flat_shapes(dataset):
  return nest.flatten(dataset_ops.get_legacy_output_shapes(dataset))


class RebatchDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBasic(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1024).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[8] if drop_remainder else [None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 1024, 8)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testScalarInputError(self):
    dataset = dataset_ops.Dataset.range(1024)
    distribute._RebatchDataset(dataset.batch(4), num_replicas=4)
    with self.assertRaisesRegexp(ValueError, "at least one dimension"):
      distribute._RebatchDataset(dataset, num_replicas=4)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testBatchNotDivisibleByNumReplicas(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1024).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=5)
    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = []
    i = 0
    for _ in range(32):  # number of steps
      # first four minibatches have seven elements
      for _ in range(4):
        expected_output.append([k for k in range(i, i + 7)])
        i += 7
      # last minibatch has four elements
      expected_output.append([k for k in range(i, i + 4)])
      i += 4
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testBatchSizeNotDivisibleByNumReplicas2(self):
    dataset = dataset_ops.Dataset.range(32).batch(16, drop_remainder=True)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=5)
    # This will rebatch into sub-batches of size 4, since
    # ceil(16 / 5) = 4. However, that means only the first 4 replicas will get
    # data.
    expected_output = [[k for k in range(i, i + 4)] for i in range(0, 16, 4)]
    expected_output.extend([[]])  # Last replica gets an empty batch
    expected_output.extend(
        [[k for k in range(i, i + 4)] for i in range(16, 32, 4)])
    expected_output.extend([[]])  # Last replica gets an empty batch
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testTupleOutput(self):
    dataset = dataset_ops.Dataset.range(1024).map(lambda x: (x, x)).batch(32)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    expected_output = [([k for k in range(i, i + 8)],  # pylint: disable=g-complex-comprehension
                        [k for k in range(i, i + 8)])
                       for i in range(0, 1024, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedDictionaryOutput(self):
    dataset = dataset_ops.Dataset.range(1024).map(
        lambda x: {"a": x, "b": {"c": x}}).batch(32)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    expected_output = [{"a": [k for k in range(i, i + 8)],  # pylint: disable=g-complex-comprehension
                        "b": {"c": [k for k in range(i, i + 8)]}}
                       for i in range(0, 1024, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testFinalPartialBatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(1032).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[8] if drop_remainder else [None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    # if drop_remainder, the final partial batch is dropped, even though it
    # makes up a complete minibatch.
    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 1024, 8)]  # pylint: disable=g-complex-comprehension
    if not drop_remainder:
      # The last partial batch of size 8 is split over 4 replicas
      expected_output.extend(
          [[k for k in range(i, i + 2)] for i in range(1024, 1032, 2)])
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(drop_remainder=[True, False])))
  def testFinalPartialBatchAfterRebatch(self, drop_remainder):
    dataset = dataset_ops.Dataset.range(34).batch(
        32, drop_remainder=drop_remainder)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[8] if drop_remainder else [None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 32, 8)]  # pylint: disable=g-complex-comprehension
    if not drop_remainder:
      # The last partial batch of size 2 is split over 4 replicas
      expected_output += [[32], [33], [], []]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleBatches(self):
    dataset = dataset_ops.Dataset.range(128).batch(4).batch(8)
    self.assertEqual([[None, None]],
                     [ts.as_list() for ts in _flat_shapes(dataset)])

    # Each element is a list of 8 elements where each element is a list of 4.
    expected_output = [[[j, j + 1, j + 2, j + 3]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 32, 4)]  # generates 8 elements
                       for i in range(0, 128, 32)]
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._RebatchDataset(dataset, 4)
    self.assertEqual([[None, None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # Each element is a list of 2 elements where each element is a list of 4.
    expected_output = [[[j, j + 1, j + 2, j + 3]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 8, 4)]  # generates 2 elements
                       for i in range(0, 128, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatch(self):
    dataset = dataset_ops.Dataset.range(1024).apply(
        batching.map_and_batch(math_ops.square, 32))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [[k**2 for k in range(i, i + 8)]  # pylint: disable=g-complex-comprehension
                       for i in range(0, 1024, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMapAndBatchWithCapturedInput(self):
    captured_t = variables.Variable(42)
    dataset = dataset_ops.Dataset.range(1024).apply(
        batching.map_and_batch(lambda x: captured_t, 32))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [[42 for _ in range(i, i + 8)]  # pylint: disable=g-complex-comprehension
                       for i in range(0, 1024, 8)]
    self.evaluate(variables.global_variables_initializer())
    self.assertDatasetProduces(
        rebatched_dataset, expected_output, requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testPaddedBatch(self):
    dataset = dataset_ops.Dataset.range(128).batch(
        4, drop_remainder=True).padded_batch(
            8, padded_shapes=[5])
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    # Each element is a list of 8 elements in which each element is a list of 5
    # elements, first four are numbers and the last one is a padded zero.
    expected_output = [[[j, j + 1, j + 2, j + 3, 0]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 32, 4)]  # generates 8 elements
                       for i in range(0, 128, 32)]
    self.assertDatasetProduces(dataset, expected_output)
    self.assertEqual([[None, 5]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # Each element is a list of 2 elements in which each element is a list of 5
    # elements, first four are numbers and the last one is a padded zero.
    expected_output = [[[j, j + 1, j + 2, j + 3, 0]  # pylint: disable=g-complex-comprehension
                        for j in range(i, i + 8, 4)]  # generates 2 elements
                       for i in range(0, 128, 8)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testConcatenate(self):
    dataset1 = dataset_ops.Dataset.range(64).batch(8)
    dataset2 = dataset_ops.Dataset.range(32).batch(8)
    dataset = dataset1.concatenate(dataset2)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = ([[i, i + 1] for i in range(0, 64, 2)] +
                       [[i, i + 1] for i in range(0, 32, 2)])
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testConcatenateDifferentShapes(self):
    dataset1 = dataset_ops.Dataset.range(64).batch(16)
    dataset2 = dataset_ops.Dataset.range(32).batch(8)
    dataset = dataset1.concatenate(dataset2)
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = ([[i, i + 1, i + 2, i + 3] for i in range(0, 64, 4)] +
                       [[i, i + 1] for i in range(0, 32, 2)])
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testZip(self):
    dataset1 = dataset_ops.Dataset.range(64).batch(8)
    dataset2 = dataset_ops.Dataset.range(32).batch(8)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[None], [None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [([i, i + 1], [i, i + 1]) for i in range(0, 32, 2)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testZipDifferentShapes(self):
    dataset1 = dataset_ops.Dataset.range(64).batch(16)
    dataset2 = dataset_ops.Dataset.range(32).batch(8)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[None], [None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [([2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3], [i, i + 1])
                       for i in range(0, 32, 2)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testFlatMapBatching(self):
    dataset = dataset_ops.Dataset.range(2).flat_map(
        lambda _: dataset_ops.Dataset.range(32).batch(  # pylint: disable=g-long-lambda
            32))
    # Two elements where each element is range(32)
    expected_output = [[k for k in range(32)] for _ in range(2)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # Two elements where each element is a list of 4 elements where each element
    # is a list of 8.
    expected_output = [[k for k in range(i, i + 8)]  # pylint: disable=g-complex-comprehension
                       for _ in range(2)
                       for i in range(0, 32, 8)]  # generates 4 elements
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testInterleaveBatching(self):
    dataset = dataset_ops.Dataset.range(2).interleave(
        lambda _: dataset_ops.Dataset.range(32).batch(  # pylint: disable=g-long-lambda
            32),
        cycle_length=2)
    # Two elements where each element is range(32)
    expected_output = [[k for k in range(32)] for _ in range(2)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 32, 8)]
    expected_output += expected_output
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testParallelInterleaveBatching(self):
    dataset = dataset_ops.Dataset.range(2).interleave(
        lambda _: dataset_ops.Dataset.range(32).batch(  # pylint: disable=g-long-lambda
            32),
        cycle_length=2,
        num_parallel_calls=2)
    # Two elements where each element is range(32)
    expected_output = [[k for k in range(32)] for _ in range(2)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(dataset, expected_output)

    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)
    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    expected_output = [[k for k in range(i, i + 8)] for i in range(0, 32, 8)]
    expected_output += expected_output
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testGroupByWindowStaticBatch(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [[array_ops.constant(i, dtype=dtypes.int64)] * 3 for i in range(40)])
    reduce_fn = lambda bucket_id, ds: ds.batch(  # pylint: disable=g-long-lambda
        batch_size=10)
    dataset = dataset.apply(
        grouping.group_by_window(
            key_func=lambda x: x[0] % 4, reduce_func=reduce_fn, window_size=10))
    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=2)

    self.assertEqual([[None, 3]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])
    # pylint: disable=g-complex-comprehension
    expected_output = [[[j + i * 4 + k * 20] * 3
                        for i in range(5)]
                       for j in range(4)
                       for k in range(2)]
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testGroupByWindowDynamicBatch(self):
    # {0, 1, 0, 1, ...}
    dataset = dataset_ops.Dataset.range(40).map(lambda x: x % 2)

    def reduce_fn(key, ds):
      # key == 0 -> .batch(5)
      # key == 1 -> .batch(10)
      return ds.batch(batch_size=(key + 1) * 5)

    dataset = dataset.apply(
        grouping.group_by_window(
            key_func=lambda x: x, reduce_func=reduce_fn, window_size=10))
    dataset = distribute._RebatchDataset(dataset, num_replicas=2)

    self.assertEqual([[None]], [ts.as_list() for ts in _flat_shapes(dataset)])

    # The batches of 5 (value == 0) will be split into minibatches of (3, 2) and
    # the batches of 10 (value == 1) split into minibatches of (5, 5)
    # [(batch_size, value), ...]
    pairs = [(3, 0), (2, 0), (3, 0), (2, 0), (5, 1), (5, 1)]
    pairs = pairs * 2
    expected_output = [[value] * batch_size for batch_size, value in pairs]
    self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testGroupByWindowDynamicBatchWithPartialBatch(self):
    # {0, 1, 0, 1, ...}
    dataset = dataset_ops.Dataset.range(40).map(lambda x: x % 2)

    def reduce_fn(key, ds):
      # key == 0 -> .batch(5)
      # key == 1 -> .batch(10)
      return ds.batch(batch_size=(key + 1) * 5)

    dataset = dataset.apply(
        grouping.group_by_window(
            key_func=lambda x: x, reduce_func=reduce_fn, window_size=11))
    dataset = distribute._RebatchDataset(dataset, num_replicas=2)

    self.assertEqual([[None]], [ts.as_list() for ts in _flat_shapes(dataset)])

    pairs = [(3, 0), (2, 0), (3, 0), (2, 0), (1, 0), (0, 0), (5, 1), (5, 1),
             (1, 1), (0, 1), (3, 0), (2, 0), (2, 0), (2, 0), (5, 1), (4, 1)]
    expected_output = [[value] * batch_size for batch_size, value in pairs]
    self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testGroupByWindowDynamicBatchWithPartialBatchWithDropRemainder(self):
    # This test exercises nested batch functionality, dynamic batch size
    # and drop_remainder=True together.
    dataset = dataset_ops.Dataset.range(40).map(lambda x: x % 2)

    def reduce_fn(key, ds):
      # key == 0 -> .batch(5)
      # key == 1 -> .batch(10)
      return ds.batch(batch_size=(key + 1) * 5, drop_remainder=True)

    dataset = dataset.apply(
        grouping.group_by_window(
            key_func=lambda x: x, reduce_func=reduce_fn, window_size=11))
    dataset = distribute._RebatchDataset(dataset, num_replicas=2)

    self.assertEqual([[None]], [ts.as_list() for ts in _flat_shapes(dataset)])

    # The batches of 5 (value == 0) will be split into minibatches of (3, 2) and
    # the batches of 10 (value == 1) split into minibatches of (5, 5)
    # [(batch_size, value), ...]
    pairs = [(3, 0), (2, 0), (3, 0), (2, 0), (5, 1), (5, 1), (3, 0), (2, 0)]
    expected_output = [[value] * batch_size for batch_size, value in pairs]
    self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testScanAfterBatch(self):
    dataset = dataset_ops.Dataset.range(40).batch(10).apply(
        scan_ops.scan(np.int64(2), lambda state, value: (state, value * state)))
    dataset = distribute._RebatchDataset(dataset, num_replicas=2)

    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(dataset)])
    expected_output = [[i * 2 for i in range(j*5, (j+1)*5)] for j in range(8)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testMakeBatchedFeaturesDataset(self):
    # Set up
    fn = os.path.join(self.get_temp_dir(), "tf_record.txt")
    writer = python_io.TFRecordWriter(fn)
    for i in range(1024):
      writer.write(
          example_pb2.Example(
              features=feature_pb2.Features(
                  feature={
                      "value":
                          feature_pb2.Feature(
                              int64_list=feature_pb2.Int64List(value=[i]))
                  })).SerializeToString())
    writer.close()

    dataset = readers.make_batched_features_dataset(
        file_pattern=fn,
        batch_size=32,
        features={"value": parsing_ops.FixedLenFeature([], dtypes.int64)},
        shuffle=False,
        num_epochs=1,
        drop_final_batch=False)

    rebatched_dataset = distribute._RebatchDataset(dataset, num_replicas=4)

    self.assertEqual([[None]],
                     [ts.as_list() for ts in _flat_shapes(rebatched_dataset)])

    expected_output = [{
        "value": [k for k in range(i, i + 8)]
    } for i in range(0, 1024, 8)]  # pylint: disable=g-complex-comprehension
    self.assertDatasetProduces(rebatched_dataset, expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testRaggedTensorDataset(self):
    # Set up a dataset that produces ragged tensors with a static batch size.
    row_lengths = np.random.randint(8, size=128)
    values = np.random.normal(size=np.sum(row_lengths)).astype(np.float32)
    dataset = dataset_ops.Dataset.from_tensor_slices(
        ragged_tensor.RaggedTensor.from_row_lengths(values, row_lengths))
    dataset = dataset.batch(32, drop_remainder=True)
    dataset = distribute._RebatchDataset(dataset, num_replicas=8)
    # After rebatching, batch size is now 4.
    expected_output = []
    value_index = 0
    for batch_row_lengths in row_lengths.reshape((-1, 4)):
      num_values = np.sum(batch_row_lengths)
      expected_output.append(
          ragged_tensor.RaggedTensor.from_row_lengths(
              values[value_index:(value_index + num_values)],
              batch_row_lengths))
      value_index += num_values
    self.assertDatasetProduces(dataset, expected_output)


if __name__ == "__main__":
  test.main()
