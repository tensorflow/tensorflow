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
"""Tests for the experimental input pipeline statistics gathering ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.kernel_tests import reader_dataset_ops_test_base
from tensorflow.python.data.experimental.kernel_tests import stats_dataset_test_base
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import stats_aggregator
from tensorflow.python.data.experimental.ops import stats_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


# TODO(jsimsa): Figure out why are graph tests failing.
class StatsDatasetTest(stats_dataset_test_base.StatsDatasetTestBase,
                       parameterized.TestCase):

  @combinations.generate(test_base.eager_only_combinations())
  def testBytesProduced(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).apply(
            stats_ops.bytes_produced_stats("bytes_produced"))
    dataset = self.datasetExperimentalStats(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    expected_sum = 0.0
    for i in range(100):
      self.assertAllEqual(
          np.array([i] * i, dtype=np.int64), self.evaluate(next_element()))
      handle = self.getHandle(aggregator)
      self.assertStatisticsHasCount(handle, "bytes_produced", float(i + 1),
                                    i + 2)
      expected_sum += i * 8.0
      self.assertStatisticsHasSum(handle, "bytes_produced", expected_sum, i + 2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    handle = self.getHandle(aggregator)
    self.assertStatisticsHasCount(handle, "bytes_produced", 100.0, 101)
    self.assertStatisticsHasSum(handle, "bytes_produced", expected_sum, 101)

  @combinations.generate(test_base.eager_only_combinations())
  def testLatencyStats(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))

    dataset = self.datasetExperimentalStats(dataset, aggregator)

    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i, self.evaluate(next_element()))
      handle = self.getHandle(aggregator)
      self.assertStatisticsHasCount(handle, "record_latency", float(i + 1),
                                    i + 2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    handle = self.getHandle(aggregator)
    self.assertStatisticsHasCount(handle, "record_latency", 100.0, 101)

  @combinations.generate(test_base.eager_only_combinations())
  def testPrefetchBufferUtilization(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).prefetch(-1)
    dataset = self.datasetExperimentalStats(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)
    for i in range(100):
      self.assertAllEqual(
          np.array([i] * i, dtype=np.int64), self.evaluate(next_element()))
      handle = self.getHandle(aggregator)
      self.assertStatisticsHasCount(
          handle,
          self.regexForNodeName("PrefetchDataset", "buffer_utilization"),
          float(i + 1),
          3 * i + 4,
          offset=2)
      self.assertStatisticsContains(
          handle, self.regexForNodeName("PrefetchDataset", "buffer_capacity"),
          3 * i + 4)
      self.assertStatisticsContains(
          handle,
          self.regexForNodeName("PrefetchDataset", "buffer_size"),
          3 * i + 4,
          offset=1)
      self.assertStatisticsHasRange(
          handle,
          self.regexForNodeName("PrefetchDataset", "buffer_utilization"),
          0,
          1,
          3 * i + 4,
          offset=2)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    handle = self.getHandle(aggregator)
    self.assertStatisticsHasCount(
        handle,
        self.regexForNodeName("PrefetchDataset", "buffer_utilization"),
        100,
        301,
        offset=2)

  @combinations.generate(test_base.eager_only_combinations())
  def testPrefetchBufferScalars(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(10).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).prefetch(1)
    dataset = self.datasetExperimentalStats(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(10):
      self.assertAllEqual(
          np.array([i] * i, dtype=np.int64), self.evaluate(next_element()))
      handle = self.getHandle(aggregator)
      self.assertStatisticsHasScalarValue(
          handle, self.regexForNodeName("PrefetchDataset", "buffer_capacity"),
          1, 3 * i + 4)
      self.assertStatisticsHasScalarValue(
          handle,
          self.regexForNodeName("PrefetchDataset", "buffer_size"),
          1,
          3 * i + 4,
          offset=1)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.eager_only_combinations())
  def testFilteredElementsStats(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(101).filter(
        lambda x: math_ops.equal(math_ops.mod(x, 3), 0))
    dataset = self.datasetExperimentalStats(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(34):
      self.assertEqual(i * 3, self.evaluate(next_element()))
      handle = self.getHandle(aggregator)
      if i != 0:
        self.assertStatisticsHasScalarValue(
            handle, self.regexForNodeName("FilterDataset", "dropped_elements"),
            float(i * 2))
      self.assertStatisticsHasScalarValue(
          handle, self.regexForNodeName("FilterDataset", "filtered_elements"),
          float(i + 1))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    handle = self.getHandle(aggregator)
    self.assertStatisticsHasScalarValue(
        handle, self.regexForNodeName("FilterDataset", "dropped_elements"),
        67.0)
    self.assertStatisticsHasScalarValue(
        handle, self.regexForNodeName("FilterDataset", "filtered_elements"),
        34.0)

  @combinations.generate(test_base.eager_only_combinations())
  def testReinitialize(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    dataset = self.datasetExperimentalStats(dataset, aggregator)

    for j in range(5):
      next_element = self.getNext(dataset, requires_initialization=True)
      for i in range(100):
        self.assertEqual(i, self.evaluate(next_element()))
        handle = self.getHandle(aggregator)
        self.assertStatisticsHasCount(handle, "record_latency",
                                      float((j * 100) + i + 1),
                                      (j * 100) + i + 2)
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element())
      handle = self.getHandle(aggregator)
      self.assertStatisticsHasCount(handle, "record_latency", (j + 1) * 100.0,
                                    (j * 100) + 101)

  @combinations.generate(test_base.eager_only_combinations())
  def testNoAggregatorRegistered(self):
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))

    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i, self.evaluate(next_element()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleTags(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.latency_stats("record_latency_2"))
    dataset = self.datasetExperimentalStats(dataset, aggregator)

    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i, self.evaluate(next_element()))
      handle = self.getHandle(aggregator)
      self.assertStatisticsHasCount(
          handle, "record_latency", float(i + 1), 2 * i + 3, offset=1)
      self.assertStatisticsHasCount(handle, "record_latency_2", float(i + 1),
                                    2 * i + 3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    handle = self.getHandle(aggregator)
    self.assertStatisticsHasCount(
        handle, "record_latency", 100.0, 201, offset=1)
    self.assertStatisticsHasCount(handle, "record_latency_2", 100.0, 201)

  @combinations.generate(test_base.eager_only_combinations())
  def testRepeatedTags(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.latency_stats("record_latency"))
    dataset = self.datasetExperimentalStats(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i, self.evaluate(next_element()))
      handle = self.getHandle(aggregator)
      self.assertStatisticsHasCount(handle, "record_latency",
                                    float(2 * (i + 1)), 2 * i + 3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    handle = self.getHandle(aggregator)
    self.assertStatisticsHasCount(handle, "record_latency", 200.0, 201)

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleIteratorsSameAggregator(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    dataset = self.datasetExperimentalStats(dataset, aggregator)
    next_element1 = self.getNext(dataset, requires_initialization=True)
    next_element2 = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i * 2, self.evaluate(next_element1() + next_element2()))
      handle = self.getHandle(aggregator)
      self.assertStatisticsHasCount(handle, "record_latency",
                                    float(2 * (i + 1)), 2 * i + 3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element1())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element2())
    handle = self.getHandle(aggregator)
    self.assertStatisticsHasCount(handle, "record_latency", 200.0, 201)

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleDatasetWithPrefixes(self):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    dataset = self.datasetExperimentalStats(
        dataset, aggregator, prefix="dataset1")
    dataset2 = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    dataset2 = self.datasetExperimentalStats(
        dataset2, aggregator, prefix="dataset2")
    next_element1 = self.getNext(dataset, requires_initialization=True)
    next_element2 = self.getNext(dataset2, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i * 2, self.evaluate(next_element1() + next_element2()))
      handle = self.getHandle(aggregator)
      self.assertStatisticsHasCount(
          handle, "dataset1::record_latency", float(i + 1), 2 * i + 3, offset=1)
      self.assertStatisticsHasCount(handle, "dataset2::record_latency",
                                    float(i + 1), 2 * i + 3)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element1())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element2())
    handle = self.getHandle(aggregator)
    self.assertStatisticsHasCount(
        handle, "dataset1::record_latency", 100.0, 201, offset=1)
    self.assertStatisticsHasCount(handle, "dataset2::record_latency", 100.0,
                                  201)

  @combinations.generate(test_base.eager_only_combinations())
  def testMultiplePrefetchStats(self):

    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(10).prefetch(
        2).filter(lambda x: math_ops.equal(math_ops.mod(x, 2), 0)).prefetch(1)

    dataset = self.datasetExperimentalStats(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(5):
      self.assertEqual(i * 2, self.evaluate(next_element()))
      handle = self.getHandle(aggregator)
      # TODO(shivaniagarwal): using exact name of prefetch node than the regex,
      # to differentiate between two prefetch. This might break in future, at
      # which point, it would be best to disable this test.
      self.assertStatisticsHasScalarValue(
          handle, "PrefetchDataset/_5::buffer_capacity", 2)
      self.assertStatisticsContains(handle, "PrefetchDataset/_5::buffer_size")
      self.assertStatisticsHasScalarValue(
          handle, "PrefetchDataset/_8::buffer_capacity", 1)
      self.assertStatisticsContains(handle, "PrefetchDataset/_8::buffer_size")
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())


class ThreadUtilizationStatsTest(stats_dataset_test_base.StatsDatasetTestBase,
                                 parameterized.TestCase):

  @combinations.generate(test_base.eager_only_combinations())
  def testMapBufferUtilization(self):

    def dataset_fn():
      return dataset_ops.Dataset.range(10).map(
          lambda x: array_ops.tile([x], ops.convert_to_tensor([x])),
          num_parallel_calls=4)

    self.parallelCallsStats(
        dataset_fn, {"ParallelMapDataset"}, 10, function_processing_time=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testMapAutoTuneBufferUtilization(self):

    def dataset_fn():
      return dataset_ops.Dataset.range(10).map(
          lambda x: array_ops.tile([x], ops.convert_to_tensor([x])),
          num_parallel_calls=dataset_ops.AUTOTUNE)

    self.parallelCallsStats(
        dataset_fn, {"ParallelMapDataset"}, 10, function_processing_time=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testInterleaveAutoTuneBufferUtilization(self):

    def dataset_fn():

      def interleave_fn(_):
        return dataset_ops.Dataset.range(
            10).map(lambda x: array_ops.tile([x], ops.convert_to_tensor([x])))

      return dataset_ops.Dataset.range(1).interleave(
          interleave_fn,
          cycle_length=1,
          num_parallel_calls=dataset_ops.AUTOTUNE)

    self.parallelCallsStats(dataset_fn, {"ParallelInterleaveDatasetV2"}, 10)

  @combinations.generate(test_base.eager_only_combinations())
  def testMapAndBatchAutoTuneBufferUtilization(self):

    def dataset_fn():
      return dataset_ops.Dataset.range(100).apply(
          batching.map_and_batch(
              lambda x: array_ops.tile([x], ops.convert_to_tensor([2])),
              num_parallel_calls=dataset_ops.AUTOTUNE,
              batch_size=16))

    num_output = 100 // 16 + 1
    self.parallelCallsStats(
        dataset_fn, {"MapAndBatchDataset"},
        num_output,
        check_elements=False,
        function_processing_time=True)


class FeatureStatsDatasetTest(
    stats_dataset_test_base.StatsDatasetTestBase,
    reader_dataset_ops_test_base.MakeBatchedFeaturesDatasetTestBase,
    parameterized.TestCase):

  @combinations.generate(test_base.eager_only_combinations())
  def testFeaturesStats(self):
    num_epochs = 5
    total_records = num_epochs * self._num_records
    batch_size = 2

    def dataset_fn():
      return self.make_batch_feature(
          filenames=self.test_filenames[0],
          num_epochs=num_epochs,
          batch_size=batch_size,
          shuffle=True,
          shuffle_seed=5,
          drop_final_batch=False)

    num_output = total_records // batch_size
    if total_records % batch_size:
      num_output = total_records // batch_size + 1

    self.parallelCallsStats(
        dataset_fn, {"ParseExampleDataset"},
        num_output,
        check_elements=False)

    aggregator = stats_aggregator.StatsAggregator()
    dataset = self.datasetExperimentalStats(
        dataset_fn(), aggregator, prefix="record_stats")

    next_element = self.getNext(dataset, requires_initialization=True)

    for _ in range(num_output):
      self.evaluate(next_element())

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    handle = self.getHandle(aggregator)
    self.assertStatisticsHasCount(
        handle,
        self.regexForNodeName("record_stats::ParseExampleDataset",
                              "features_count"), total_records)
    self.assertStatisticsHasCount(
        handle,
        self.regexForNodeName("record_stats::ParseExampleDataset",
                              "feature_values_count"), total_records)
    self.assertStatisticsHasSum(
        handle,
        self.regexForNodeName("record_stats::ParseExampleDataset",
                              "features_count"), total_records * 4)
    self.assertStatisticsHasSum(
        handle,
        self.regexForNodeName("record_stats::ParseExampleDataset",
                              "feature_values_count"),
        self._sum_keywords(1) * num_epochs + 3 * total_records)


if __name__ == "__main__":
  test.main()
