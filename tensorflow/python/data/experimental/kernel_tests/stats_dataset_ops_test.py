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
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.experimental.ops import stats_aggregator
from tensorflow.python.data.experimental.ops import stats_ops
from tensorflow.python.data.experimental.ops import stats_options
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
def function_set_stats_aggregator(dataset,
                                  aggregator,
                                  prefix="",
                                  counter_prefix=""):
  return dataset.apply(
      stats_ops.set_stats_aggregator(aggregator, prefix, counter_prefix))


def function_apply_options(dataset, aggregator, prefix="", counter_prefix=""):
  options = dataset_ops.Options()
  options.experimental_stats = stats_options.StatsOptions()
  options.experimental_stats.aggregator = aggregator
  options.experimental_stats.prefix = prefix
  options.experimental_stats.counter_prefix = counter_prefix
  options.experimental_stats.latency_all_edges = False
  return dataset.with_options(options)


@parameterized.named_parameters(
    ("SetStatsAggregator", function_set_stats_aggregator),
    ("StatsOptions", function_apply_options),
)
class StatsDatasetTest(stats_dataset_test_base.StatsDatasetTestBase):

  def testBytesProduced(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).apply(
            stats_ops.bytes_produced_stats("bytes_produced"))
    dataset = dataset_transformation(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)
    summary_t = aggregator.get_summary()

    expected_sum = 0.0
    for i in range(100):
      self.assertAllEqual(
          np.array([i] * i, dtype=np.int64), self.evaluate(next_element()))
      summary_str = self.evaluate(aggregator.get_summary())
      self._assertSummaryHasCount(summary_str, "bytes_produced", float(i + 1))
      expected_sum += i * 8.0
      self._assertSummaryHasSum(summary_str, "bytes_produced", expected_sum)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    # TODO(shivaniagrawal): ntentional breaking case
    summary_str = self.evaluate(summary_t)
    self._assertSummaryHasCount(summary_str, "bytes_produced", 100.0)
    self._assertSummaryHasSum(summary_str, "bytes_produced", expected_sum)

  def testLatencyStats(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    dataset = dataset_transformation(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i, self.evaluate(next_element()))
      self._assertSummaryHasCount(
          self.evaluate(aggregator.get_summary()), "record_latency",
          float(i + 1))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    self._assertSummaryHasCount(
        self.evaluate(aggregator.get_summary()), "record_latency", 100.0)

  def testPrefetchBufferUtilization(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).prefetch(-1)
    dataset = dataset_transformation(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)
    for i in range(100):
      self.assertAllEqual(
          np.array([i] * i, dtype=np.int64), self.evaluate(next_element()))
      summary_str = self.evaluate(aggregator.get_summary())
      self._assertSummaryHasCount(summary_str, "Prefetch::buffer_utilization",
                                  float(i + 1))
      self._assertSummaryContains(summary_str, "Prefetch::buffer_capacity")
      self._assertSummaryContains(summary_str, "Prefetch::buffer_size")
      self._assertSummaryHasRange(summary_str, "Prefetch::buffer_utilization",
                                  0, 1)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    summary_str = self.evaluate(aggregator.get_summary())
    self._assertSummaryHasCount(summary_str, "Prefetch::buffer_utilization",
                                100)

  def testPrefetchBufferScalars(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(10).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).prefetch(1)
    dataset = dataset_transformation(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(10):
      self.assertAllEqual(
          np.array([i] * i, dtype=np.int64), self.evaluate(next_element()))
      summary_str = self.evaluate(aggregator.get_summary())
      self._assertSummaryHasScalarValue(summary_str,
                                        "Prefetch::buffer_capacity", 1)
      self._assertSummaryHasScalarValue(summary_str, "Prefetch::buffer_size", 1)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  def testFilteredElementsStats(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(101).filter(
        lambda x: math_ops.equal(math_ops.mod(x, 3), 0))
    dataset = dataset_transformation(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(34):
      self.assertEqual(i * 3, self.evaluate(next_element()))
      summary_str = self.evaluate(aggregator.get_summary())
      if i is not 0:
        self._assertSummaryHasScalarValue(summary_str,
                                          "Filter::dropped_elements",
                                          float(i * 2))
      self._assertSummaryHasScalarValue(
          summary_str, "Filter::filtered_elements", float(i + 1))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    summary_str = self.evaluate(aggregator.get_summary())
    self._assertSummaryHasScalarValue(summary_str, "Filter::dropped_elements",
                                      67.0)
    self._assertSummaryHasScalarValue(summary_str, "Filter::filtered_elements",
                                      34.0)

  def testMapBufferUtilization(self, dataset_transformation):

    def dataset_fn():
      return dataset_ops.Dataset.range(10).map(
          lambda x: array_ops.tile([x], ops.convert_to_tensor([x])),
          num_parallel_calls=4)

    self._testParallelCallsStats(
        dataset_fn,
        "ParallelMap",
        10,
        dataset_transformation,
        function_processing_time=True)

  def testMapAutoTuneBufferUtilization(self, dataset_transformation):

    def dataset_fn():
      dataset = dataset_ops.Dataset.range(10).map(
          lambda x: array_ops.tile([x], ops.convert_to_tensor([x])),
          num_parallel_calls=optimization.AUTOTUNE)
      options = dataset_ops.Options()
      options.experimental_autotune = True
      return dataset.with_options(options)

    self._testParallelCallsStats(
        dataset_fn,
        "ParallelMap",
        10,
        dataset_transformation,
        function_processing_time=True)

  def testInterleaveAutoTuneBufferUtilization(self, dataset_transformation):

    def dataset_fn():
      dataset = dataset_ops.Dataset.range(10).map(
          lambda x: array_ops.tile([x], ops.convert_to_tensor([x])))
      dataset = dataset_ops.Dataset.range(1).interleave(
          lambda _: dataset,
          cycle_length=1,
          num_parallel_calls=optimization.AUTOTUNE)
      options = dataset_ops.Options()
      options.experimental_autotune = True
      return dataset.with_options(options)

    self._testParallelCallsStats(dataset_fn, "ParallelInterleaveV2", 10,
                                 dataset_transformation)

  def testMapAndBatchAutoTuneBufferUtilization(self, dataset_transformation):

    def dataset_fn():
      dataset = dataset_ops.Dataset.range(100).apply(
          batching.map_and_batch(
              lambda x: array_ops.tile([x], ops.convert_to_tensor([2])),
              num_parallel_calls=optimization.AUTOTUNE,
              batch_size=16))
      options = dataset_ops.Options()
      options.experimental_autotune = True
      return dataset.with_options(options)

    num_output = 100 // 16 + 1
    self._testParallelCallsStats(
        dataset_fn,
        "MapAndBatch",
        num_output,
        dataset_transformation,
        check_elements=False,
        function_processing_time=True)

  def testReinitialize(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    dataset = dataset_transformation(dataset, aggregator)

    for j in range(5):
      next_element = self.getNext(dataset, requires_initialization=True)
      for i in range(100):
        self.assertEqual(i, self.evaluate(next_element()))
        self._assertSummaryHasCount(
            self.evaluate(aggregator.get_summary()), "record_latency",
            float((j * 100) + i + 1))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element())
      self._assertSummaryHasCount(
          self.evaluate(aggregator.get_summary()), "record_latency",
          (j + 1) * 100.0)

  def testNoAggregatorRegistered(self, dataset_transformation):
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))

    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i, self.evaluate(next_element()))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  def testMultipleTags(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.latency_stats("record_latency_2"))
    dataset = dataset_transformation(dataset, aggregator)

    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i, self.evaluate(next_element()))
      self._assertSummaryHasCount(
          self.evaluate(aggregator.get_summary()), "record_latency",
          float(i + 1))
      self._assertSummaryHasCount(
          self.evaluate(aggregator.get_summary()), "record_latency_2",
          float(i + 1))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    self._assertSummaryHasCount(
        self.evaluate(aggregator.get_summary()), "record_latency", 100.0)
    self._assertSummaryHasCount(
        self.evaluate(aggregator.get_summary()), "record_latency_2", 100.0)

  def testRepeatedTags(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.latency_stats("record_latency"))
    dataset = dataset_transformation(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i, self.evaluate(next_element()))
      self._assertSummaryHasCount(
          self.evaluate(aggregator.get_summary()), "record_latency",
          float(2 * (i + 1)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    self._assertSummaryHasCount(
        self.evaluate(aggregator.get_summary()), "record_latency", 200.0)

  def testMultipleIteratorsSameAggregator(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    dataset = dataset_transformation(dataset, aggregator)
    next_element1 = self.getNext(dataset, requires_initialization=True)
    next_element2 = self.getNext(dataset, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i * 2, self.evaluate(next_element1() + next_element2()))
      self._assertSummaryHasCount(
          self.evaluate(aggregator.get_summary()), "record_latency",
          float(2 * (i + 1)))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element1())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element2())
    self._assertSummaryHasCount(
        self.evaluate(aggregator.get_summary()), "record_latency", 200.0)

  def testMultipleDatasetWithPrefixes(self, dataset_transformation):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    dataset = dataset_transformation(dataset, aggregator, prefix="dataset1")
    dataset2 = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    dataset2 = dataset_transformation(dataset2, aggregator, prefix="dataset2")
    next_element1 = self.getNext(dataset, requires_initialization=True)
    next_element2 = self.getNext(dataset2, requires_initialization=True)

    for i in range(100):
      self.assertEqual(i * 2, self.evaluate(next_element1() + next_element2()))
      self._assertSummaryHasCount(
          self.evaluate(aggregator.get_summary()), "dataset1_record_latency",
          float(i + 1))
      self._assertSummaryHasCount(
          self.evaluate(aggregator.get_summary()), "dataset2_record_latency",
          float(i + 1))
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element1())
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element2())
    self._assertSummaryHasCount(
        self.evaluate(aggregator.get_summary()), "dataset1_record_latency",
        100.0)
    self._assertSummaryHasCount(
        self.evaluate(aggregator.get_summary()), "dataset2_record_latency",
        100.0)


@parameterized.named_parameters(
    dict(
        testcase_name="SetStatsAggregator",
        dataset_transformation=function_set_stats_aggregator),
    dict(
        testcase_name="StatsOptions",
        dataset_transformation=function_apply_options))
class FeatureStatsDatasetTest(
    stats_dataset_test_base.StatsDatasetTestBase,
    reader_dataset_ops_test_base.MakeBatchedFeaturesDatasetTestBase):

  def testFeaturesStats(self, dataset_transformation):
    num_epochs = 5
    total_records = num_epochs * self._num_records
    batch_size = 2
    aggregator = stats_aggregator.StatsAggregator()

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

    self._testParallelCallsStats(
        dataset_fn,
        "ParseExample",
        num_output,
        dataset_transformation,
        check_elements=False)

    dataset = dataset_transformation(
        dataset_fn(), aggregator, prefix="record_stats")

    next_element = self.getNext(dataset, requires_initialization=True)

    for _ in range(num_output):
      self.evaluate(next_element())

    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    self._assertSummaryHasCount(
        self.evaluate(aggregator.get_summary()), "record_stats_features",
        total_records)
    self._assertSummaryHasCount(
        self.evaluate(aggregator.get_summary()), "record_stats_feature-values",
        total_records)
    self._assertSummaryHasSum(
        self.evaluate(aggregator.get_summary()), "record_stats_features",
        total_records * 4)
    self._assertSummaryHasSum(
        self.evaluate(aggregator.get_summary()), "record_stats_feature-values",
        self._sum_keywords(1) * num_epochs + 3 * total_records)


if __name__ == "__main__":
  test.main()
