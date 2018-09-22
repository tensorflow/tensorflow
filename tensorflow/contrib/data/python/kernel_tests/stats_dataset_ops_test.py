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

import numpy as np

from tensorflow.contrib.data.python.kernel_tests import stats_dataset_test_base
from tensorflow.contrib.data.python.ops import stats_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class StatsDatasetTest(stats_dataset_test_base.StatsDatasetTestBase):

  def testBytesProduced(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).apply(
            stats_ops.bytes_produced_stats("bytes_produced")).apply(
                stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.cached_session() as sess:
      sess.run(iterator.initializer)
      expected_sum = 0.0
      for i in range(100):
        self.assertAllEqual(
            np.array([i] * i, dtype=np.int64), sess.run(next_element))
        summary_str = sess.run(summary_t)
        self._assertSummaryHasCount(summary_str, "bytes_produced", float(i + 1))
        expected_sum += i * 8.0
        self._assertSummaryHasSum(summary_str, "bytes_produced", expected_sum)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      summary_str = sess.run(summary_t)
      self._assertSummaryHasCount(summary_str, "bytes_produced", 100.0)
      self._assertSummaryHasSum(summary_str, "bytes_produced", expected_sum)

  def testLatencyStats(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.cached_session() as sess:
      sess.run(iterator.initializer)
      for i in range(100):
        self.assertEqual(i, sess.run(next_element))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", float(i + 1))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(sess.run(summary_t), "record_latency", 100.0)

  def testPrefetchBufferUtilization(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).prefetch(
            -1).apply(stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.cached_session() as sess:
      sess.run(iterator.initializer)
      for i in range(100):
        self.assertAllEqual(
            np.array([i] * i, dtype=np.int64), sess.run(next_element))
        summary_str = sess.run(summary_t)
        self._assertSummaryHasCount(summary_str, "Prefetch::buffer_utilization",
                                    float(i + 1))
        self._assertSummaryContains(summary_str, "Prefetch::buffer_capacity")
        self._assertSummaryContains(summary_str, "Prefetch::buffer_size")
        self._assertSummaryHasRange(summary_str, "Prefetch::buffer_utilization",
                                    0, 1)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      summary_str = sess.run(summary_t)
      self._assertSummaryHasCount(summary_str, "Prefetch::buffer_utilization",
                                  100)

  def testPrefetchBufferScalars(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(10).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).prefetch(
            0).apply(stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.cached_session() as sess:
      sess.run(iterator.initializer)
      for i in range(10):
        self.assertAllEqual(
            np.array([i] * i, dtype=np.int64), sess.run(next_element))
        summary_str = sess.run(summary_t)
        self._assertSummaryHasScalarValue(summary_str,
                                          "Prefetch::buffer_capacity", 0)
        self._assertSummaryHasScalarValue(summary_str, "Prefetch::buffer_size",
                                          0)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testFilteredElementsStats(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(101).filter(
        lambda x: math_ops.equal(math_ops.mod(x, 3), 0)).apply(
            stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      for i in range(34):
        self.assertEqual(i * 3, sess.run(next_element))
        if i is not 0:
          self._assertSummaryHasScalarValue(
              sess.run(summary_t), "Filter::dropped_elements", float(i * 2))
        self._assertSummaryHasScalarValue(
            sess.run(summary_t), "Filter::filtered_elements", float(i + 1))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasScalarValue(
          sess.run(summary_t), "Filter::dropped_elements", 67.0)
      self._assertSummaryHasScalarValue(
          sess.run(summary_t), "Filter::filtered_elements", 34.0)

  def testReinitialize(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.cached_session() as sess:
      for j in range(5):
        sess.run(iterator.initializer)
        for i in range(100):
          self.assertEqual(i, sess.run(next_element))
          self._assertSummaryHasCount(
              sess.run(summary_t), "record_latency", float((j * 100) + i + 1))
        with self.assertRaises(errors.OutOfRangeError):
          sess.run(next_element)
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", (j + 1) * 100.0)

  def testNoAggregatorRegistered(self):
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      sess.run(iterator.initializer)
      for i in range(100):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testMultipleTags(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.latency_stats("record_latency_2")).apply(
                stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.cached_session() as sess:
      sess.run(iterator.initializer)
      for i in range(100):
        self.assertEqual(i, sess.run(next_element))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", float(i + 1))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency_2", float(i + 1))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(sess.run(summary_t), "record_latency", 100.0)
      self._assertSummaryHasCount(
          sess.run(summary_t), "record_latency_2", 100.0)

  def testRepeatedTags(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.latency_stats("record_latency")).apply(
                stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.cached_session() as sess:
      sess.run(iterator.initializer)
      for i in range(100):
        self.assertEqual(i, sess.run(next_element))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", float(2 * (i + 1)))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(sess.run(summary_t), "record_latency", 200.0)

  def testMultipleIteratorsSameAggregator(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.set_stats_aggregator(stats_aggregator))
    iterator_0 = dataset.make_initializable_iterator()
    iterator_1 = dataset.make_initializable_iterator()
    next_element = iterator_0.get_next() + iterator_1.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.cached_session() as sess:
      sess.run([iterator_0.initializer, iterator_1.initializer])
      for i in range(100):
        self.assertEqual(i * 2, sess.run(next_element))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", float(2 * (i + 1)))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(sess.run(summary_t), "record_latency", 200.0)


if __name__ == "__main__":
  test.main()
