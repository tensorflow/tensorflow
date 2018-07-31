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

from tensorflow.contrib.data.python.kernel_tests import reader_dataset_ops_test_base
from tensorflow.contrib.data.python.ops import stats_ops
from tensorflow.core.framework import summary_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class StatsDatasetTestBase(test.TestCase):

  def _assertSummaryHasCount(self, summary_str, tag, expected_value):
    summary_proto = summary_pb2.Summary()
    summary_proto.ParseFromString(summary_str)
    for value in summary_proto.value:
      if tag == value.tag:
        self.assertEqual(expected_value, value.histo.num)
        return
    self.fail("Expected tag %r not found in summary %r" % (tag, summary_proto))

  def _assertSummaryHasSum(self, summary_str, tag, expected_value):
    summary_proto = summary_pb2.Summary()
    summary_proto.ParseFromString(summary_str)
    for value in summary_proto.value:
      if tag == value.tag:
        self.assertEqual(expected_value, value.histo.sum)
        return
    self.fail("Expected tag %r not found in summary %r" % (tag, summary_proto))


class StatsDatasetTest(StatsDatasetTestBase):

  def testBytesProduced(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).apply(
            stats_ops.bytes_produced_stats("bytes_produced")).apply(
                stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
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

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      for i in range(100):
        self.assertEqual(i, sess.run(next_element))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", float(i + 1))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(sess.run(summary_t), "record_latency", 100.0)

  def testReinitialize(self):
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
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

    with self.test_session() as sess:
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

    with self.test_session() as sess:
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

    with self.test_session() as sess:
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

    with self.test_session() as sess:
      sess.run([iterator_0.initializer, iterator_1.initializer])
      for i in range(100):
        self.assertEqual(i * 2, sess.run(next_element))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", float(2 * (i + 1)))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(sess.run(summary_t), "record_latency", 200.0)


class FeatureStatsDatasetTest(
    StatsDatasetTestBase,
    reader_dataset_ops_test_base.ReadBatchFeaturesTestBase):

  def testFeaturesStats(self):
    num_epochs = 5
    total_records = num_epochs * self._num_records
    batch_size = 2
    stats_aggregator = stats_ops.StatsAggregator()
    dataset = self.make_batch_feature(
        filenames=self.test_filenames[0],
        num_epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True,
        shuffle_seed=5,
        drop_final_batch=True).apply(
            stats_ops.set_stats_aggregator(stats_aggregator))
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      for _ in range(total_records // batch_size):
        sess.run(next_element)

      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(
          sess.run(summary_t), "record_stats:features", total_records)
      self._assertSummaryHasCount(
          sess.run(summary_t), "record_stats:feature-values", total_records)
      self._assertSummaryHasSum(
          sess.run(summary_t), "record_stats:features", total_records * 3)
      self._assertSummaryHasSum(
          sess.run(summary_t), "record_stats:feature-values",
          self._sum_keywords(1) * num_epochs + 2 * total_records)


if __name__ == "__main__":
  test.main()
