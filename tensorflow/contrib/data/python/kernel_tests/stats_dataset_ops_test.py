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

from tensorflow.contrib.data.python.kernel_tests import dataset_serialization_test_base
from tensorflow.contrib.data.python.ops import stats_ops
from tensorflow.core.framework import summary_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class StatsDatasetTest(test.TestCase):

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

  def testBytesProduced(self):
    dataset = dataset_ops.Dataset.range(100).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).apply(
            stats_ops.bytes_produced_stats("bytes_produced"))
    iterator = dataset.make_initializable_iterator()
    stats_aggregator = stats_ops.StatsAggregator()
    stats_aggregator_subscriber = stats_aggregator.subscribe(iterator)
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run([iterator.initializer, stats_aggregator_subscriber])
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
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    iterator = dataset.make_initializable_iterator()
    stats_aggregator = stats_ops.StatsAggregator()
    stats_aggregator_subscriber = stats_aggregator.subscribe(iterator)
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run([iterator.initializer, stats_aggregator_subscriber])
      for i in range(100):
        self.assertEqual(i, sess.run(next_element))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", float(i + 1))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(sess.run(summary_t), "record_latency", 100.0)

  def testReinitialize(self):
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    iterator = dataset.make_initializable_iterator()
    stats_aggregator = stats_ops.StatsAggregator()
    stats_aggregator_subscriber = stats_aggregator.subscribe(iterator)
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run(stats_aggregator_subscriber)
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
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.latency_stats("record_latency_2"))
    iterator = dataset.make_initializable_iterator()
    stats_aggregator = stats_ops.StatsAggregator()
    stats_aggregator_subscriber = stats_aggregator.subscribe(iterator)
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run([iterator.initializer, stats_aggregator_subscriber])
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
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency")).apply(
            stats_ops.latency_stats("record_latency"))
    iterator = dataset.make_initializable_iterator()
    stats_aggregator = stats_ops.StatsAggregator()
    stats_aggregator_subscriber = stats_aggregator.subscribe(iterator)
    next_element = iterator.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run([iterator.initializer, stats_aggregator_subscriber])
      for i in range(100):
        self.assertEqual(i, sess.run(next_element))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", float(2 * (i + 1)))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(sess.run(summary_t), "record_latency", 200.0)

  def testMultipleIteratorsSameAggregator(self):
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    iterator_0 = dataset.make_initializable_iterator()
    iterator_1 = dataset.make_initializable_iterator()
    stats_aggregator = stats_ops.StatsAggregator()
    stats_aggregator_subscribers = [stats_aggregator.subscribe(iterator_0),
                                    stats_aggregator.subscribe(iterator_1)]
    next_element = iterator_0.get_next() + iterator_1.get_next()
    summary_t = stats_aggregator.get_summary()

    with self.test_session() as sess:
      sess.run([iterator_0.initializer, iterator_1.initializer,
                stats_aggregator_subscribers])
      for i in range(100):
        self.assertEqual(i * 2, sess.run(next_element))
        self._assertSummaryHasCount(
            sess.run(summary_t), "record_latency", float(2 * (i + 1)))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
      self._assertSummaryHasCount(sess.run(summary_t), "record_latency", 200.0)

  def testMultipleStatsAggregatorsSameIteratorFail(self):
    dataset = dataset_ops.Dataset.range(100).apply(
        stats_ops.latency_stats("record_latency"))
    iterator = dataset.make_initializable_iterator()
    stats_aggregator_0 = stats_ops.StatsAggregator()
    stats_aggregator_1 = stats_ops.StatsAggregator()

    with self.test_session() as sess:
      sess.run(stats_aggregator_0.subscribe(iterator))
      # TODO(mrry): Consider making this allowable (and also allowing
      # aggregators to unsubscribe).
      with self.assertRaises(errors.FailedPreconditionError):
        sess.run(stats_aggregator_1.subscribe(iterator))


class StatsDatasetSerializationTest(
    dataset_serialization_test_base.DatasetSerializationTestBase):

  def _build_dataset_bytes_stats(self, num_elements):
    return dataset_ops.Dataset.range(num_elements).map(
        lambda x: array_ops.tile([x], ops.convert_to_tensor([x]))).apply(
            stats_ops.bytes_produced_stats("bytes_produced"))

  def testBytesStatsDatasetSaveableCore(self):
    num_outputs = 100
    self.run_core_tests(
        lambda: self._build_dataset_bytes_stats(num_outputs),
        lambda: self._build_dataset_bytes_stats(num_outputs // 10), num_outputs)

  def _build_dataset_latency_stats(self, num_elements, tag="record_latency"):
    return dataset_ops.Dataset.range(num_elements).apply(
        stats_ops.latency_stats(tag))

  def _build_dataset_multiple_tags(self,
                                   num_elements,
                                   tag1="record_latency",
                                   tag2="record_latency_2"):
    return dataset_ops.Dataset.range(num_elements).apply(
        stats_ops.latency_stats(tag1)).apply(stats_ops.latency_stats(tag2))

  def testLatencyStatsDatasetSaveableCore(self):
    num_outputs = 100

    self.run_core_tests(
        lambda: self._build_dataset_latency_stats(num_outputs),
        lambda: self._build_dataset_latency_stats(num_outputs // 10),
        num_outputs)

    self.run_core_tests(lambda: self._build_dataset_multiple_tags(num_outputs),
                        None, num_outputs)

    tag1 = "record_latency"
    tag2 = "record_latency"
    self.run_core_tests(
        lambda: self._build_dataset_multiple_tags(num_outputs, tag1, tag2),
        None, num_outputs)


if __name__ == "__main__":
  test.main()
