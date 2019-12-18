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
"""Base class for testing the input pipeline statistics gathering ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import stats_aggregator
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import gfile


class StatsDatasetTestBase(test_base.DatasetTestBase):
  """Base class for testing statistics gathered in `StatsAggregator`."""

  @classmethod
  def setUpClass(cls):
    if tf2.enabled():
      stats_aggregator._DEFAULT_MAX_QUEUE = 0  # pylint: disable=protected-access
      stats_aggregator.StatsAggregator = stats_aggregator.StatsAggregatorV2
      # TODO(b/116314787): add graph mode support for StatsAggregatorV2.
    else:
      stats_aggregator.StatsAggregator = stats_aggregator.StatsAggregatorV1
      return test_util.run_all_in_graph_and_eager_modes(cls)

  def datasetExperimentalStats(self,
                               dataset,
                               aggregator,
                               prefix="",
                               counter_prefix=""):
    options = dataset_ops.Options()
    options.experimental_stats.aggregator = aggregator
    options.experimental_stats.prefix = prefix
    options.experimental_stats.counter_prefix = counter_prefix
    options.experimental_stats.latency_all_edges = False
    return dataset.with_options(options)

  def regexForNodeName(self, op_name, stats_type=""):
    if stats_type:
      return "".join([op_name, r"/_\d+::", stats_type])
    return "".join([op_name, r"/_\d+"])

  def assertStatisticsContains(self, handle, tag, num_events=-1, offset=0):
    if tf2.enabled():
      self._assertEventContains(handle, tag, num_events, offset)
    else:
      self._assertSummaryContains(handle, tag)

  def assertStatisticsHasCount(self,
                               handle,
                               tag,
                               count,
                               num_events=-1,
                               greater_than=False,
                               offset=0):
    if tf2.enabled():
      self._assertEventHasCount(handle, tag, count, num_events, greater_than,
                                offset)
    else:
      self._assertSummaryHasCount(handle, tag, count, greater_than)

  def assertStatisticsHasSum(self,
                             handle,
                             tag,
                             expected_value,
                             num_events=-1,
                             offset=0):
    if tf2.enabled():
      self._assertEventHasSum(handle, tag, expected_value, num_events, offset)
    else:
      self._assertSummaryHasSum(handle, tag, expected_value)

  def assertStatisticsHasScalarValue(self,
                                     handle,
                                     tag,
                                     expected_value,
                                     num_events=-1,
                                     offset=0):
    if tf2.enabled():
      self._assertEventHasScalarValue(handle, tag, expected_value, num_events,
                                      offset)
    else:
      self._assertSummaryHasScalarValue(handle, tag, expected_value)

  def assertStatisticsHasRange(self,
                               handle,
                               tag,
                               min_value,
                               max_value,
                               num_events=-1,
                               offset=0):
    if tf2.enabled():
      self._assertEventHasRange(handle, tag, min_value, max_value, num_events,
                                offset)
    else:
      self._assertSummaryHasRange(handle, tag, min_value, max_value)

  def _assertSummaryContains(self, summary_str, tag):
    summary_proto = summary_pb2.Summary()
    summary_proto.ParseFromString(summary_str)
    for value in summary_proto.value:
      if re.match(tag, value.tag):
        return
    self.fail("Expected tag %r not found in summary %r" % (tag, summary_proto))

  def _assertSummaryHasCount(self,
                             summary_str,
                             tag,
                             expected_value,
                             greater_than=False):
    summary_proto = summary_pb2.Summary()
    summary_proto.ParseFromString(summary_str)
    for value in summary_proto.value:
      if re.match(tag, value.tag):
        if greater_than:
          self.assertGreaterEqual(value.histo.num, expected_value)
        else:
          self.assertEqual(expected_value, value.histo.num)
        return
    self.fail("Expected tag %r not found in summary %r" % (tag, summary_proto))

  def _assertSummaryHasRange(self, summary_str, tag, min_value, max_value):
    summary_proto = summary_pb2.Summary()
    summary_proto.ParseFromString(summary_str)
    for value in summary_proto.value:
      if re.match(tag, value.tag):
        self.assertLessEqual(min_value, value.histo.min)
        self.assertGreaterEqual(max_value, value.histo.max)
        return
    self.fail("Expected tag %r not found in summary %r" % (tag, summary_proto))

  def _assertSummaryHasSum(self, summary_str, tag, expected_value):
    summary_proto = summary_pb2.Summary()
    summary_proto.ParseFromString(summary_str)
    for value in summary_proto.value:
      if re.match(tag, value.tag):
        self.assertEqual(expected_value, value.histo.sum)
        return
    self.fail("Expected tag %r not found in summary %r" % (tag, summary_proto))

  def _assertSummaryHasScalarValue(self, summary_str, tag, expected_value):
    summary_proto = summary_pb2.Summary()
    summary_proto.ParseFromString(summary_str)
    for value in summary_proto.value:
      if re.match(tag, value.tag):
        self.assertEqual(expected_value, value.simple_value)
        return
    self.fail("Expected tag %r not found in summary %r" % (tag, summary_proto))

  # TODO(b/116314787): add tests to check the correctness of steps as well.
  def _assertEventContains(self, logdir, tag, num_events, offset):
    events = _events_from_logdir(logdir)
    if num_events == -1:
      self.assertGreater(len(events), 1)
      for event in events[::-1]:
        if re.match(tag, event.summary.value[0].tag):
          return
      self.fail("Expected tag %r not found in event file in %r" % (tag, logdir))
    else:
      self.assertEqual(len(events), num_events)
      self.assertTrue(
          re.match(tag, events[num_events - offset - 1].summary.value[0].tag))

  def _assertEventHasCount(self, logdir, tag, count, num_events, greater_than,
                           offset):
    events = _events_from_logdir(logdir)
    if num_events == -1:
      self.assertGreater(len(events), 1)
      for event in events[::-1]:
        if re.match(tag, event.summary.value[0].tag):
          if greater_than:
            self.assertGreaterEqual(event.summary.value[0].histo.num, count)
          else:
            self.assertEqual(count, event.summary.value[0].histo.num)
          return
      self.fail("Expected tag %r not found in event file in %r" % (tag, logdir))
    else:
      self.assertEqual(len(events), num_events)
      self.assertTrue(
          re.match(tag, events[num_events - offset - 1].summary.value[0].tag))
      if greater_than:
        self.assertGreaterEqual(
            events[num_events - offset - 1].summary.value[0].histo.num, count)
      else:
        self.assertEqual(
            events[num_events - offset - 1].summary.value[0].histo.num, count)

  def _assertEventHasSum(self, logdir, tag, expected_value, num_events, offset):
    events = _events_from_logdir(logdir)
    if num_events == -1:
      self.assertGreater(len(events), 1)
      for event in events[::-1]:
        if re.match(tag, event.summary.value[0].tag):
          self.assertEqual(expected_value, event.summary.value[0].histo.sum)
          return
      self.fail("Expected tag %r not found in event file in %r" % (tag, logdir))
    else:
      self.assertEqual(len(events), num_events)
      self.assertTrue(
          re.match(tag, events[num_events - offset - 1].summary.value[0].tag))
      self.assertEqual(
          events[num_events - offset - 1].summary.value[0].histo.sum,
          expected_value)

  def _assertEventHasRange(self, logdir, tag, min_value, max_value, num_events,
                           offset):
    events = _events_from_logdir(logdir)
    if num_events == -1:
      self.assertGreater(len(events), 1)
      for event in events[::-1]:
        if re.match(tag, event.summary.value[0].tag):
          self.assertLessEqual(min_value, event.summary.value[0].histo.min)
          self.assertGreaterEqual(max_value, event.summary.value[0].histo.max)
          return
      self.fail("Expected tag %r not found in event file in %r" % (tag, logdir))
    else:
      self.assertEqual(len(events), num_events)
      self.assertTrue(
          re.match(tag, events[num_events - offset - 1].summary.value[0].tag))
      self.assertLessEqual(
          min_value, events[num_events - offset - 1].summary.value[0].histo.min)
      self.assertGreaterEqual(
          max_value, events[num_events - offset - 1].summary.value[0].histo.max)

  def _assertEventHasScalarValue(self, logdir, tag, expected_value, num_events,
                                 offset):
    events = _events_from_logdir(logdir)
    if num_events == -1:
      self.assertGreater(len(events), 1)
      for event in events[::-1]:
        if re.match(tag, event.summary.value[0].tag):
          self.assertEqual(expected_value, event.summary.value[0].simple_value)
          return
      self.fail("Expected tag %r not found in event file in %r" % (tag, logdir))
    else:
      self.assertEqual(len(events), num_events)
      self.assertTrue(
          re.match(tag, events[num_events - offset - 1].summary.value[0].tag))
      self.assertLessEqual(
          expected_value,
          events[num_events - offset - 1].summary.value[0].simple_value)

  def getHandle(self, aggregator):
    # pylint: disable=protected-access
    if isinstance(aggregator, stats_aggregator.StatsAggregatorV1):
      return self.evaluate(aggregator.get_summary())
    assert isinstance(aggregator, (stats_aggregator.StatsAggregatorV2))
    return aggregator._logdir

  def parallelCallsStats(self,
                         dataset_fn,
                         dataset_names,
                         num_output,
                         function_processing_time=False,
                         check_elements=True):
    aggregator = stats_aggregator.StatsAggregator()
    dataset = dataset_fn()
    dataset = self.datasetExperimentalStats(dataset, aggregator)
    next_element = self.getNext(dataset, requires_initialization=True)

    for i in range(num_output):
      value = self.evaluate(next_element())
      if check_elements:
        self.assertAllEqual(np.array([i] * i, dtype=np.int64), value)
      handle = self.getHandle(aggregator)
      for dataset_name in dataset_names:
        if function_processing_time:
          self.assertStatisticsHasCount(
              handle, r"(.*)::execution_time$", float(i + 1), greater_than=True)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())
    for dataset_name in dataset_names:
      self.assertStatisticsContains(
          handle, self.regexForNodeName(dataset_name, "thread_utilization"))
    if function_processing_time:
      handle = self.getHandle(aggregator)
      for dataset_name in dataset_names:
        self.assertStatisticsHasCount(
            handle,
            r"(.*)::execution_time$",
            float(num_output),
            greater_than=True)


# Adding these two methods from summary_test_util, as summary_test_util is in
# contrib.
def _events_from_file(filepath):
  """Returns all events in a single event file.

  Args:
    filepath: Path to the event file.

  Returns:
    A list of all tf.Event protos in the event file.
  """
  records = list(tf_record.tf_record_iterator(filepath))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def _events_from_logdir(logdir):
  """Returns all events in the single eventfile in logdir.

  Args:
    logdir: The directory in which the single event file is sought.

  Returns:
    A list of all tf.Event protos from the single event file.

  Raises:
    AssertionError: If logdir does not contain exactly one file.
  """
  assert gfile.Exists(logdir)
  files = gfile.ListDirectory(logdir)
  assert len(files) == 1, "Found not exactly one file in logdir: %s" % files
  return _events_from_file(os.path.join(logdir, files[0]))
