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


from tensorflow.core.framework import summary_pb2
from tensorflow.python.platform import test


class StatsDatasetTestBase(test.TestCase):
  """Base class for testing statistics gathered in `StatsAggregator`."""

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
