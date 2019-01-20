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
"""Unit tests for the basic data structures and algorithms for profiling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import step_stats_pb2
from tensorflow.python.debug.lib import profiling
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class AggregateProfile(test_util.TensorFlowTestCase):

  def setUp(self):
    node_1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)
    self.profile_datum_1 = profiling.ProfileDatum(
        "cpu:0", node_1, "/foo/bar.py", 10, "func1", "Add")

    node_2 = step_stats_pb2.NodeExecStats(
        node_name="Mul/456",
        op_start_rel_micros=13,
        op_end_rel_micros=16,
        all_end_rel_micros=17)
    self.profile_datum_2 = profiling.ProfileDatum(
        "cpu:0", node_2, "/foo/bar.py", 11, "func1", "Mul")

    node_3 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=103,
        op_end_rel_micros=105,
        all_end_rel_micros=4)
    self.profile_datum_3 = profiling.ProfileDatum(
        "cpu:0", node_3, "/foo/bar.py", 12, "func1", "Add")

    node_4 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=203,
        op_end_rel_micros=205,
        all_end_rel_micros=4)
    self.profile_datum_4 = profiling.ProfileDatum(
        "gpu:0", node_4, "/foo/bar.py", 13, "func1", "Add")

  def testAggregateProfileConstructorWorks(self):
    aggregate_data = profiling.AggregateProfile(self.profile_datum_1)

    self.assertEqual(2, aggregate_data.total_op_time)
    self.assertEqual(4, aggregate_data.total_exec_time)
    self.assertEqual(1, aggregate_data.node_count)
    self.assertEqual(1, aggregate_data.node_exec_count)

  def testAddToAggregateProfileWithDifferentNodeWorks(self):
    aggregate_data = profiling.AggregateProfile(self.profile_datum_1)
    aggregate_data.add(self.profile_datum_2)

    self.assertEqual(5, aggregate_data.total_op_time)
    self.assertEqual(21, aggregate_data.total_exec_time)
    self.assertEqual(2, aggregate_data.node_count)
    self.assertEqual(2, aggregate_data.node_exec_count)

  def testAddToAggregateProfileWithSameNodeWorks(self):
    aggregate_data = profiling.AggregateProfile(self.profile_datum_1)
    aggregate_data.add(self.profile_datum_2)
    aggregate_data.add(self.profile_datum_3)

    self.assertEqual(7, aggregate_data.total_op_time)
    self.assertEqual(25, aggregate_data.total_exec_time)
    self.assertEqual(2, aggregate_data.node_count)
    self.assertEqual(3, aggregate_data.node_exec_count)

  def testAddToAggregateProfileWithDifferentDeviceSameNodeWorks(self):
    aggregate_data = profiling.AggregateProfile(self.profile_datum_1)
    aggregate_data.add(self.profile_datum_4)

    self.assertEqual(4, aggregate_data.total_op_time)
    self.assertEqual(8, aggregate_data.total_exec_time)
    self.assertEqual(2, aggregate_data.node_count)
    self.assertEqual(2, aggregate_data.node_exec_count)


if __name__ == "__main__":
  googletest.main()
