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
"""Tests for profile_analyzer_cli."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.core.framework import step_stats_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.cli import profile_analyzer_cli
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class ProfileAnalyzerTest(test_util.TensorFlowTestCase):

  def testNodeInfoEmpty(self):
    graph = ops.Graph()
    run_metadata = config_pb2.RunMetadata()

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)
    prof_output = prof_analyzer.list_profile([]).lines
    self.assertEquals([""], prof_output)

  def testSingleDevice(self):
    node1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)

    node2 = step_stats_pb2.NodeExecStats(
        node_name="Mul/456",
        op_start_rel_micros=1,
        op_end_rel_micros=2,
        all_end_rel_micros=3)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = "deviceA"
    device1.node_stats.extend([node1, node2])

    graph = test.mock.MagicMock()
    op1 = test.mock.MagicMock()
    op1.name = "Add/123"
    op1.traceback = [("a/b/file1", 10, "some_var")]
    op1.type = "add"
    op2 = test.mock.MagicMock()
    op2.name = "Mul/456"
    op2.traceback = [("a/b/file1", 11, "some_var")]
    op2.type = "mul"
    graph.get_operations.return_value = [op1, op2]

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)
    prof_output = prof_analyzer.list_profile([]).lines

    self._assertAtLeastOneLineMatches(r"Device 1 of 1: deviceA", prof_output)
    self._assertAtLeastOneLineMatches(r"^Add/123.*2us.*4us", prof_output)
    self._assertAtLeastOneLineMatches(r"^Mul/456.*1us.*3us", prof_output)

  def testMultipleDevices(self):
    node1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=3)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = "deviceA"
    device1.node_stats.extend([node1])

    device2 = run_metadata.step_stats.dev_stats.add()
    device2.device = "deviceB"
    device2.node_stats.extend([node1])

    graph = test.mock.MagicMock()
    op = test.mock.MagicMock()
    op.name = "Add/123"
    op.traceback = [("a/b/file1", 10, "some_var")]
    op.type = "abc"
    graph.get_operations.return_value = [op]

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)
    prof_output = prof_analyzer.list_profile([]).lines

    self._assertAtLeastOneLineMatches(r"Device 1 of 2: deviceA", prof_output)
    self._assertAtLeastOneLineMatches(r"Device 2 of 2: deviceB", prof_output)

    # Try filtering by device.
    prof_output = prof_analyzer.list_profile(["-d", "deviceB"]).lines
    self._assertAtLeastOneLineMatches(r"Device 2 of 2: deviceB", prof_output)
    self._assertNoLinesMatch(r"Device 1 of 2: deviceA", prof_output)

  def testWithSession(self):
    options = config_pb2.RunOptions()
    options.trace_level = config_pb2.RunOptions.FULL_TRACE
    run_metadata = config_pb2.RunMetadata()

    with session.Session() as sess:
      a = constant_op.constant([1, 2, 3])
      b = constant_op.constant([2, 2, 1])
      result = math_ops.add(a, b)

      sess.run(result, options=options, run_metadata=run_metadata)

      prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(
          sess.graph, run_metadata)
      prof_output = prof_analyzer.list_profile([]).lines

      self._assertAtLeastOneLineMatches("Device 1 of", prof_output)
      expected_headers = [
          "Node", "Op Time", "Exec Time", r"Filename:Lineno\(function\)"]
      self._assertAtLeastOneLineMatches(
          ".*".join(expected_headers), prof_output)
      self._assertAtLeastOneLineMatches(r"^Add/", prof_output)
      self._assertAtLeastOneLineMatches(r"Device Total", prof_output)

  def testSorting(self):
    node1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        all_start_micros=123,
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)

    node2 = step_stats_pb2.NodeExecStats(
        node_name="Mul/456",
        all_start_micros=122,
        op_start_rel_micros=1,
        op_end_rel_micros=2,
        all_end_rel_micros=5)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = "deviceA"
    device1.node_stats.extend([node1, node2])

    graph = test.mock.MagicMock()
    op1 = test.mock.MagicMock()
    op1.name = "Add/123"
    op1.traceback = [("a/b/file2", 10, "some_var")]
    op1.type = "add"
    op2 = test.mock.MagicMock()
    op2.name = "Mul/456"
    op2.traceback = [("a/b/file1", 11, "some_var")]
    op2.type = "mul"
    graph.get_operations.return_value = [op1, op2]

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)

    # Default sort by start time (i.e. all_start_micros).
    prof_output = prof_analyzer.list_profile([]).lines
    self.assertRegexpMatches("".join(prof_output), r"Mul/456.*Add/123")
    # Default sort in reverse.
    prof_output = prof_analyzer.list_profile(["-r"]).lines
    self.assertRegexpMatches("".join(prof_output), r"Add/123.*Mul/456")
    # Sort by name.
    prof_output = prof_analyzer.list_profile(["-s", "node"]).lines
    self.assertRegexpMatches("".join(prof_output), r"Add/123.*Mul/456")
    # Sort by op time (i.e. op_end_rel_micros - op_start_rel_micros).
    prof_output = prof_analyzer.list_profile(["-s", "op_time"]).lines
    self.assertRegexpMatches("".join(prof_output), r"Mul/456.*Add/123")
    # Sort by exec time (i.e. all_end_rel_micros).
    prof_output = prof_analyzer.list_profile(["-s", "exec_time"]).lines
    self.assertRegexpMatches("".join(prof_output), r"Add/123.*Mul/456")
    # Sort by line number.
    prof_output = prof_analyzer.list_profile(["-s", "line"]).lines
    self.assertRegexpMatches("".join(prof_output), r"Mul/456.*Add/123")

  def testFiltering(self):
    node1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        all_start_micros=123,
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)

    node2 = step_stats_pb2.NodeExecStats(
        node_name="Mul/456",
        all_start_micros=122,
        op_start_rel_micros=1,
        op_end_rel_micros=2,
        all_end_rel_micros=5)

    run_metadata = config_pb2.RunMetadata()
    device1 = run_metadata.step_stats.dev_stats.add()
    device1.device = "deviceA"
    device1.node_stats.extend([node1, node2])

    graph = test.mock.MagicMock()
    op1 = test.mock.MagicMock()
    op1.name = "Add/123"
    op1.traceback = [("a/b/file2", 10, "some_var")]
    op1.type = "add"
    op2 = test.mock.MagicMock()
    op2.name = "Mul/456"
    op2.traceback = [("a/b/file1", 11, "some_var")]
    op2.type = "mul"
    graph.get_operations.return_value = [op1, op2]

    prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(graph, run_metadata)

    # Filter by name
    prof_output = prof_analyzer.list_profile(["-n", "Add"]).lines
    self._assertAtLeastOneLineMatches(r"Add/123", prof_output)
    self._assertNoLinesMatch(r"Mul/456", prof_output)
    # Filter by op_type
    prof_output = prof_analyzer.list_profile(["-t", "mul"]).lines
    self._assertAtLeastOneLineMatches(r"Mul/456", prof_output)
    self._assertNoLinesMatch(r"Add/123", prof_output)
    # Filter by file name.
    prof_output = prof_analyzer.list_profile(["-f", "file2"]).lines
    self._assertAtLeastOneLineMatches(r"Add/123", prof_output)
    self._assertNoLinesMatch(r"Mul/456", prof_output)
    # Fitler by execution time.
    prof_output = prof_analyzer.list_profile(["-e", "[5, 10]"]).lines
    self._assertAtLeastOneLineMatches(r"Mul/456", prof_output)
    self._assertNoLinesMatch(r"Add/123", prof_output)
    # Fitler by op time.
    prof_output = prof_analyzer.list_profile(["-o", ">=2"]).lines
    self._assertAtLeastOneLineMatches(r"Add/123", prof_output)
    self._assertNoLinesMatch(r"Mul/456", prof_output)

  def _atLeastOneLineMatches(self, pattern, lines):
    pattern_re = re.compile(pattern)
    for line in lines:
      if pattern_re.match(line):
        return True
    return False

  def _assertAtLeastOneLineMatches(self, pattern, lines):
    if not self._atLeastOneLineMatches(pattern, lines):
      raise AssertionError(
          "%s does not match any line in %s." % (pattern, str(lines)))

  def _assertNoLinesMatch(self, pattern, lines):
    if self._atLeastOneLineMatches(pattern, lines):
      raise AssertionError(
          "%s matched at least one line in %s." % (pattern, str(lines)))


if __name__ == "__main__":
  googletest.main()
