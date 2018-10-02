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
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import profile_analyzer_cli
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def no_rewrite_session_config():
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True,
      constant_folding=rewriter_config_pb2.RewriterConfig.OFF)
  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


def _line_number_above():
  return tf_inspect.stack()[1][2] - 1


def _at_least_one_line_matches(pattern, lines):
  pattern_re = re.compile(pattern)
  for i, line in enumerate(lines):
    if pattern_re.search(line):
      return True, i
  return False, None


def _assert_at_least_one_line_matches(pattern, lines):
  any_match, _ = _at_least_one_line_matches(pattern, lines)
  if not any_match:
    raise AssertionError(
        "%s does not match any line in %s." % (pattern, str(lines)))


def _assert_no_lines_match(pattern, lines):
  any_match, _ = _at_least_one_line_matches(pattern, lines)
  if any_match:
    raise AssertionError(
        "%s matched at least one line in %s." % (pattern, str(lines)))


class ProfileAnalyzerListProfileTest(test_util.TensorFlowTestCase):

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

    _assert_at_least_one_line_matches(r"Device 1 of 1: deviceA", prof_output)
    _assert_at_least_one_line_matches(r"^Add/123.*add.*2us.*4us", prof_output)
    _assert_at_least_one_line_matches(r"^Mul/456.*mul.*1us.*3us", prof_output)

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

    _assert_at_least_one_line_matches(r"Device 1 of 2: deviceA", prof_output)
    _assert_at_least_one_line_matches(r"Device 2 of 2: deviceB", prof_output)

    # Try filtering by device.
    prof_output = prof_analyzer.list_profile(["-d", "deviceB"]).lines
    _assert_at_least_one_line_matches(r"Device 2 of 2: deviceB", prof_output)
    _assert_no_lines_match(r"Device 1 of 2: deviceA", prof_output)

  def testWithSession(self):
    options = config_pb2.RunOptions()
    options.trace_level = config_pb2.RunOptions.FULL_TRACE
    run_metadata = config_pb2.RunMetadata()

    with session.Session(config=no_rewrite_session_config()) as sess:
      a = constant_op.constant([1, 2, 3])
      b = constant_op.constant([2, 2, 1])
      result = math_ops.add(a, b)

      sess.run(result, options=options, run_metadata=run_metadata)

      prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(
          sess.graph, run_metadata)
      prof_output = prof_analyzer.list_profile([]).lines

      _assert_at_least_one_line_matches("Device 1 of", prof_output)
      expected_headers = [
          "Node", r"Start Time \(us\)", r"Op Time \(.*\)", r"Exec Time \(.*\)",
          r"Filename:Lineno\(function\)"]
      _assert_at_least_one_line_matches(
          ".*".join(expected_headers), prof_output)
      _assert_at_least_one_line_matches(r"^Add/", prof_output)
      _assert_at_least_one_line_matches(r"Device Total", prof_output)

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
    _assert_at_least_one_line_matches(r"Add/123", prof_output)
    _assert_no_lines_match(r"Mul/456", prof_output)
    # Filter by op_type
    prof_output = prof_analyzer.list_profile(["-t", "mul"]).lines
    _assert_at_least_one_line_matches(r"Mul/456", prof_output)
    _assert_no_lines_match(r"Add/123", prof_output)
    # Filter by file name.
    prof_output = prof_analyzer.list_profile(["-f", ".*file2"]).lines
    _assert_at_least_one_line_matches(r"Add/123", prof_output)
    _assert_no_lines_match(r"Mul/456", prof_output)
    # Fitler by execution time.
    prof_output = prof_analyzer.list_profile(["-e", "[5, 10]"]).lines
    _assert_at_least_one_line_matches(r"Mul/456", prof_output)
    _assert_no_lines_match(r"Add/123", prof_output)
    # Fitler by op time.
    prof_output = prof_analyzer.list_profile(["-o", ">=2"]).lines
    _assert_at_least_one_line_matches(r"Add/123", prof_output)
    _assert_no_lines_match(r"Mul/456", prof_output)

  def testSpecifyingTimeUnit(self):
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

    # Force time unit.
    prof_output = prof_analyzer.list_profile(["--time_unit", "ms"]).lines
    _assert_at_least_one_line_matches(r"Add/123.*add.*0\.002ms", prof_output)
    _assert_at_least_one_line_matches(r"Mul/456.*mul.*0\.005ms", prof_output)
    _assert_at_least_one_line_matches(r"Device Total.*0\.009ms", prof_output)


class ProfileAnalyzerPrintSourceTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(ProfileAnalyzerPrintSourceTest, self).setUp()

    options = config_pb2.RunOptions()
    options.trace_level = config_pb2.RunOptions.FULL_TRACE
    run_metadata = config_pb2.RunMetadata()
    with session.Session() as sess:
      loop_cond = lambda x: math_ops.less(x, 10)
      self.loop_cond_lineno = _line_number_above()
      loop_body = lambda x: math_ops.add(x, 1)
      self.loop_body_lineno = _line_number_above()
      x = constant_op.constant(0, name="x")
      self.x_lineno = _line_number_above()
      loop = control_flow_ops.while_loop(loop_cond, loop_body, [x])
      self.loop_lineno = _line_number_above()
      self.assertEqual(
          10, sess.run(loop, options=options, run_metadata=run_metadata))

      self.prof_analyzer = profile_analyzer_cli.ProfileAnalyzer(
          sess.graph, run_metadata)

  def tearDown(self):
    ops.reset_default_graph()
    super(ProfileAnalyzerPrintSourceTest, self).tearDown()

  def testPrintSourceForWhileLoop(self):
    prof_output = self.prof_analyzer.print_source([__file__])

    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*2\(22\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*2\(20\) .*L%d.*(\S)+" % self.loop_body_lineno,
        prof_output.lines)
    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*7\(55\) .*L%d.*(\S)+" % self.loop_lineno,
        prof_output.lines)

  def testPrintSourceOutputContainsClickableLinks(self):
    prof_output = self.prof_analyzer.print_source([__file__])
    any_match, line_index = _at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*2\(22\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    self.assertTrue(any_match)
    any_menu_item_match = False
    for seg in prof_output.font_attr_segs[line_index]:
      if (isinstance(seg[2][1], debugger_cli_common.MenuItem) and
          seg[2][1].content.startswith("lp --file_path_filter ") and
          "--min_lineno %d" % self.loop_cond_lineno in seg[2][1].content and
          "--max_lineno %d" % (self.loop_cond_lineno + 1) in seg[2][1].content):
        any_menu_item_match = True
        break
    self.assertTrue(any_menu_item_match)

  def testPrintSourceWithNonDefaultTimeUnit(self):
    prof_output = self.prof_analyzer.print_source([
        __file__, "--time_unit", "ms"])

    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*ms .*2\(22\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*ms .*2\(20\) .*L%d.*(\S)+" % self.loop_body_lineno,
        prof_output.lines)
    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*ms .*7\(55\) .*L%d.*(\S)+" % self.loop_lineno,
        prof_output.lines)

  def testPrintSourceWithNodeNameFilter(self):
    prof_output = self.prof_analyzer.print_source([
        __file__, "--node_name_filter", "x$"])

    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*1\(1\) .*L%d.*(\S)+" % self.x_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*us .*2\(22\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*us .*2\(20\) .*L%d.*(\S)+" % self.loop_body_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*ms .*7\(55\) .*L%d.*(\S)+" % self.loop_lineno,
        prof_output.lines)

    # Check clickable link.
    _, line_index = _at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*1\(1\) .*L%d.*(\S)+" % self.x_lineno,
        prof_output.lines)
    any_menu_item_match = False
    for seg in prof_output.font_attr_segs[line_index]:
      if (isinstance(seg[2][1], debugger_cli_common.MenuItem) and
          seg[2][1].content.startswith("lp --file_path_filter ") and
          "--node_name_filter x$" in seg[2][1].content and
          "--min_lineno %d" % self.x_lineno in seg[2][1].content and
          "--max_lineno %d" % (self.x_lineno + 1) in seg[2][1].content):
        any_menu_item_match = True
        break
    self.assertTrue(any_menu_item_match)

  def testPrintSourceWithOpTypeFilter(self):
    prof_output = self.prof_analyzer.print_source([
        __file__, "--op_type_filter", "Less"])

    _assert_at_least_one_line_matches(
        r"\[(\|)+(\s)*\] .*us .*1\(11\) .*L%d.*(\S)+" % self.loop_cond_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*us .*2\(20\) .*L%d.*(\S)+" % self.loop_body_lineno,
        prof_output.lines)
    _assert_no_lines_match(
        r"\[(\|)+(\s)*\] .*us .*7\(55\) .*L%d.*(\S)+" % self.loop_lineno,
        prof_output.lines)

  def testPrintSourceWithNonexistentDeviceGivesCorrectErrorMessage(self):
    prof_output = self.prof_analyzer.print_source([
        __file__, "--device_name_filter", "foo_device"])

    _assert_at_least_one_line_matches(
        r"The source file .* does not contain any profile information for the "
        "previous Session run", prof_output.lines)
    _assert_at_least_one_line_matches(
        r".*--device_name_filter: foo_device", prof_output.lines)

  def testPrintSourceWithUnrelatedFileShowsCorrectErrorMessage(self):
    prof_output = self.prof_analyzer.print_source([tf_inspect.__file__])
    _assert_at_least_one_line_matches(
        r"The source file .* does not contain any profile information for the "
        "previous Session run", prof_output.lines)

  def testPrintSourceOutputContainsInitScrollPosAnnotation(self):
    prof_output = self.prof_analyzer.print_source([
        __file__, "--init_line", str(self.loop_cond_lineno)])
    self.assertEqual(
        self.loop_cond_lineno + 1,  # The extra line is due to the head lines.
        prof_output.annotations[debugger_cli_common.INIT_SCROLL_POS_KEY])


if __name__ == "__main__":
  googletest.main()
