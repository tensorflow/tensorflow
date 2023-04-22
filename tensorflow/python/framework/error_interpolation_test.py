# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.errors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import ops
from tensorflow.python.framework import traceable_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

# A mock for ``tf_stack.FrameSummary``.
FrameSummary = collections.namedtuple(
    "StackFrame", ["filename", "lineno", "name", "line"])


def _make_frame_with_filename(op, idx, filename):
  """Return a copy of an existing stack frame with a new filename."""
  frame = op._traceback[idx]
  return FrameSummary(
      filename,
      frame.lineno,
      frame.name,
      frame.line)


def _modify_op_stack_with_filenames(op, num_user_frames, user_filename,
                                    num_inner_tf_frames):
  """Replace op._traceback with a new traceback using special filenames."""
  tf_filename = error_interpolation._FRAMEWORK_PATH_PREFIXES[0] + "%d.py"
  user_filename = os.path.join("%d", "my_favorite_file.py")

  num_requested_frames = num_user_frames + num_inner_tf_frames
  num_actual_frames = len(op._traceback)
  num_outer_frames = num_actual_frames - num_requested_frames
  assert num_requested_frames <= num_actual_frames, "Too few real frames."

  # The op's traceback has outermost frame at index 0.
  stack = []
  for idx in range(0, num_outer_frames):
    stack.append(op._traceback[idx])
  for idx in range(len(stack), len(stack) + num_user_frames):
    stack.append(_make_frame_with_filename(op, idx, user_filename % idx))
  for idx in range(len(stack), len(stack) + num_inner_tf_frames):
    stack.append(_make_frame_with_filename(op, idx, tf_filename % idx))
  op._traceback = stack


class ComputeDeviceSummaryFromOpTest(test.TestCase):

  def testCorrectFormatWithActiveDeviceAssignments(self):
    assignments = []
    assignments.append(
        traceable_stack.TraceableObject(
            "/cpu:0", filename="hope.py", lineno=24))
    assignments.append(
        traceable_stack.TraceableObject(
            "/gpu:2", filename="please.py", lineno=42))

    summary = error_interpolation._compute_device_summary_from_list(
        "nodename", assignments, prefix="  ")

    self.assertIn("nodename", summary)
    self.assertIn("tf.device(/cpu:0)", summary)
    self.assertIn("<hope.py:24>", summary)
    self.assertIn("tf.device(/gpu:2)", summary)
    self.assertIn("<please.py:42>", summary)

  def testCorrectFormatWhenNoColocationsWereActive(self):
    device_assignment_list = []
    summary = error_interpolation._compute_device_summary_from_list(
        "nodename", device_assignment_list, prefix="  ")
    self.assertIn("nodename", summary)
    self.assertIn("No device assignments", summary)


class ComputeColocationSummaryFromOpTest(test.TestCase):

  def testCorrectFormatWithActiveColocations(self):
    t_obj_1 = traceable_stack.TraceableObject(
        None, filename="test_1.py", lineno=27)
    t_obj_2 = traceable_stack.TraceableObject(
        None, filename="test_2.py", lineno=38)
    colocation_dict = {
        "test_node_1": t_obj_1,
        "test_node_2": t_obj_2,
    }
    summary = error_interpolation._compute_colocation_summary_from_dict(
        "node_name", colocation_dict, prefix="  ")
    self.assertIn("node_name", summary)
    self.assertIn("colocate_with(test_node_1)", summary)
    self.assertIn("<test_1.py:27>", summary)
    self.assertIn("colocate_with(test_node_2)", summary)
    self.assertIn("<test_2.py:38>", summary)

  def testCorrectFormatWhenNoColocationsWereActive(self):
    colocation_dict = {}
    summary = error_interpolation._compute_colocation_summary_from_dict(
        "node_name", colocation_dict, prefix="  ")
    self.assertIn("node_name", summary)
    self.assertIn("No node-device colocations", summary)


# Note that the create_graph_debug_info_def needs to run on graph mode ops,
# so it is excluded from eager tests. Even when used in eager mode, it is
# via FunctionGraphs, and directly verifying in graph mode is the narrowest
# way to unit test the functionality.
class CreateGraphDebugInfoDefTest(test.TestCase):

  def _getFirstStackTraceForFile(self, graph_debug_info, key, file_index):
    self.assertIn(key, graph_debug_info.traces)
    stack_trace = graph_debug_info.traces[key]
    found_flc = None
    for flc in stack_trace.file_line_cols:
      if flc.file_index == file_index:
        found_flc = flc
        break
    self.assertIsNotNone(found_flc,
                         "Could not find a stack trace entry for file")
    return found_flc

  def testStackTraceExtraction(self):
    # This test is verifying stack trace information added in graph mode, so
    # only makes sense in graph mode.
    with ops.Graph().as_default():
      # Since the create_graph_debug_info_def() function does not actually
      # do anything special with functions except name mangling, just verify
      # it with a loose op and manually provided function name.
      # The following ops *must* be on consecutive lines (it will be verified
      # in the resulting trace).
      # pyformat: disable
      global_op = constant_op.constant(0, name="Global").op
      op1 = constant_op.constant(1, name="One").op
      op2 = constant_op.constant(2, name="Two").op
      non_traceback_op = constant_op.constant(3, name="NonTraceback").op
      # Ensure op without traceback does not fail
      del non_traceback_op._traceback
      # pyformat: enable

      export_ops = [("", global_op), ("func1", op1), ("func2", op2),
                    ("func2", non_traceback_op)]
      graph_debug_info = error_interpolation.create_graph_debug_info_def(
          export_ops)
      this_file_index = -1
      for file_index, file_name in enumerate(graph_debug_info.files):
        if "{}error_interpolation_test.py".format(os.sep) in file_name:
          this_file_index = file_index
      self.assertGreaterEqual(
          this_file_index, 0,
          "Could not find this file in trace:" + repr(graph_debug_info))

      # Verify the traces exist for each op.
      global_flc = self._getFirstStackTraceForFile(graph_debug_info, "Global@",
                                                   this_file_index)
      op1_flc = self._getFirstStackTraceForFile(graph_debug_info, "One@func1",
                                                this_file_index)
      op2_flc = self._getFirstStackTraceForFile(graph_debug_info, "Two@func2",
                                                this_file_index)

      global_line = global_flc.line
      self.assertEqual(op1_flc.line, global_line + 1, "op1 not on next line")
      self.assertEqual(op2_flc.line, global_line + 2, "op2 not on next line")


class InterpolateFilenamesAndLineNumbersTest(test.TestCase):

  def testFindIndexOfDefiningFrameForOp(self):
    with ops.Graph().as_default():
      local_op = constant_op.constant(42).op
      user_filename = "hope.py"
      _modify_op_stack_with_filenames(
          local_op,
          num_user_frames=3,
          user_filename=user_filename,
          num_inner_tf_frames=5)
      idx = error_interpolation._find_index_of_defining_frame(
          local_op._traceback)
      # Expected frame is 6th from the end because there are 5 inner frames with
      # TF filenames.
      expected_frame = len(local_op._traceback) - 6
      self.assertEqual(expected_frame, idx)

  def testFindIndexOfDefiningFrameForOpReturnsZeroOnError(self):
    with ops.Graph().as_default():
      local_op = constant_op.constant(43).op
      # Truncate stack to known length.
      local_op._traceback = local_op._traceback[:7]
      # Ensure all frames look like TF frames.
      _modify_op_stack_with_filenames(
          local_op,
          num_user_frames=0,
          user_filename="user_file.py",
          num_inner_tf_frames=7)
      idx = error_interpolation._find_index_of_defining_frame(
          local_op._traceback)
      self.assertEqual(0, idx)

  def testNothingToDo(self):
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      normal_string = "This is just a normal string"
      interpolated_string = error_interpolation.interpolate(
          normal_string, ops.get_default_graph())
      self.assertEqual(interpolated_string, normal_string)

  def testOneTagWithAFakeNameResultsInPlaceholders(self):
    with ops.Graph().as_default():
      one_tag_string = "{{node MinusOne}}"
      interpolated_string = error_interpolation.interpolate(
          one_tag_string, ops.get_default_graph())
      self.assertEqual(one_tag_string, interpolated_string)

  def testTwoTagsNoSeps(self):
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      constant_op.constant(3, name="Three")
      two_tags_no_seps = "{{node One}}{{node Three}}"
      interpolated_string = error_interpolation.interpolate(
          two_tags_no_seps, ops.get_default_graph())
      self.assertRegex(
          interpolated_string, r"error_interpolation_test\.py:[0-9]+."
          r"*error_interpolation_test\.py:[0-9]+")

  def testTwoTagsWithSeps(self):
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      constant_op.constant(3, name="Three")
      two_tags_with_seps = ";;;{{node Two}},,,{{node Three}};;;"
      interpolated_string = error_interpolation.interpolate(
          two_tags_with_seps, ops.get_default_graph())
      expected_regex = (r"^;;;.*error_interpolation_test\.py:[0-9]+\) "
                        r",,,.*error_interpolation_test\.py:[0-9]+\) ;;;$")
      self.assertRegex(interpolated_string, expected_regex)

  def testNewLine(self):
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      newline = "\n\n{{node One}}"
      interpolated_string = error_interpolation.interpolate(
          newline, ops.get_default_graph())
      self.assertRegex(interpolated_string,
                       r"error_interpolation_test\.py:[0-9]+.*")


class InputNodesTest(test.TestCase):

  def testNoInputs(self):
    with ops.Graph().as_default():
      one = constant_op.constant(1, name="One")
      two = constant_op.constant(2, name="Two")
      _ = math_ops.add(one, two, name="Three")
      two_tags_with_seps = ";;;{{node One}},,,{{node Two}};;;"
      interpolated_string = error_interpolation.interpolate(
          two_tags_with_seps, ops.get_default_graph())
      expected_regex = (r"^;;;.*error_interpolation_test\.py:[0-9]+\) "
                        r",,,.*error_interpolation_test\.py:[0-9]+\) ;;;$")
      self.assertRegex(interpolated_string, expected_regex)

  def testBasicInputs(self):
    with ops.Graph().as_default():
      one = constant_op.constant(1, name="One")
      two = constant_op.constant(2, name="Two")
      _ = math_ops.add(one, two, name="Three")
      tag = ";;;{{node Three}};;;"
      interpolated_string = error_interpolation.interpolate(
          tag, ops.get_default_graph())
      expected_regex = re.compile(
          r"^;;;.*error_interpolation_test\.py:[0-9]+\) "
          r";;;.*Input.*error_interpolation_test\.py:[0-9]+\)", re.DOTALL)
      self.assertRegex(interpolated_string, expected_regex)


class InterpolateDeviceSummaryTest(test.TestCase):

  def _fancy_device_function(self, unused_op):
    return "/cpu:*"

  def testNodeZeroHasNoDeviceSummaryInfo(self):
    with ops.Graph().as_default():
      self.zero = constant_op.constant([0.0], name="zero")
      message = "{{colocation_node zero}}"
      result = error_interpolation.interpolate(message, ops.get_default_graph())
      self.assertIn("No device assignments were active", result)

  def testNodeOneHasExactlyOneInterpolatedDevice(self):
    with ops.Graph().as_default():
      with ops.device("/cpu"):
        self.one = constant_op.constant([1.0], name="one")
      message = "{{colocation_node one}}"
      result = error_interpolation.interpolate(message, ops.get_default_graph())
      self.assertEqual(2, result.count("tf.device(/cpu)"))

  def testNodeTwoHasTwoInterpolatedDevice(self):
    with ops.Graph().as_default():
      with ops.device("/cpu"):
        with ops.device("/cpu:0"):
          self.two = constant_op.constant([2.0], name="two")
      message = "{{colocation_node two}}"
      result = error_interpolation.interpolate(message, ops.get_default_graph())
      self.assertEqual(2, result.count("tf.device(/cpu)"))
      self.assertEqual(2, result.count("tf.device(/cpu:0)"))

  def testNodeThreeHasFancyFunctionDisplayNameForInterpolatedDevice(self):
    with ops.Graph().as_default():
      with ops.device(self._fancy_device_function):
        self.three = constant_op.constant(3.0, name="three")
      message = "{{colocation_node three}}"
      result = error_interpolation.interpolate(message, ops.get_default_graph())
      num_devices = result.count("tf.device")
      self.assertEqual(2, num_devices)
      name_re = r"_fancy_device_function<.*error_interpolation_test.py, [0-9]+>"
      expected_re = r"with tf.device\(.*%s\)" % name_re
      self.assertRegex(result, expected_re)


class InterpolateColocationSummaryTest(test.TestCase):

  def _set_up_graph(self):
    # Add nodes to the graph for retrieval by name later.
    node_one = constant_op.constant(1, name="One")
    node_two = constant_op.constant(2, name="Two")

    # node_three has one colocation group, obviously.
    with ops.colocate_with(node_one):
      node_three = constant_op.constant(3, name="Three_with_one")

    # node_four has one colocation group even though three is (transitively)
    # colocated with one.
    with ops.colocate_with(node_three):
      constant_op.constant(4, name="Four_with_three")

    # node_five has two colocation groups because one and two are not colocated.
    with ops.colocate_with(node_two):
      with ops.colocate_with(node_one):
        constant_op.constant(5, name="Five_with_one_with_two")

  def testNodeThreeHasColocationInterpolation(self):
    with ops.Graph().as_default():
      self._set_up_graph()
      message = "{{colocation_node Three_with_one}}"
      result = error_interpolation.interpolate(message, ops.get_default_graph())
      self.assertIn("colocate_with(One)", result)

  def testNodeFourHasColocationInterpolationForNodeThreeOnly(self):
    with ops.Graph().as_default():
      self._set_up_graph()
      message = "{{colocation_node Four_with_three}}"
      result = error_interpolation.interpolate(message, ops.get_default_graph())
      self.assertIn("colocate_with(Three_with_one)", result)
      self.assertNotIn(
          "One", result,
          "Node One should not appear in Four_with_three's summary:\n%s" %
          result)

  def testNodeFiveHasColocationInterpolationForNodeOneAndTwo(self):
    with ops.Graph().as_default():
      self._set_up_graph()
      message = "{{colocation_node Five_with_one_with_two}}"
      result = error_interpolation.interpolate(message, ops.get_default_graph())
      self.assertIn("colocate_with(One)", result)
      self.assertIn("colocate_with(Two)", result)

  def testColocationInterpolationForNodeLackingColocation(self):
    with ops.Graph().as_default():
      self._set_up_graph()
      message = "{{colocation_node One}}"
      result = error_interpolation.interpolate(message, ops.get_default_graph())
      self.assertIn("No node-device colocations", result)
      self.assertNotIn("Two", result)


class IsFrameworkFilenameTest(test.TestCase):

  def testAllowsUnitTests(self):
    self.assertFalse(
        error_interpolation._is_framework_filename(
            error_interpolation._FRAMEWORK_PATH_PREFIXES[0] + "foobar_test.py"))

  def testFrameworkPythonFile(self):
    self.assertTrue(
        error_interpolation._is_framework_filename(
            error_interpolation.__file__))

  def testEmbedded(self):
    self.assertTrue(
        error_interpolation._is_framework_filename(
            "<embedded stdlib>/context_lib.py"))


if __name__ == "__main__":
  test.main()
