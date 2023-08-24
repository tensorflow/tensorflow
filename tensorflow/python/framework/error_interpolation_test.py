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

import collections
import os
import re

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.platform import test


# A mock for ``tf_stack.FrameSummary``.
FrameSummary = collections.namedtuple(
    "StackFrame", ["filename", "lineno", "name", "line"]
)

# TODO(feyu): convert tests to tf function from graph when appropriate.


def _make_frame_with_filename(tb, idx, filename):
  """Return a copy of an existing stack frame with a new filename."""
  frame = tb[idx]
  return FrameSummary(filename, frame.lineno, frame.name, frame.line)


def _modify_op_stack_with_filenames(
    tb, num_user_frames, user_filename, num_inner_tf_frames
):
  """Replace traceback with a new traceback using special filenames."""
  tf_filename = error_interpolation._FRAMEWORK_PATH_PREFIXES[0] + "%d.py"
  user_filename = os.path.join("%d", "my_favorite_file.py")

  num_requested_frames = num_user_frames + num_inner_tf_frames
  num_actual_frames = len(tb)
  num_outer_frames = num_actual_frames - num_requested_frames
  assert num_requested_frames <= num_actual_frames, "Too few real frames."

  # The op's traceback has outermost frame at index 0.
  stack = []
  for idx in range(0, num_outer_frames):
    stack.append(tb[idx])
  for idx in range(len(stack), len(stack) + num_user_frames):
    stack.append(_make_frame_with_filename(tb, idx, user_filename % idx))
  for idx in range(len(stack), len(stack) + num_inner_tf_frames):
    stack.append(_make_frame_with_filename(tb, idx, tf_filename % idx))
  return stack


class ComputeDeviceSummaryFromOpTest(test.TestCase):

  def testCorrectFormatWithActiveDeviceAssignments(self):
    assignments = []
    assignments.append(
        traceable_stack.TraceableObject("/cpu:0", filename="hope.py", lineno=24)
    )
    assignments.append(
        traceable_stack.TraceableObject(
            "/gpu:2", filename="please.py", lineno=42
        )
    )

    summary = error_interpolation._compute_device_summary_from_list(
        "nodename", assignments, prefix="  "
    )

    self.assertIn("nodename", summary)
    self.assertIn("tf.device(/cpu:0)", summary)
    self.assertIn("<hope.py:24>", summary)
    self.assertIn("tf.device(/gpu:2)", summary)
    self.assertIn("<please.py:42>", summary)

  def testCorrectFormatWhenNoColocationsWereActive(self):
    device_assignment_list = []
    summary = error_interpolation._compute_device_summary_from_list(
        "nodename", device_assignment_list, prefix="  "
    )
    self.assertIn("nodename", summary)
    self.assertIn("No device assignments", summary)


class ComputeColocationSummaryFromOpTest(test.TestCase):

  def testCorrectFormatWithActiveColocations(self):
    t_obj_1 = traceable_stack.TraceableObject(
        None, filename="test_1.py", lineno=27
    )
    t_obj_2 = traceable_stack.TraceableObject(
        None, filename="test_2.py", lineno=38
    )
    colocation_dict = {
        "test_node_1": t_obj_1,
        "test_node_2": t_obj_2,
    }
    summary = error_interpolation._compute_colocation_summary_from_dict(
        "node_name", colocation_dict, prefix="  "
    )
    self.assertIn("node_name", summary)
    self.assertIn("colocate_with(test_node_1)", summary)
    self.assertIn("<test_1.py:27>", summary)
    self.assertIn("colocate_with(test_node_2)", summary)
    self.assertIn("<test_2.py:38>", summary)

  def testCorrectFormatWhenNoColocationsWereActive(self):
    colocation_dict = {}
    summary = error_interpolation._compute_colocation_summary_from_dict(
        "node_name", colocation_dict, prefix="  "
    )
    self.assertIn("node_name", summary)
    self.assertIn("No node-device colocations", summary)


class InterpolateFilenamesAndLineNumbersTest(test.TestCase):

  def testFindIndexOfDefiningFrameForOp(self):
    with ops.Graph().as_default():
      local_op = constant_op.constant(42).op
      user_filename = "hope.py"
      modified_tb = _modify_op_stack_with_filenames(
          local_op.traceback,
          num_user_frames=3,
          user_filename=user_filename,
          num_inner_tf_frames=5,
      )
      idx = error_interpolation._find_index_of_defining_frame(modified_tb)
      # Expected frame is 6th from the end because there are 5 inner frames with
      # TF filenames.
      expected_frame = len(modified_tb) - 6
      self.assertEqual(expected_frame, idx)

  def testFindIndexOfDefiningFrameForOpReturnsZeroOnError(self):
    with ops.Graph().as_default():
      local_op = constant_op.constant(43).op
      # Ensure all frames look like TF frames.
      modified_tb = _modify_op_stack_with_filenames(
          local_op.traceback[:7],  # Truncate stack to known length.
          num_user_frames=0,
          user_filename="user_file.py",
          num_inner_tf_frames=7,
      )
      idx = error_interpolation._find_index_of_defining_frame(modified_tb)
      self.assertEqual(0, idx)

  def testNothingToDo(self):
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      normal_string = "This is just a normal string"
      interpolated_string = error_interpolation.interpolate_graph(
          normal_string, ops.get_default_graph()
      )
      self.assertIn(normal_string, interpolated_string)

  def testOneTagWithAFakeNameResultsInPlaceholders(self):
    with ops.Graph().as_default():
      one_tag_string = "{{node MinusOne}}"
      interpolated_string = error_interpolation.interpolate_graph(
          one_tag_string, ops.get_default_graph()
      )
      self.assertIn(one_tag_string, interpolated_string)

  def testOneTagWithAFakeFunctionTag(self):
    defined_at = r"defined at.*error_interpolation_test\.py"
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      one_tag_with_a_fake_function_tag = "{{function_node fake}}{{node One}}"
      interpolated_string = error_interpolation.interpolate_graph(
          one_tag_with_a_fake_function_tag, ops.get_default_graph()
      )
      # Fragments the expression to avoid matching the pattern itself.
      expected_regex = re.compile(rf"node 'One'.*{defined_at}", re.DOTALL)
      self.assertRegex(interpolated_string, expected_regex)
      self.assertNotIn("function_node", interpolated_string)
      self.assertNotIn("node 'Two'", interpolated_string)

  def testTwoTagsNoSeps(self):
    defined_at = r"defined at.*error_interpolation_test\.py"
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      constant_op.constant(3, name="Three")
      two_tags_no_seps = "{{node One}}{{node Three}}"
      interpolated_string = error_interpolation.interpolate_graph(
          two_tags_no_seps, ops.get_default_graph()
      )
      # Fragments the expression to avoid matching the pattern itself.
      expected_regex = re.compile(
          rf"node 'One'.*{defined_at}.*node 'Three'.*{defined_at}", re.DOTALL
      )
      self.assertRegex(interpolated_string, expected_regex)

  def testTwoTagsWithSeps(self):
    defined_at = r"defined at.*error_interpolation_test\.py"
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      constant_op.constant(3, name="Three")
      two_tags_with_seps = ";;;{{node Two}},,,{{node Three}};;;"
      interpolated_string = error_interpolation.interpolate_graph(
          two_tags_with_seps, ops.get_default_graph()
      )
      # Fragments the expression to avoid matching the pattern itself.
      expected_regex = re.compile(
          rf"node 'Two'.*{defined_at}.*node 'Three'.*{defined_at}", re.DOTALL
      )
      self.assertRegex(interpolated_string, expected_regex)

  def testNewLine(self):
    defined_at = r"defined at.*error_interpolation_test\.py"
    with ops.Graph().as_default():
      constant_op.constant(1, name="One")
      constant_op.constant(2, name="Two")
      newline = "\n\n;;;{{node One}};;;"
      interpolated_string = error_interpolation.interpolate_graph(
          newline, ops.get_default_graph()
      )
      expected_regex = re.compile(rf"node 'One'.*{defined_at}", re.DOTALL)
      self.assertRegex(interpolated_string, expected_regex)


class OperationDefinedAtTraceTest(test.TestCase):

  @test_util.run_v2_only
  def testSimpleCall(self):
    @def_function.function
    def func():
      x = constant_op.constant([[1, 2, 3]])
      y = script_ops.eager_py_func(lambda: [[1, 2, 3]], (), dtypes.int32)
      return math_ops.matmul(x, y)

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        re.compile(
            r"defined at.*" r"in testSimpleCall.*" r"in func", re.DOTALL
        ),
    ):
      func()

  @test_util.run_v2_only
  def testNestedCall(self):
    def inner():
      x = constant_op.constant([[1, 2, 3]])
      y = script_ops.eager_py_func(lambda: [[1, 2, 3]], (), dtypes.int32)
      return math_ops.matmul(x, y)

    @def_function.function
    def func():
      return inner()

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        re.compile(
            r"defined at.*" r"in testNestedCall.*" r"in func.*" r"in inner",
            re.DOTALL,
        ),
    ):
      func()

  @test_util.run_v2_only
  def testAssert(self):
    @def_function.function
    def func():
      control_flow_assert.Assert(False, [False])
      return

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        re.compile(r"defined at.*" r"in testAssert.*" r"in func", re.DOTALL),
    ):
      func()

  @test_util.run_v2_only
  def testControlFlow(self):
    @def_function.function
    def func():
      if constant_op.constant(False):
        return constant_op.constant(1)

      else:
        x = constant_op.constant([[1, 2, 3]])
        y = script_ops.eager_py_func(lambda: [[1, 2, 3]], (), dtypes.int32)
        return math_ops.matmul(x, y)

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        re.compile(
            r"defined at.*" r"in testControlFlow.*" r"in func", re.DOTALL
        ),
    ):
      func()


class IsFrameworkFilenameTest(test.TestCase):

  def testAllowsUnitTests(self):
    self.assertFalse(
        error_interpolation._is_framework_filename(
            error_interpolation._FRAMEWORK_PATH_PREFIXES[0] + "foobar_test.py"
        )
    )

  def testFrameworkPythonFile(self):
    self.assertTrue(
        error_interpolation._is_framework_filename(error_interpolation.__file__)
    )

  def testEmbedded(self):
    self.assertTrue(
        error_interpolation._is_framework_filename(
            "<embedded stdlib>/context_lib.py"
        )
    )


if __name__ == "__main__":
  test.main()
