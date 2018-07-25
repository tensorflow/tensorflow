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

import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import ops
from tensorflow.python.framework import traceable_stack
from tensorflow.python.platform import test
from tensorflow.python.util import tf_stack


def _make_frame_with_filename(op, idx, filename):
  """Return a copy of an existing stack frame with a new filename."""
  stack_frame = list(op._traceback[idx])
  stack_frame[tf_stack.TB_FILENAME] = filename
  return tuple(stack_frame)


def _modify_op_stack_with_filenames(op, num_user_frames, user_filename,
                                    num_inner_tf_frames):
  """Replace op._traceback with a new traceback using special filenames."""
  tf_filename = "%d" + error_interpolation._BAD_FILE_SUBSTRINGS[0]
  user_filename = os.path.join("%d", "my_favorite_file.py")

  num_requested_frames = num_user_frames + num_inner_tf_frames
  num_actual_frames = len(op._traceback)
  num_outer_frames = num_actual_frames - num_requested_frames
  assert num_requested_frames <= num_actual_frames, "Too few real frames."

  # The op's traceback has outermost frame at index 0.
  stack = []
  for idx in range(0, num_outer_frames):
    stack.append(op._traceback[idx])
  for idx in range(len(stack), len(stack)+num_user_frames):
    stack.append(_make_frame_with_filename(op, idx, user_filename % idx))
  for idx in range(len(stack), len(stack)+num_inner_tf_frames):
    stack.append(_make_frame_with_filename(op, idx, tf_filename % idx))
  op._traceback = stack


def assert_node_in_colocation_summary(test_obj, colocation_summary_string,
                                      name, filename="", lineno=""):
  lineno = str(lineno)
  name_phrase = "colocate_with(%s)" % name
  for term in [name_phrase, filename, lineno]:
    test_obj.assertIn(term, colocation_summary_string)
  test_obj.assertNotIn("loc:@", colocation_summary_string)


class ComputeColocationSummaryFromOpTest(test.TestCase):

  def testCorrectFormatWithActiveColocations(self):
    t_obj_1 = traceable_stack.TraceableObject(None,
                                              filename="test_1.py",
                                              lineno=27)
    t_obj_2 = traceable_stack.TraceableObject(None,
                                              filename="test_2.py",
                                              lineno=38)
    colocation_dict = {
        "test_node_1": t_obj_1,
        "test_node_2": t_obj_2,
    }
    summary = error_interpolation._compute_colocation_summary_from_dict(
        colocation_dict, prefix="  ")
    assert_node_in_colocation_summary(self,
                                      summary,
                                      name="test_node_1",
                                      filename="test_1.py",
                                      lineno=27)
    assert_node_in_colocation_summary(self, summary,
                                      name="test_node_2",
                                      filename="test_2.py",
                                      lineno=38)

  def testCorrectFormatWhenNoColocationsWereActive(self):
    colocation_dict = {}
    summary = error_interpolation._compute_colocation_summary_from_dict(
        colocation_dict, prefix="  ")
    self.assertIn("No node-device colocations", summary)


class InterpolateTest(test.TestCase):

  def setUp(self):
    # Add nodes to the graph for retrieval by name later.
    constant_op.constant(1, name="One")
    constant_op.constant(2, name="Two")
    three = constant_op.constant(3, name="Three")
    self.graph = three.graph

    # Change the list of bad file substrings so that constant_op.py is chosen
    # as the defining stack frame for constant_op.constant ops.
    self.old_bad_strings = error_interpolation._BAD_FILE_SUBSTRINGS
    error_interpolation._BAD_FILE_SUBSTRINGS = [
        "%sops.py" % os.sep,
        "%sutil" % os.sep,
    ]

  def tearDown(self):
    error_interpolation._BAD_FILE_SUBSTRINGS = self.old_bad_strings

  def testFindIndexOfDefiningFrameForOp(self):
    local_op = constant_op.constant(42).op
    user_filename = "hope.py"
    _modify_op_stack_with_filenames(local_op,
                                    num_user_frames=3,
                                    user_filename=user_filename,
                                    num_inner_tf_frames=5)
    idx = error_interpolation._find_index_of_defining_frame_for_op(local_op)
    # Expected frame is 6th from the end because there are 5 inner frames witih
    # TF filenames.
    expected_frame = len(local_op._traceback) - 6
    self.assertEqual(expected_frame, idx)

  def testFindIndexOfDefiningFrameForOpReturnsZeroOnError(self):
    local_op = constant_op.constant(43).op
    # Truncate stack to known length.
    local_op._traceback = local_op._traceback[:7]
    # Ensure all frames look like TF frames.
    _modify_op_stack_with_filenames(local_op,
                                    num_user_frames=0,
                                    user_filename="user_file.py",
                                    num_inner_tf_frames=7)
    idx = error_interpolation._find_index_of_defining_frame_for_op(local_op)
    self.assertEqual(0, idx)

  def testNothingToDo(self):
    normal_string = "This is just a normal string"
    interpolated_string = error_interpolation.interpolate(normal_string,
                                                          self.graph)
    self.assertEqual(interpolated_string, normal_string)

  def testOneTag(self):
    one_tag_string = "^^node:Two:${file}^^"
    interpolated_string = error_interpolation.interpolate(one_tag_string,
                                                          self.graph)
    self.assertTrue(interpolated_string.endswith("constant_op.py"),
                    "interpolated_string '%s' did not end with constant_op.py"
                    % interpolated_string)

  def testOneTagWithAFakeNameResultsInPlaceholders(self):
    one_tag_string = "^^node:MinusOne:${file}^^"
    interpolated_string = error_interpolation.interpolate(one_tag_string,
                                                          self.graph)
    self.assertEqual(interpolated_string, "<NA>")

  def testTwoTagsNoSeps(self):
    two_tags_no_seps = "^^node:One:${file}^^^^node:Three:${line}^^"
    interpolated_string = error_interpolation.interpolate(two_tags_no_seps,
                                                          self.graph)
    self.assertRegexpMatches(interpolated_string, "constant_op.py[0-9]+")

  def testTwoTagsWithSeps(self):
    two_tags_with_seps = ";;;^^node:Two:${file}^^,,,^^node:Three:${line}^^;;;"
    interpolated_string = error_interpolation.interpolate(two_tags_with_seps,
                                                          self.graph)
    expected_regex = "^;;;.*constant_op.py,,,[0-9]*;;;$"
    self.assertRegexpMatches(interpolated_string, expected_regex)


class InterpolateColocationSummaryTest(test.TestCase):

  def setUp(self):
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

    self.graph = node_three.graph

  def testNodeThreeHasColocationInterpolation(self):
    message = "^^node:Three_with_one:${colocations}^^"
    result = error_interpolation.interpolate(message, self.graph)
    assert_node_in_colocation_summary(self, result, name="One")

  def testNodeFourHasColocationInterpolationForNodeThreeOnly(self):
    message = "^^node:Four_with_three:${colocations}^^"
    result = error_interpolation.interpolate(message, self.graph)
    assert_node_in_colocation_summary(self, result, name="Three_with_one")
    self.assertNotIn(
        "One", result,
        "Node One should not appear in Four_with_three's summary:\n%s"
        % result)

  def testNodeFiveHasColocationInterpolationForNodeOneAndTwo(self):
    message = "^^node:Five_with_one_with_two:${colocations}^^"
    result = error_interpolation.interpolate(message, self.graph)
    assert_node_in_colocation_summary(self, result, name="One")
    assert_node_in_colocation_summary(self, result, name="Two")

  def testColocationInterpolationForNodeLackingColocation(self):
    message = "^^node:One:${colocations}^^"
    result = error_interpolation.interpolate(message, self.graph)
    self.assertIn("No node-device colocations", result)
    self.assertNotIn("One", result)
    self.assertNotIn("Two", result)


if __name__ == "__main__":
  test.main()
