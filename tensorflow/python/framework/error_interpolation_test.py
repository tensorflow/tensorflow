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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import error_interpolation
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
  user_filename = "%d/my_favorite_file.py"

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
    error_interpolation._BAD_FILE_SUBSTRINGS = ["/ops.py", "/util"]

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
    self.assertTrue(interpolated_string.endswith("op.py"),
                    "interpolated_string '%s' did not end with op.py"
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
    self.assertRegexpMatches(interpolated_string, "op.py[0-9]+")

  def testTwoTagsWithSeps(self):
    two_tags_with_seps = ";;;^^node:Two:${file}^^,,,^^node:Three:${line}^^;;;"
    interpolated_string = error_interpolation.interpolate(two_tags_with_seps,
                                                          self.graph)
    expected_regex = "^;;;.*op.py,,,[0-9]*;;;$"
    self.assertRegexpMatches(interpolated_string, expected_regex)


if __name__ == "__main__":
  test.main()
