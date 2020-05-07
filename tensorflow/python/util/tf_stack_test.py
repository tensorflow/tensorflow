# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for functions used to extract and analyze stacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback

from tensorflow.python.platform import test
from tensorflow.python.util import tf_stack


class TFStackTest(test.TestCase):

  def testLimit(self):
    self.assertEmpty(tf_stack.extract_stack(limit=0))
    self.assertLen(tf_stack.extract_stack(limit=1), 1)
    self.assertEqual(
        len(tf_stack.extract_stack(limit=-1)),
        len(tf_stack.extract_stack()))

  def testConsistencyWithTraceback(self):
    stack, expected_stack = extract_stack()
    for frame, expected in zip(stack, expected_stack):
      self.assertEqual(convert_stack_frame(frame), expected)

  def testFormatStack(self):
    stack, expected_stack = extract_stack()
    self.assertEqual(
        traceback.format_list(stack),
        traceback.format_list(expected_stack))

  def testFrameSummaryEquality(self):
    frame0, frame1 = tf_stack.extract_stack(limit=2)
    self.assertNotEqual(frame0, frame1)
    self.assertEqual(frame0, frame0)

    another_frame0, _ = tf_stack.extract_stack(limit=2)
    self.assertEqual(frame0, another_frame0)


def extract_stack(limit=None):
  # Both defined on the same line to produce identical stacks.
  return tf_stack.extract_stack(limit), traceback.extract_stack(limit)


def convert_stack_frame(frame):
  """Converts a TF stack frame into Python's."""
  # TODO(mihaimaruseac): Remove except case when dropping suport for py2
  try:
    return traceback.FrameSummary(
        frame.filename, frame.lineno, frame.name, line=frame.line)
  except AttributeError:
    # On Python < 3.5 (i.e., Python2), we don't have traceback.FrameSummary so
    # we don't need to match with that class. Instead, just a tuple is enough.
    return tuple(frame)


if __name__ == "__main__":
  test.main()
