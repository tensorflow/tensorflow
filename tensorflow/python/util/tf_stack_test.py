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
      self.assertEqual(frame, expected)

  def testFormatStack(self):
    stack, expected_stack = extract_stack()
    self.assertEqual(
        traceback.format_list(stack),
        traceback.format_list(expected_stack))


def extract_stack(limit=None):
  convert = tf_stack.convert_stack
  # Both defined on the same line to produce identical stacks.
  return convert(tf_stack.extract_stack(limit)), traceback.extract_stack(limit)


if __name__ == "__main__":
  test.main()
