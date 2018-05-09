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
"""Tests for RegexFullMatch op from string_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class RegexFullMatchOpTest(test.TestCase):

  def testRegexFullMatch(self):
    values = ["abaaba", "abcdabcde"]
    with self.test_session():
      input_vector = constant_op.constant(values, dtypes.string)
      matched = string_ops.regex_full_match(input_vector, "a.*a").eval()
      self.assertAllEqual([True, False], matched)

  def testEmptyMatch(self):
    values = ["abc", "1"]
    with self.test_session():
      input_vector = constant_op.constant(values, dtypes.string)
      matched = string_ops.regex_full_match(input_vector, "").eval()
      self.assertAllEqual([False, False], matched)

  def testInvalidPattern(self):
    values = ["abc", "1"]
    with self.test_session():
      input_vector = constant_op.constant(values, dtypes.string)
      invalid_pattern = "A["
      matched = string_ops.regex_full_match(input_vector, invalid_pattern)
      with self.assertRaisesOpError("Invalid pattern"):
        matched.eval()


if __name__ == "__main__":
  test.main()
