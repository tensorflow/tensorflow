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
"""Tests for RegexReplace op from string_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class RegexReplaceOpTest(test.TestCase):

  def testRemovePrefix(self):
    values = ["a:foo", "a:bar", "a:foo", "b:baz", "b:qux", "ca:b"]
    with self.test_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = string_ops.regex_replace(
          input_vector, "^(a:|b:)", "", replace_global=False).eval()
      self.assertAllEqual([b"foo", b"bar", b"foo", b"baz", b"qux", b"ca:b"],
                          stripped)

  def testRegexReplace(self):
    values = ["aba\naba", "abcdabcde"]
    with self.test_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = string_ops.regex_replace(input_vector, "a.*a", "(\\0)").eval()
      self.assertAllEqual([b"(aba)\n(aba)", b"(abcda)bcde"], stripped)

  def testEmptyMatch(self):
    values = ["abc", "1"]
    with self.test_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = string_ops.regex_replace(input_vector, "", "x").eval()
      self.assertAllEqual([b"xaxbxcx", b"x1x"], stripped)

  def testInvalidPattern(self):
    values = ["abc", "1"]
    with self.test_session():
      input_vector = constant_op.constant(values, dtypes.string)
      invalid_pattern = "A["
      replace = string_ops.regex_replace(input_vector, invalid_pattern, "x")
      with self.assertRaisesOpError("Invalid pattern"):
        replace.eval()

  def testGlobal(self):
    values = ["ababababab", "abcabcabc", ""]
    with self.test_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = string_ops.regex_replace(input_vector, "ab", "abc",
                                          True).eval()
      self.assertAllEqual([b"abcabcabcabcabc", b"abccabccabcc", b""], stripped)


if __name__ == "__main__":
  test.main()
