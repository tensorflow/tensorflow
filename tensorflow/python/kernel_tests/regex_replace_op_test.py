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

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


@parameterized.parameters(
    (gen_string_ops.regex_replace),
    (gen_string_ops.static_regex_replace))
class RegexReplaceOpVariantsTest(test.TestCase, parameterized.TestCase):

  @test_util.run_deprecated_v1
  def testForwarding(self, op):
    with self.cached_session():
      # Generate an input that is uniquely consumed by the regex op.
      # This exercises code paths which are optimized for this case
      # (e.g., using forwarding).
      inp = string_ops.substr(
          constant_op.constant(["AbCdEfG",
                                "HiJkLmN"], dtypes.string),
          pos=0,
          len=5)
      stripped = op(inp, "\\p{Ll}", ".")
      self.assertAllEqual([b"A.C.E", b"H.J.L"], stripped)

  @test_util.run_deprecated_v1
  def testRemovePrefix(self, op):
    values = ["a:foo", "a:bar", "a:foo", "b:baz", "b:qux", "ca:b"]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = op(input_vector, "^(a:|b:)", "", replace_global=False)
      self.assertAllEqual([b"foo", b"bar", b"foo", b"baz", b"qux", b"ca:b"],
                          stripped)

  @test_util.run_deprecated_v1
  def testRegexReplace(self, op):
    values = ["aba\naba", "abcdabcde"]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = op(input_vector, "a.*a", "(\\0)")
      self.assertAllEqual([b"(aba)\n(aba)", b"(abcda)bcde"], stripped)

  @test_util.run_deprecated_v1
  def testEmptyMatch(self, op):
    values = ["abc", "1"]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = op(input_vector, "", "x")
      self.assertAllEqual([b"xaxbxcx", b"x1x"], stripped)

  @test_util.run_deprecated_v1
  def testInvalidPattern(self, op):
    values = ["abc", "1"]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      invalid_pattern = "A["
      replace = op(input_vector, invalid_pattern, "x")
      with self.assertRaisesOpError("Invalid pattern"):
        self.evaluate(replace)

  @test_util.run_deprecated_v1
  def testGlobal(self, op):
    values = ["ababababab", "abcabcabc", ""]
    with self.cached_session():
      input_vector = constant_op.constant(values, dtypes.string)
      stripped = op(input_vector, "ab", "abc", True)
      self.assertAllEqual([b"abcabcabcabcabc", b"abccabccabcc", b""], stripped)


def as_string(s):
  return s


def as_tensor(s):
  return constant_op.constant(s, dtypes.string)


class RegexReplaceTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (as_string, as_tensor),
      (as_tensor, as_string),
      (as_tensor, as_tensor))
  @test_util.run_deprecated_v1
  def testRegexReplaceDelegation(self, pattern_fn, rewrite_fn):
    with self.cached_session():
      input_vector = constant_op.constant("foo", dtypes.string)
      pattern = pattern_fn("[a-z]")
      replace = rewrite_fn(".")
      op = string_ops.regex_replace(input_vector, pattern, replace)
      self.assertTrue(op.name.startswith("RegexReplace"))

  @test_util.run_deprecated_v1
  def testStaticRegexReplaceDelegation(self):
    with self.cached_session():
      input_vector = constant_op.constant("foo", dtypes.string)
      pattern = "[a-z]"
      replace = "."
      op = string_ops.regex_replace(input_vector, pattern, replace)
      self.assertTrue(op.name.startswith("StaticRegexReplace"))

if __name__ == "__main__":
  test.main()
