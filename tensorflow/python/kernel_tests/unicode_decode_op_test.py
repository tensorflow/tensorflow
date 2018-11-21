# -*- coding: utf-8 -*-
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
"""Tests for unicode_decode and unicode_decode_with_splits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl as errors
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.platform import test


# Account for python2 and python3 execution of the test.
def codepoint(s):
  if isinstance(s, bytes):
    return ord(s.decode("utf-8"))
  elif isinstance(s, str):
    return ord(s)


class UnicodeDecodeTest(test.TestCase):

  def testBatchDecode(self):
    text = constant_op.constant(
        ["仅今年前", "分享介面終於迎來更新"])
    row_splits, utf8_text, offsets = gen_string_ops.unicode_decode_with_offsets(
        text, "utf-8")

    with self.test_session():
      self.assertAllEqual([
          codepoint("仅"),
          codepoint("今"),
          codepoint("年"),
          codepoint("前"),
          codepoint("分"),
          codepoint("享"),
          codepoint("介"),
          codepoint("面"),
          codepoint("終"),
          codepoint("於"),
          codepoint("迎"),
          codepoint("來"),
          codepoint("更"),
          codepoint("新")
      ],
                          self.evaluate(utf8_text).tolist())
      self.assertAllEqual([0, 4, 14], self.evaluate(row_splits).tolist())
      self.assertAllEqual([0, 3, 6, 9, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
                          self.evaluate(offsets).tolist())

  def testBasicDecodeWithOffset(self):
    text = constant_op.constant(["仅今年前"])
    row_splits, utf8_text, starts = gen_string_ops.unicode_decode_with_offsets(
        text, "utf-8")

    with self.test_session():
      self.assertAllEqual([
          codepoint("仅"),
          codepoint("今"),
          codepoint("年"),
          codepoint("前"),
      ],
                          self.evaluate(utf8_text).tolist())
      self.assertAllEqual(self.evaluate(row_splits).tolist(), [0, 4])
      self.assertAllEqual(self.evaluate(starts).tolist(), [0, 3, 6, 9])

  def testStrictError(self):
    text = constant_op.constant([b"\xFEED"])
    _, error, _ = gen_string_ops.unicode_decode_with_offsets(
        text, "utf-8", errors="strict")

    with self.assertRaises(errors.InvalidArgumentError):
      with self.test_session():
        self.evaluate(error)

  def testReplaceOnError(self):
    text = constant_op.constant([b"\xFE"])

    _, utf8_text, _ = gen_string_ops.unicode_decode_with_offsets(
        text, "utf-8", errors="replace")

    with self.test_session():
      self.assertAllEqual(self.evaluate(utf8_text).tolist(), [65533])

  def testBadReplacementChar(self):
    text = constant_op.constant([b"\xFE"])
    _, error, _ = gen_string_ops.unicode_decode_with_offsets(
        text, "utf-8", errors="replace", replacement_char=11141111)

    with self.assertRaises(errors.InvalidArgumentError):
      with self.test_session():
        self.evaluate(error)

  def testIgnoreOnError(self):
    text = constant_op.constant([b"\xFEhello"])

    _, utf8_text, _ = gen_string_ops.unicode_decode_with_offsets(
        text, "utf-8", errors="ignore")

    with self.test_session():
      self.assertAllEqual(self.evaluate(utf8_text).tolist(), [
          codepoint("h"),
          codepoint("e"),
          codepoint("l"),
          codepoint("l"),
          codepoint("o")
      ])

  def testBadErrorPolicy(self):
    text = constant_op.constant(["hippopotamus"])

    with self.assertRaises(ValueError):
      _, _, _ = gen_string_ops.unicode_decode_with_offsets(
          text, "utf-8", errors="oranguatan")

  def testReplaceControlChars(self):
    text = constant_op.constant(["\x02仅今年前"])
    row_splits, utf8_text, _ = gen_string_ops.unicode_decode_with_offsets(
        text, "utf-8", replace_control_characters=True)

    with self.test_session():
      self.assertAllEqual([
          65533,
          codepoint("仅"),
          codepoint("今"),
          codepoint("年"),
          codepoint("前"),
      ],
                          self.evaluate(utf8_text).tolist())
      self.assertAllEqual([0, 5], self.evaluate(row_splits).tolist())


if __name__ == "__main__":
  test.main()
