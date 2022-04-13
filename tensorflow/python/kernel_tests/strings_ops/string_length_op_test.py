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
"""Tests for string_length_op."""

from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringLengthOpTest(test.TestCase):

  def testStringLength(self):
    strings = [[["1", "12"], ["123", "1234"], ["12345", "123456"]]]

    with self.cached_session() as sess:
      lengths = string_ops.string_length(strings)
      values = self.evaluate(lengths)
      self.assertAllEqual(values, [[[1, 2], [3, 4], [5, 6]]])

  @test_util.run_deprecated_v1
  def testUnit(self):
    unicode_strings = [u"H\xc3llo", u"\U0001f604"]
    utf8_strings = [s.encode("utf-8") for s in unicode_strings]
    expected_utf8_byte_lengths = [6, 4]
    expected_utf8_char_lengths = [5, 1]

    with self.session() as sess:
      utf8_byte_lengths = string_ops.string_length(utf8_strings, unit="BYTE")
      utf8_char_lengths = string_ops.string_length(
          utf8_strings, unit="UTF8_CHAR")
      self.assertAllEqual(
          self.evaluate(utf8_byte_lengths), expected_utf8_byte_lengths)
      self.assertAllEqual(
          self.evaluate(utf8_char_lengths), expected_utf8_char_lengths)
      with self.assertRaisesRegex(
          ValueError, "Attr 'unit' of 'StringLength' Op passed string 'XYZ' "
          'not in: "BYTE", "UTF8_CHAR"'):
        string_ops.string_length(utf8_strings, unit="XYZ")

  @test_util.run_deprecated_v1
  def testLegacyPositionalName(self):
    # Code that predates the 'unit' parameter may have used a positional
    # argument for the 'name' parameter.  Check that we don't break such code.
    strings = [[["1", "12"], ["123", "1234"], ["12345", "123456"]]]
    lengths = string_ops.string_length(strings, "some_name")
    with self.session():
      self.assertAllEqual(lengths, [[[1, 2], [3, 4], [5, 6]]])


if __name__ == "__main__":
  test.main()
