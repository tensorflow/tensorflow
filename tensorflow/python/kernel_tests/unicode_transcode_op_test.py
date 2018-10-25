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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


# Note: for now only tests for algorithmic converters since no file-based
# converters can be loaded. TODO(gbillock): add ability to include at least
# the ucmcore converters from the conversion data sets.
class UnicodeTranscodeOpTest(test.TestCase, parameterized.TestCase):

  def test_transcode_utf8_simple(self):
    strings = [[b"a", b"abc"], [b"ABC", b"DEF"]]

    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      self.assertAllEqual(values, strings)

      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="ISO-8859-1",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      self.assertAllEqual(values, strings)

      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="US-ASCII",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      self.assertAllEqual(values, strings)

  def test_transcode_utf16_to_utf8(self):
    strings = [b"\x00a\x00b\x20\xAC", b"\xD8\x01\xDC\x37"]  # U+10437
    expected = [s.decode("UTF-16-BE").encode("UTF-8") for s in strings]

    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-16",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      self.assertAllEqual(values, expected)

  def test_transcode_bad_utf8(self):
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=True)
      values = sess.run(outputs)
      self.assertAllEqual(values, b"  ")

      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      self.assertAllEqual(values, b"\x00 ")

  def test_transcode_bad_utf8_with_some_good(self):
    bad_string = b"abc\xffabcdefg"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      self.assertAllEqual(values, b"abc abcdefg")

  def test_transcode_bad_utf8_with_defaults(self):
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string, input_encoding="UTF-8", output_encoding="UTF-8")
      values = sess.run(outputs)
      self.assertAllEqual(values, b"\x00\xef\xbf\xbd")

  def test_transcode_bad_utf8_with_space_replacement(self):
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string, input_encoding="UTF-8", output_encoding="UTF-8",
          replacement_char=ord(" "))
      values = sess.run(outputs)
      self.assertAllEqual(values, b"\x00 ")

  def test_transcode_bad_utf8_with_strict_errors(self):
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="strict")
      with self.assertRaisesOpError(
          "Invalid formatting on input string"):
        sess.run(outputs)

  def test_transcode_bad_utf8_start_with_strict_errors(self):
    bad_string = b"\xffabcd"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="strict")
      with self.assertRaisesOpError(
          "Invalid formatting on input string"):
        sess.run(outputs)

  def test_transcode_bad_utf8_with_elision_of_malformatting(self):
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="ignore")
      values = sess.run(outputs)
      self.assertAllEqual(values, b"\x00")

  def test_transcode_bad_utf8_with_elision_including_control_chars(self):
    bad_string = b"\x00\xff"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string,
          input_encoding="UTF-8",
          output_encoding="UTF-8",
          errors="ignore",
          replace_control_characters=True)
      values = sess.run(outputs)
      self.assertAllEqual(values, b"")

  def test_transcode_bad_utf8_termination_with_defaults(self):
    bad_string = b"a\xf0"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bad_string, input_encoding="UTF-8", output_encoding="UTF-8")
      values = sess.run(outputs)
      self.assertAllEqual(values, b"a\xef\xbf\xbd")   # 0xFFFD

  def test_transcode_utf8_with_replacement_char(self):
    strings = [b"a\xef\xbf\xbd"]
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings, input_encoding="UTF-8", output_encoding="UTF-8",
          errors="strict")
      values = sess.run(outputs)
      self.assertAllEqual(values, [b"a\xef\xbf\xbd"])

      outputs = string_ops.unicode_transcode(
          strings, input_encoding="UTF-8", output_encoding="UTF-8",
          errors="replace", replacement_char=ord("?"))
      values = sess.run(outputs)
      self.assertAllEqual(values, [b"a\xef\xbf\xbd"])

  def test_transcode_utf8_to_utf16(self):
    strings = [b"ab\xe2\x82\xac", b"\xf0\x90\x90\xb7"]  # U+10437
    expected = [s.decode("UTF-8").encode("UTF-16-BE") for s in strings]

    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-8",
          output_encoding="UTF-16-BE",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      print("values=", values)
      self.assertAllEqual(values, expected)

  def test_transcode_utf32_to_utf8(self):
    strings = [
        b"\x00\x00\x00a\x00\x00\x00b\x00\x00\x20\xAC", b"\x00\x01\x04\x37"
    ]  # U+10437
    expected = [s.decode("UTF-32-BE").encode("UTF-8") for s in strings]
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-32",
          output_encoding="UTF-8",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      self.assertAllEqual(values, expected)

  def test_transcode_utf8_to_utf32(self):
    strings = [b"ab\xe2\x82\xac", b"\xf0\x90\x90\xb7"]
    expected = [s.decode("UTF-8").encode("UTF-32-BE") for s in strings]
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="UTF-8",
          output_encoding="UTF-32-BE",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      self.assertAllEqual(values, expected)

  # Documentation in ICU suggests that getNextUChar may produce a different
  # error code if the input sequence contains particular non-coding sequences.
  # This test checks that condition.
  def test_transcode_ascii_with_shift_chars(self):
    strings = [b"\x0e\x0e", b"\x0f\x0f"]
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="US-ASCII",
          output_encoding="UTF-8",
          replacement_char=ord(" "),
          replace_control_characters=False)
      values = sess.run(outputs)
      self.assertAllEqual(values, strings)

  def test_transcode_utf8_with_bom(self):
    bom_string = b"\xef\xbb\xbfabcdefg"
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-8", output_encoding="UTF-8")
      values = sess.run(outputs)
      self.assertAllEqual(values, b"\xef\xbb\xbfabcdefg")  # BOM preserved

      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-8", output_encoding="UTF-16-BE")
      values = sess.run(outputs)
      utf16expected = bom_string.decode("UTF-8").encode("UTF-16-BE")
      self.assertAllEqual(values, utf16expected)

  def test_transcode_utf16_le_be_with_bom(self):
    bom_string = b"\xfe\xff\x00\x61"  # Big-endian BOM with 'a' encoded
    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-16-BE", output_encoding="UTF-8")
      values = sess.run(outputs)
      # BOM is preserved in output
      self.assertAllEqual(values, b"\xef\xbb\xbfa")

      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-16-LE", output_encoding="UTF-8")
      values = sess.run(outputs)
      # mangled BOM and value from (incorrect) LE encoding
      self.assertAllEqual(values, b"\xef\xbf\xbe\xe6\x84\x80")

      bom_string = b"\xff\xfe\x61\x00"  # Little-endian BOM with 'a' encoded
      outputs = string_ops.unicode_transcode(
          bom_string, input_encoding="UTF-16-LE", output_encoding="UTF-8")
      values = sess.run(outputs)
      self.assertAllEqual(values, b"\xef\xbb\xbfa")

  @parameterized.parameters(
      # BOM is stripped if it is used to decide the byte order of the input.
      (b"\xfe\xff\x00*", "UTF-16", b"*"),
      (b"\xff\xfe*\x00", "UTF-16", b"*"),
      # BOM is *not* stripped if it is not used to decide the byte order of
      # the input.
      (b"\xef\xbb\xbf*", "UTF-8", b"\xef\xbb\xbf*"),
      (b"\xfe\xff\x00*", "UTF-16-BE", b"\xef\xbb\xbf*"),
      (b"\xff\xfe*\x00", "UTF-16-LE", b"\xef\xbb\xbf*"),
      # If the encoding is UTF-16, and no BOM is present, then UTF-16-BE
      # is assumed.
      (b"\x00*", "UTF-16", b"*"),
      # BOM is never stripped from any position other than the beginning of
      # the string, for any encoding.
      (b"<\xef\xbb\xbf>", "UTF-8", b"<\xef\xbb\xbf>"),
      (b"\x00<\xfe\xff\x00>", "UTF-16", b"<\xef\xbb\xbf>"),
      (b"\x00<\xfe\xff\x00>", "UTF-16-BE", b"<\xef\xbb\xbf>"),
      (b"<\x00\xff\xfe>\x00", "UTF-16-LE", b"<\xef\xbb\xbf>"),
      (b"\xfe\xff\x00<\xfe\xff\x00>", "UTF-16", b"<\xef\xbb\xbf>"),
      (b"\xff\xfe<\x00\xff\xfe>\x00", "UTF-16", b"<\xef\xbb\xbf>"),
  )
  def test_bom_handling(self, string, input_encoding, expected):
    with self.test_session():
      output = string_ops.unicode_transcode(
          string, input_encoding=input_encoding, output_encoding="UTF-8")
      self.assertAllEqual(output.eval(), expected)

  def test_invalid_encoding_causes_errors(self):
    strings = [[b"a", b"abc"], [b"ABC", b"DEF"]]

    with self.cached_session() as sess:
      outputs = string_ops.unicode_transcode(
          strings,
          input_encoding="invalid",
          output_encoding="UTF-8",
          errors="replace",
          replacement_char=ord(" "),
          replace_control_characters=False)
      with self.assertRaisesOpError(
          "Could not create converter for input encoding: invalid"):
        sess.run(outputs)

    with self.assertRaisesRegexp(ValueError, "Op passed string 'invalid'"):
      with self.cached_session() as sess:
        outputs = string_ops.unicode_transcode(
            strings,
            input_encoding="UTF-8",
            output_encoding="invalid",
            errors="replace",
            replacement_char=ord(" "),
            replace_control_characters=False)
        sess.run(outputs)

  def test_invalid_error_policy_causes_errors(self):
    strings = [[b"a", b"abc"], [b"ABC", b"DEF"]]

    with self.assertRaisesRegexp(
        ValueError, "'invalid' not in: \"strict\", \"replace\", \"ignore\"."):
      with self.cached_session() as sess:
        outputs = string_ops.unicode_transcode(
            strings,
            input_encoding="UTF-8",
            output_encoding="UTF-8",
            errors="invalid",
            replacement_char=ord(" "),
            replace_control_characters=False)
        sess.run(outputs)

  def test_forwarding(self):
    with self.cached_session():
      # Generate an input that is uniquely consumed by the transcode op.
      # This exercises code paths which are optimized for this case
      # (e.g., using forwarding).
      inp = string_ops.substr(
          constant_op.constant([b"AbCdEfG", b"HiJkLmN"], dtypes.string),
          pos=0,
          len=5)
      transcoded = string_ops.unicode_transcode(
          inp, input_encoding="UTF-8", output_encoding="UTF-8")

      self.assertAllEqual([b"AbCdE", b"HiJkL"], transcoded)


if __name__ == "__main__":
  test.main()
