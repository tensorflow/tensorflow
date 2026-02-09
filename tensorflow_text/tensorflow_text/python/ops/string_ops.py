# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tensorflow operations for UTF8 strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import string_ops


def _unichr(codepoint):
  try:
    return unichr(codepoint)
  except NameError:
    return chr(codepoint)


# pylint: disable=redefined-builtin
def coerce_to_structurally_valid_utf8(input,
                                      replacement_char=_unichr(65533),
                                      name=None):
  r"""Coerce UTF-8 input strings to structurally valid UTF-8.

  Any bytes which cause the input string to be invalid UTF-8 are substituted
  with the provided replacement character codepoint (default 65533). If you plan
  on overriding the default, use a single byte replacement character codepoint
  to preserve alignment to the source input string.

  In this example, the character \xDEB2 is an invalid UTF-8 bit sequence; the
  call to `coerce_to_structurally_valid_utf8` replaces it with \xef\xbf\xbd,
  which is the default replacement character encoding.
  >>> input_data = ["A", b"\xDEB2", "C"]
  >>> coerce_to_structurally_valid_utf8(input_data)
  <tf.Tensor: shape=(3,), dtype=string,
              numpy=array([b'A', b'\xef\xbf\xbdB2', b'C'], dtype=object)>

  Args:
    input: UTF-8 string tensor to coerce to valid UTF-8.
    replacement_char: The replacement character to be used in place of any
        invalid byte in the input. Any valid Unicode character may be used. The
        default value is the default Unicode replacement character which is
        0xFFFD (or U+65533). Note that passing a replacement character
        expressible in 1 byte, such as ' ' or '?', will preserve string
        alignment to the source since individual invalid bytes will be replaced
        with a 1-byte replacement. (optional)
    name: A name for the operation (optional).

  Returns:
    A tensor of type string with the same shape as the input.
  """
  return string_ops.unicode_transcode(
      input,
      input_encoding='UTF-8',
      output_encoding='UTF-8',
      errors='replace',
      replacement_char=ord(replacement_char),
      name=name)
