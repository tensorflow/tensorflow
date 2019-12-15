# -*- coding: utf-8 -*-
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for string_upper_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringUpperOpTest(test.TestCase):
  """Test cases for tf.strings.upper."""

  def test_string_upper(self):
    strings = ["Pigs on The Wing", "aNimals"]

    with self.cached_session():
      output = string_ops.string_upper(strings)
      output = self.evaluate(output)
      self.assertAllEqual(output, [b"PIGS ON THE WING", b"ANIMALS"])

  def test_string_upper_2d(self):
    strings = [["pigS on THE wIng", "aniMals"], [" hello ", "\n\tWorld! \r \n"]]

    with self.cached_session():
      output = string_ops.string_upper(strings)
      output = self.evaluate(output)
      self.assertAllEqual(output, [[b"PIGS ON THE WING", b"ANIMALS"],
                                   [b" HELLO ", b"\n\tWORLD! \r \n"]])

  def test_string_upper_unicode(self):
    strings = [["óósschloë"]]
    with self.cached_session():
      output = string_ops.string_upper(strings, encoding="utf-8")
      output = self.evaluate(output)
      # output: "ÓÓSSCHLOË"
      self.assertAllEqual(output, [[b"\xc3\x93\xc3\x93SSCHLO\xc3\x8b"]])


if __name__ == "__main__":
  test.main()
