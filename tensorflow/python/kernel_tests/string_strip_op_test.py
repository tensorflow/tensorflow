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
"""Tests for string_strip_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringStripOpTest(test.TestCase):
  """ Test cases for tf.string_strip."""

  def test_string_strip(self):
    strings = ["pigs on the wing", "animals"]

    with self.cached_session() as sess:
      output = string_ops.string_strip(strings)
      output = sess.run(output)
      self.assertAllEqual(output, [b"pigs on the wing", b"animals"])

  def test_string_strip_2d(self):
    strings = [["pigs on the wing", "animals"],
               [" hello ", "\n\tworld \r \n"]]

    with self.cached_session() as sess:
      output = string_ops.string_strip(strings)
      output = sess.run(output)
      self.assertAllEqual(output, [[b"pigs on the wing", b"animals"],
                                   [b"hello", b"world"]])

  def test_string_strip_with_empty_strings(self):
    strings = [" hello ", "", "world ", " \t \r \n "]

    with self.cached_session() as sess:
      output = string_ops.string_strip(strings)
      output = sess.run(output)
      self.assertAllEqual(output, [b"hello", b"", b"world", b""])


if __name__ == "__main__":
  test.main()
