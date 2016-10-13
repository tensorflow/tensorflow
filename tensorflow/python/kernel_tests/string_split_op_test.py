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
"""Tests for string_split_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class StringSplitOpTest(tf.test.TestCase):

  def testStringSplit(self):
    strings = ["pigs on the wing", "animals"]

    with self.test_session() as sess:
      tokens = tf.string_split(strings)
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]])
      self.assertAllEqual(values, [b"pigs", b"on", b"the", b"wing", b"animals"])
      self.assertAllEqual(shape, [2, 4])

  def testStringSplitEmptyDelimiter(self):
    strings = ["hello", "hola"]

    with self.test_session() as sess:
      tokens = tf.string_split(strings, delimiter="")
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
                                    [1, 0], [1, 1], [1, 2], [1, 3]])
      self.assertAllEqual(values, [b"h", b"e", b"l", b"l", b"o", b"h", b"o",
                                   b"l", b"a"])
      self.assertAllEqual(shape, [2, 5])

  def testStringSplitEmptyToken(self):
    strings = [" hello ", "", "world "]

    with self.test_session() as sess:
      tokens = tf.string_split(strings)
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [2, 0]])
      self.assertAllEqual(values, [b"hello", b"world"])
      self.assertAllEqual(shape, [3, 1])

  def testStringSplitWithDelimiter(self):
    strings = ["hello|world", "hello world"]

    with self.test_session() as sess:
      self.assertRaises(
          ValueError, tf.string_split, strings, delimiter="delimiter")

      self.assertRaises(
          ValueError, tf.string_split, strings, delimiter=["|", ""])

      self.assertRaises(ValueError, tf.string_split, strings, delimiter=["a"])

      tokens = tf.string_split(strings, delimiter="|")
      indices, values, shape = sess.run(tokens)
      self.assertAllEqual(indices, [[0, 0], [0, 1], [1, 0]])
      self.assertAllEqual(values, [b"hello", b"world", b"hello world"])
      self.assertAllEqual(shape, [2, 2])

  def testStringSplitWithDelimiterTensor(self):
    strings = ["hello|world", "hello world"]

    with self.test_session() as sess:
      delimiter = tf.placeholder(tf.string)

      tokens = tf.string_split(strings, delimiter=delimiter)

      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a", "b"]})
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: ["a"]})
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(tokens, feed_dict={delimiter: "abc"})
      indices, values, shape = sess.run(tokens, feed_dict={delimiter: "|"})

      self.assertAllEqual(indices, [[0, 0], [0, 1], [1, 0]])
      self.assertAllEqual(values, [b"hello", b"world", b"hello world"])
      self.assertAllEqual(shape, [2, 2])


if __name__ == "__main__":
  tf.test.main()
